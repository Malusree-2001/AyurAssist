import modal
import json
import csv
import os
from difflib import SequenceMatcher
from contextlib import asynccontextmanager

from config import (
    MODAL_APP_NAME, MODAL_VOLUME_NAME,
    MODAL_SECRET_HUGGINGFACE, MODAL_SECRET_UMLS,
    GPU_TYPE, GPU_TIMEOUT, GPU_MIN_CONTAINERS, GPU_SCALEDOWN_WINDOW,
    CPU_TIMEOUT, CPU_SCALEDOWN_WINDOW,
    LLM_MODEL_ID, LLM_MAX_MODEL_LEN, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_REPETITION_PENALTY, LLM_DTYPE,
    NER_MODEL_NAME,
    CSV_SOURCE_PATH, CSV_CONTAINER_PATH, MODEL_CACHE_DIR, VOLUME_MOUNT_PATH,
    UMLS_SEARCH_URL, UMLS_ATOMS_URL_TEMPLATE, UMLS_REQUEST_TIMEOUT,
    FUZZY_MATCH_THRESHOLD, PYTHON_VERSION,
)

app = modal.App(MODAL_APP_NAME)

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
cpu_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "scispacy==0.5.5",
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz",
        "fastapi[standard]==0.109.0",
        "requests==2.31.0",
    )
    .add_local_file(CSV_SOURCE_PATH, CSV_CONTAINER_PATH)
    .add_local_file("config.py", "/root/config.py")
)

gpu_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "torch==2.1.0",
        "numpy==1.24.3",
        "transformers==4.46.0",
        "accelerate==0.34.0",
        "huggingface_hub==0.25.0",
        # Needed because Modal loads the full module in every container
        "fastapi[standard]==0.109.0",
        "requests==2.31.0",
    )
    .add_local_file("config.py", "/root/config.py")
)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


# ===================================================================
# GPU Tier: Transformers engine for AyurParam (only LLM inference)
# ===================================================================

@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    timeout=GPU_TIMEOUT,
    min_containers=GPU_MIN_CONTAINERS,
    scaledown_window=GPU_SCALEDOWN_WINDOW,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name(MODAL_SECRET_HUGGINGFACE)]
)
class LLMEngine:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import login

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_ID,
            use_fast=False,
            trust_remote_code=False,
            cache_dir=MODEL_CACHE_DIR,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
            device_map="auto",
        )
        print("LLM engine ready (transformers).")

    @modal.method()
    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        # Cap generation so prompt + output stays within 2048 context
        max_new = min(LLM_MAX_TOKENS, LLM_MAX_MODEL_LEN - prompt_len)
        if max_new < 50:
            print(f"Warning: prompt too long ({prompt_len} tokens), only {max_new} tokens left for generation")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                top_k=LLM_TOP_K,
                repetition_penalty=LLM_REPETITION_PENALTY,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        new_tokens = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @modal.method()
    def warmup(self) -> dict:
        return {"status": "ready"}


# ===================================================================
# CPU Tier: NER + CSV + UMLS orchestration, served as ASGI app
# ===================================================================

# --- Helper functions (pure, no class state) ---

def _load_csv_lookup(csv_path):
    snomed_lookup = {}
    term_lookup = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Match_Status") == "Unmatched":
                    continue
                snomed = row.get("SNOMED_Code", "").strip()
                if snomed:
                    snomed_lookup[snomed] = row
                for field in ("Search_Term_Used", "Ayurveda_Term"):
                    term = row.get(field, "").strip().lower()
                    if term:
                        term_lookup[term] = row
    except Exception as e:
        print(f"CSV load error: {e}")
    return snomed_lookup, term_lookup


def _fuzzy_csv_lookup(term_lookup, keyword):
    key = keyword.strip().lower()
    if key in term_lookup:
        return term_lookup[key]
    best_match = None
    best_score = 0.0
    for term, row in term_lookup.items():
        score = SequenceMatcher(None, key, term).ratio()
        if score > best_score:
            best_score = score
            best_match = row
    return best_match if best_score >= FUZZY_MATCH_THRESHOLD else None


def _lookup_umls(api_key, keyword, search_sabs=None):
    """Two-step UMLS: search -> CUI, then CUI atoms -> SNOMED code.
    If search_sabs is set (e.g. 'ICD10CM'), only match terms from that source."""
    import requests

    umls_cui = "N/A"
    snomed_code = "N/A"
    if not api_key:
        return umls_cui, snomed_code

    # Step 1: keyword -> CUI
    try:
        params = {"string": keyword, "apiKey": api_key, "returnIdType": "concept"}
        if search_sabs:
            params["sabs"] = search_sabs
        r = requests.get(
            UMLS_SEARCH_URL,
            params=params,
            timeout=UMLS_REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            results = r.json().get("result", {}).get("results", [])
            if results:
                umls_cui = results[0].get("ui", "N/A")
    except Exception as e:
        print(f"UMLS search error: {e}")
        return umls_cui, snomed_code

    if umls_cui == "N/A":
        return umls_cui, snomed_code

    # Step 2: CUI -> SNOMED code via atoms
    try:
        r = requests.get(
            UMLS_ATOMS_URL_TEMPLATE.format(cui=umls_cui),
            params={"apiKey": api_key, "sabs": "SNOMEDCT_US", "ttys": "PT", "pageSize": 5},
            timeout=UMLS_REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            atoms = r.json().get("result", [])
            if atoms:
                code_uri = atoms[0].get("code", "")
                snomed_code = code_uri.rsplit("/", 1)[-1] if "/" in code_uri else code_uri
    except Exception as e:
        print(f"UMLS atoms error: {e}")

    return umls_cui, snomed_code


def _build_questions(condition, sanskrit, description):
    """Build the 6 focused questions matching the notebook's multi-question approach."""
    sanskrit_part = f" ({sanskrit})" if sanskrit else ""
    return [
        (
            f"Explain {condition}{sanskrit_part} in Ayurveda in 2-3 sentences. "
            f"Which doshas and srotas are involved? List the main nidana (causes)."
        ),
        (
            f"What are the purvarupa (prodromal symptoms) and rupa (main symptoms) "
            f"of {condition}{sanskrit_part} in Ayurveda? List them clearly."
        ),
        (
            f"List 3 single drug remedies (dravya/ottamooli) for {condition}{sanskrit_part}. "
            f"For each give: name, Sanskrit name, part used, preparation, dosage, and duration."
        ),
        (
            f"List 2-3 classical Ayurvedic compound formulations (yogas) for {condition}{sanskrit_part}. "
            f"Give name, form, dosage, and reference text."
        ),
        (
            f"For {condition}{sanskrit_part}: "
            f"1) Recommended panchakarma treatment. "
            f"2) Pathya - foods to eat and avoid. "
            f"3) Vihara - lifestyle advice. "
            f"4) Recommended yoga and pranayama."
        ),
        (
            f"For {condition}{sanskrit_part}: "
            f"1) What is the prognosis - Sadhya, Yapya, or Asadhya? "
            f"2) What is the modern medical correlation? "
            f"3) What are the danger signs needing immediate attention?"
        ),
    ]


def _build_treatment_from_responses(responses, condition, sanskrit, csv_data):
    """Assemble the 6 text responses into a structured treatment dict."""
    return {
        "condition_name": (csv_data.get("Ayurveda_Term") if csv_data else None) or condition,
        "sanskrit_name": (csv_data.get("Sanskrit_IAST") if csv_data else None) or sanskrit,
        "brief_description": responses[0][:500] if responses[0] else "",
        "dosha_involvement": "",
        "nidana_causes": [],
        "rupa_symptoms": [],
        "ottamooli_single_remedies": [],
        "classical_formulations": [],
        "pathya_dietary_advice": {"foods_to_favor": [], "foods_to_avoid": [], "specific_dietary_rules": ""},
        "vihara_lifestyle": [],
        "yoga_exercises": [],
        "prognosis": "",
        "warning_signs": [],
        "disclaimer": "This information is for educational purposes only. Consult a qualified Ayurvedic practitioner.",
        "ayurparam_responses": {
            "overview_dosha_causes": responses[0],
            "symptoms": responses[1],
            "single_drug_remedies": responses[2],
            "classical_formulations": responses[3],
            "panchakarma_diet_lifestyle_yoga": responses[4],
            "prognosis_modern_warnings": responses[5],
        },
    }


# --- ASGI lifespan: loads NER + CSV once, kicks off GPU warmup in parallel ---

@asynccontextmanager
async def lifespan(web_app):
    import asyncio

    # Fire GPU warmup immediately so it runs in parallel with NER loading
    async def _gpu_warmup():
        try:
            await LLMEngine().warmup.remote.aio()
            print("GPU container warm.")
        except Exception as e:
            print(f"GPU warmup failed (will cold-start on first request): {e}")

    asyncio.create_task(_gpu_warmup())

    # Load NER model in a thread so it doesn't block the event loop
    # (allows the GPU warmup coroutine above to make progress)
    import spacy

    def _load_ner():
        return spacy.load(NER_MODEL_NAME)

    web_app.state.ner = await asyncio.to_thread(_load_ner)

    # CSV and config (fast)
    web_app.state.snomed_lookup, web_app.state.term_lookup = _load_csv_lookup(
        CSV_CONTAINER_PATH
    )
    web_app.state.umls_api_key = os.environ.get("UMLS_API_KEY", "")

    print("CPU engine ready.")
    yield


# --- ASGI app function ---

@app.function(
    image=cpu_image,
    timeout=CPU_TIMEOUT,
    scaledown_window=CPU_SCALEDOWN_WINDOW,
    secrets=[
        modal.Secret.from_name(MODAL_SECRET_UMLS),
        modal.Secret.from_name(MODAL_SECRET_HUGGINGFACE),
    ],
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    web = FastAPI(lifespan=lifespan)
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/warmup")
    async def warmup():
        """Called by the frontend on page load. Returns immediately;
        the GPU container spins up in the background."""
        import asyncio

        async def _wake():
            try:
                await LLMEngine().warmup.remote.aio()
            except Exception as e:
                print(f"Warmup ping error: {e}")

        asyncio.create_task(_wake())
        return {"status": "warming"}

    @web.post("/")
    async def analyze(request: Request):
        import asyncio

        try:
            body = await request.json()
            user_input = body.get("text", "").strip()
            if not user_input:
                raise HTTPException(status_code=400, detail="Missing 'text' field")

            st = request.app.state

            # 1. NER (CPU, in-process -- run in thread to keep event loop free)
            entities = []
            keyword = user_input
            try:
                doc = await asyncio.to_thread(st.ner, user_input)
                for ent in doc.ents:
                    entities.append({
                        "word": ent.text,
                        "score": 1.0,
                        "entity_group": ent.label_,
                    })
            except Exception as e:
                print(f"NER error: {e}")

            # 2. Pick keyword: CSV match first, then UMLS (diseases only)
            umls_cui, snomed_code = "N/A", "N/A"
            csv_data = None

            if entities:
                unique_words = list(dict.fromkeys(e["word"] for e in entities))

                # 2a. Exact CSV match on each entity (no fuzzy — avoids
                #      garbage like "fat" fuzzy-matching short CSV terms)
                for word in unique_words:
                    key = word.strip().lower()
                    if key in st.term_lookup:
                        keyword = word
                        csv_data = st.term_lookup[key]
                        snomed_code = csv_data.get("SNOMED_Code", "N/A").strip() or "N/A"
                        break

                # 2b. No exact CSV hit — try UMLS restricted to ICD-10
                #      (covers diseases AND conditions/symptoms via R-codes)
                if csv_data is None:
                    umls_results = await asyncio.gather(*(
                        asyncio.to_thread(
                            _lookup_umls, st.umls_api_key, w, search_sabs="ICD10CM"
                        )
                        for w in unique_words
                    ))
                    # Rank: SNOMED in our CSV > has SNOMED > CUI only;
                    # among ties prefer longer entity (more specific term)
                    candidates = []
                    for word, (cui, snomed) in zip(unique_words, umls_results):
                        if cui == "N/A":
                            continue
                        in_csv = (snomed != "N/A"
                                  and snomed in st.snomed_lookup)
                        has_snomed = snomed != "N/A"
                        candidates.append(
                            (in_csv, has_snomed, len(word), word, cui, snomed)
                        )
                    if candidates:
                        candidates.sort(reverse=True)
                        _, _, _, keyword, umls_cui, snomed_code = candidates[0]

            # 3. CSV lookup from SNOMED or fuzzy keyword match
            if csv_data is None:
                if snomed_code != "N/A":
                    csv_data = st.snomed_lookup.get(snomed_code)
                if csv_data is None:
                    csv_data = _fuzzy_csv_lookup(st.term_lookup, keyword)

            # 4. LLM generation — 6 focused questions (matching notebook approach)
            sanskrit = (csv_data.get("Sanskrit_IAST", "") if csv_data else "") or ""
            description = (csv_data.get("Description", "") if csv_data else "") or ""
            questions = _build_questions(keyword, sanskrit, description)

            llm = LLMEngine()
            responses = []
            for q in questions:
                prompt = f"<user> {q} <assistant>"
                try:
                    resp = await llm.generate.remote.aio(prompt)
                    responses.append(resp.strip())
                except Exception as e:
                    print(f"LLM error for question: {e}")
                    responses.append("")

            # 5. Assemble treatment from responses
            treatment = _build_treatment_from_responses(
                responses, keyword, sanskrit, csv_data
            )

            return {
                "input_text": user_input,
                "clinical_entities": entities if entities else [{"word": keyword, "score": 1.0}],
                "umls_cui": umls_cui,
                "snomed_code": snomed_code,
                "csv_match": {
                    "ita_id": csv_data.get("ITA_ID", ""),
                    "ayurveda_term": csv_data.get("Ayurveda_Term", ""),
                    "sanskrit_iast": csv_data.get("Sanskrit_IAST", ""),
                    "sanskrit": csv_data.get("Sanskrit", ""),
                    "description": csv_data.get("Description", ""),
                } if csv_data else None,
                "results": [{
                    "ayurveda_term": (csv_data.get("Ayurveda_Term") if csv_data else None)
                                     or treatment.get("condition_name", keyword),
                    "snomed_code": snomed_code,
                    "treatment_info": treatment,
                }],
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"Request error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return web
