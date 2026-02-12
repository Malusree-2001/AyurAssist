import modal
import json
import os
from contextlib import asynccontextmanager

from config import (
    MODAL_APP_NAME, MODAL_VOLUME_NAME,
    MODAL_SECRET_HUGGINGFACE, MODAL_SECRET_UMLS,
    GPU_TYPE, GPU_TIMEOUT, GPU_MIN_CONTAINERS, GPU_SCALEDOWN_WINDOW,
    CPU_TIMEOUT, CPU_SCALEDOWN_WINDOW,
    LLM_MODEL_ID, LLM_MAX_MODEL_LEN, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_DTYPE,
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
            trust_remote_code=True,
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
# Helper functions (no CSV; NER + UMLS + AyurParam only)
# ===================================================================

def _lookup_umls_snomed(api_key, keyword):
    """keyword -> (CUI, SNOMED code) using UMLS search + SNOMED atoms."""
    import requests

    umls_cui = "N/A"
    snomed_code = "N/A"
    if not api_key:
        return umls_cui, snomed_code

    # Step 1: keyword -> CUI
    try:
        params = {"string": keyword, "apiKey": api_key, "returnIdType": "concept"}
        r = requests.get(UMLS_SEARCH_URL, params=params, timeout=UMLS_REQUEST_TIMEOUT)
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
        r2 = requests.get(
            UMLS_ATOMS_URL_TEMPLATE.format(cui=umls_cui),
            params={
                "apiKey": api_key,
                "sabs": "SNOMEDCT_US",
                "ttys": "PT",
                "pageSize": 10,
            },
            timeout=UMLS_REQUEST_TIMEOUT,
        )
        if r2.status_code == 200:
            atoms = r2.json().get("result", [])
            if atoms:
                code_uri = atoms[0].get("code", "")
                snomed_code = code_uri.rsplit("/", 1)[-1] if "/" in code_uri else code_uri
    except Exception as e:
        print(f"UMLS atoms (SNOMED) error: {e}")

    return umls_cui, snomed_code


def _lookup_icd10cm_from_cui(api_key, umls_cui):
    """CUI -> ICD10CM code via UMLS atoms."""
    import requests

    icd_code = "N/A"
    if umls_cui == "N/A" or not api_key:
        return icd_code

    try:
        r = requests.get(
            UMLS_ATOMS_URL_TEMPLATE.format(cui=umls_cui),
            params={
                "apiKey": api_key,
                "sabs": "ICD10CM",
                "pageSize": 10,
            },
            timeout=UMLS_REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            atoms = r.json().get("result", [])
            if atoms:
                code_uri = atoms[0].get("code", "")
                icd_code = code_uri.rsplit("/", 1)[-1] if "/" in code_uri else code_uri
    except Exception as e:
        print(f"UMLS atoms (ICD10CM) error: {e}")

    return icd_code


def _build_questions(condition, sanskrit, description):
    """Build the 6 focused questions (AyurParam notebook style)."""
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


def _build_treatment_from_responses(responses, condition, sanskrit):
    """Assemble the 6 text responses into a structured treatment dict."""
    return {
        "condition_name": condition,
        "sanskrit_name": sanskrit or "",
        "brief_description": responses[0][:500] if responses[0] else "",
        "dosha_involvement": "",
        "nidana_causes": [],
        "rupa_symptoms": [],
        "ottamooli_single_remedies": [],
        "classical_formulations": [],
        "pathya_dietary_advice": {
            "foods_to_favor": [],
            "foods_to_avoid": [],
            "specific_dietary_rules": "",
        },
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


async def _get_modern_diagnosis(llm, entities):
    """Use AyurParam to pick a single modern (Western) diagnosis from symptoms."""
    symptom_list = ", ".join(entities) if entities else "no clear entities"
    prompt = f"""<user>
Patient presents with these symptoms: {symptom_list}

From a modern Western medical perspective, what is the single most likely diagnosis?
Answer with ONLY ONE standard English disease name (1-4 words) that would appear in ICD-10 or SNOMED CT, no 'or', 'and', commas, or explanations.

Examples:
- severe unilateral headache with nausea and vomiting -> Migraine
- chronic joint pain with swelling in small joints of hands -> Rheumatoid arthritis
- frequent urination, excessive thirst, weight loss -> Diabetes mellitus
- burning urination with fever and flank pain -> Acute pyelonephritis

Now give just the single modern diagnosis:
<assistant>"""
    resp = await llm.generate.remote.aio(prompt)
    text = resp.strip()
    for sep in [",", " or ", " and ", ";", "/", "|"]:
        if sep in text:
            text = text.split(sep)[0]
    return text.strip()


async def _modern_to_ayurvedic(llm, modern_dx):
    """Map a modern diagnosis to a single Ayurvedic disease term."""
    prompt = f"""<user>
Map this modern medical diagnosis to the most accurate single Ayurvedic disease name (roga):

Diagnosis: {modern_dx}

Answer with ONLY ONE Ayurvedic term (Sanskrit/IAST or standard Ayurvedic English term) that best corresponds to this diagnosis, no explanations.

Examples:
- Migraine -> Ardhavabhedaka
- Rheumatoid arthritis -> Amavata
- Diabetes mellitus -> Madhumeha
- Irritable bowel syndrome -> Grahani
- Chronic sinusitis -> Peenasa

Now map: {modern_dx}
<assistant>"""
    resp = await llm.generate.remote.aio(prompt)
    text = resp.strip()
    for sep in [",", " or ", " and ", ";", "/", "|"]:
        if sep in text:
            text = text.split(sep)[0]
    return text.strip()


# --- ASGI lifespan: loads NER once, kicks off GPU warmup in parallel ---

@asynccontextmanager
async def lifespan(web_app):
    import asyncio
    import spacy

    async def _gpu_warmup():
        try:
            await LLMEngine().warmup.remote.aio()
            print("GPU container warm.")
        except Exception as e:
            print(f"GPU warmup failed (will cold-start on first request): {e}")

    asyncio.create_task(_gpu_warmup())

    def _load_ner():
        return spacy.load(NER_MODEL_NAME)

    web_app.state.ner = await asyncio.to_thread(_load_ner)
    web_app.state.umls_api_key = os.environ.get("UMLS_API_KEY", "")

    print("CPU engine ready (NER + UMLS, no CSV).")
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
    import asyncio

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
        async def _wake():
            try:
                await LLMEngine().warmup.remote.aio()
            except Exception as e:
                print(f"Warmup ping error: {e}")

        asyncio.create_task(_wake())
        return {"status": "warming"}

    @web.post("/")
    async def analyze(request: Request):
        try:
            body = await request.json()
            user_input = body.get("text", "").strip()
            if not user_input:
                raise HTTPException(status_code=400, detail="Missing 'text' field")

            st = request.app.state

            # 1. NER → entities (symptoms)
            entities = []
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

            entity_words = [e["word"] for e in entities] if entities else [user_input]

            llm = LLMEngine()

            # 2. AyurParam → modern diagnosis (single English term)
            try:
                modern_diagnosis = await _get_modern_diagnosis(llm, entity_words)
            except Exception as e:
                print(f"Modern diagnosis error: {e}")
                modern_diagnosis = ""

            modern_term_for_umls = (modern_diagnosis or entity_words[0] or user_input).strip()
            print(f"Modern diagnosis for UMLS: '{modern_term_for_umls}'")

            # 3. UMLS → CUI, SNOMED, ICD10CM using modern diagnosis
            umls_cui, snomed_code = _lookup_umls_snomed(
                st.umls_api_key, modern_term_for_umls
            )
            icd10cm_code = _lookup_icd10cm_from_cui(st.umls_api_key, umls_cui)

            # 4. Modern diagnosis → Ayurvedic diagnosis
            try:
                ayurvedic_diagnosis = await _modern_to_ayurvedic(llm, modern_term_for_umls)
            except Exception as e:
                print(f"Modern→Ayurvedic mapping error: {e}")
                ayurvedic_diagnosis = modern_term_for_umls

            # 5. AyurParam → 6-question treatment using Ayurvedic diagnosis
            questions = _build_questions(
                condition=ayurvedic_diagnosis,
                sanskrit="",
                description="",
            )

            responses = []
            for q in questions:
                prompt = f"<user> {q} <assistant>"
                try:
                    resp = await llm.generate.remote.aio(prompt)
                    responses.append(resp.strip())
                except Exception as e:
                    print(f"LLM error for question: {e}")
                    responses.append("")

            treatment = _build_treatment_from_responses(
                responses,
                condition=ayurvedic_diagnosis,
                sanskrit="",
            )

            return {
                "input_text": user_input,
                "clinical_entities": entities if entities else [{"word": user_input, "score": 1.0}],
                "modern_diagnosis": modern_diagnosis,
                "umls_cui": umls_cui,
                "snomed_code": snomed_code,
                "icd10cm_code": icd10cm_code,
                "ayurvedic_diagnosis": ayurvedic_diagnosis,
                "treatment_info": treatment,
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"Request error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return web
