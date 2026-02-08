# AyurAssist

AI-powered clinical decision support that bridges modern medical terminology (SNOMED CT, UMLS) with traditional Ayurvedic knowledge using the WHO International Terminologies for Ayurveda (ITA).

## Architecture

AyurAssist runs on [Modal](https://modal.com) as a two-tier serverless system that separates GPU-heavy LLM inference from lightweight CPU orchestration.

```
Browser (static site on GitHub Pages)
  |
  |─ GET /warmup ──> [CPU Container]  ─── warmup.remote() ──> [GPU Container]
  |                  (fires on page load,                      (starts loading
  |                   returns immediately)                      AyurParam model)
  |
  |  ... user types for 20-30s ...
  |
  |─ POST / ──────> [CPU Container: ASGI + NER + CSV + UMLS]
                      |
                      ├─ 1. NER extraction (scispacy en_core_sci_lg, CPU)
                      ├─ 2. Entity resolution: exact CSV match, then UMLS ICD-10
                      ├─ 3. CSV lookup (226 WHO ITA conditions)
                      ├─ 4. generate.remote() ──> [GPU Container: transformers]
                      |                            AyurParam generates treatment
                      └─ 5. Assemble response with clinical codes + treatment
```

### CPU Container (ASGI + orchestration)

Handles everything that doesn't need a GPU:

- **Biomedical NER** -- [scispacy](https://allenai.github.io/scispacy/) `en_core_sci_lg` runs on CPU. Extracts biomedical entities from patient narratives. All entities are labeled `ENTITY` (generic), so keyword selection uses a two-stage resolution strategy (see [Entity Resolution](#entity-resolution) below).
- **UMLS mapping** -- Two-step API lookup: search for keyword to get a UMLS CUI, then query the CUI's atoms endpoint filtered by `SNOMEDCT_US` to get the actual SNOMED code. When used for entity resolution, the search is restricted to `ICD10CM` (ICD-10 Clinical Modification) to ensure only diseases and clinical conditions are matched.
- **WHO ITA CSV lookup** -- `ayurveda_snomed_mapping.csv` contains 226 Ayurvedic conditions from the WHO International Terminologies for Ayurveda, mapped to SNOMED codes. Lookup by SNOMED code first, then fuzzy text match on condition name.
- **FastAPI ASGI app** -- Serves the `/` (analyze) and `/warmup` endpoints. Uses FastAPI's `lifespan` to load NER and CSV once at startup.

Stays warm for 5 minutes after the last request (`scaledown_window=300`).

### GPU Container (transformers)

Runs only the AyurParam LLM (`bharatgenai/AyurParam`) via HuggingFace `transformers` with `device_map="auto"` for native generation (AyurParam's custom tokenizer is not compatible with vLLM). Receives a prompt enriched with SNOMED codes and WHO ITA context, returns text containing:

- Condition name (Sanskrit and English)
- Dosha involvement
- Nidana (causes), Rupa (symptoms)
- Ottamooli (single remedies) with dosage and preparation
- Classical formulations with textual references
- Pathya (dietary advice)
- Vihara (lifestyle) and Yoga recommendations
- Warning signs and disclaimer

Scales to zero when idle. Shuts down after 1 minute of inactivity (`scaledown_window=60`).

### Warmup Strategy

Cold-starting the GPU container takes 30-60 seconds (model loading). To hide this latency:

1. The frontend fires `GET /warmup` on page load (fire-and-forget).
2. The CPU container starts, loads NER model, and simultaneously kicks off `LLMEngine.warmup.remote()` in a background asyncio task.
3. By the time the user finishes typing (20-30 seconds), the GPU container is typically already warm.

### Entity Resolution

scispacy's `en_core_sci_lg` labels all entities as generic `ENTITY` (no Disease/Symptom distinction), so free-text input like *"I feel like I've been hit by a bus, my teeth were chattering, I had soft-serve from the dairy"* produces entities like `bus`, `teeth`, `dairy` alongside real clinical terms. The keyword selection strategy filters these out:

1. **Exact CSV match** -- Each entity is checked against the WHO ITA term lookup (exact, case-insensitive). If a match is found, it is used immediately. This is fast and guarantees an Ayurvedic disease term.
2. **UMLS ICD-10 lookup** -- If no CSV match, all entities are queried against UMLS in parallel, restricted to `ICD10CM` (ICD-10 Clinical Modification). ICD-10 only contains diseases and clinical conditions, so non-medical terms like `bus` or `dairy` are filtered out. Results are ranked: entities whose SNOMED code exists in the CSV are preferred, then those with any SNOMED code, then longest entity text (more specific medical term).
3. **Fuzzy CSV fallback** -- The selected keyword is fuzzy-matched against CSV terms (threshold 0.6) to find the closest Ayurvedic condition.

### Data Flow

```
Patient narrative
    -> NER: scispacy extracts biomedical entities
    -> Entity resolution:
         1. exact CSV match on each entity
         2. UMLS ICD-10 lookup (diseases/conditions only), ranked by CSV match
    -> CSV:  SNOMED code -> WHO ITA match (Sanskrit name, ITA ID, description)
    -> LLM:  6 focused questions sent to AyurParam -> treatment responses
    -> Response: assembled treatment info with clinical codes
```

## Project Structure

```
AyurAssist/
├── main.py                        # Modal backend (GPU + CPU tiers)
├── config.py                      # All tuneable constants (no secrets)
├── ayurveda_snomed_mapping.csv    # 226 WHO ITA conditions -> SNOMED
├── docs/                          # Frontend (GitHub Pages)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## Configuration

All tuneable constants are in `config.py`. Edit this file to change model IDs, timeouts, GPU types, etc.

### Modal Secrets

Secrets are **not** stored in code. They are managed through [Modal's secret manager](https://modal.com/docs/guide/secrets).
You need to create two secrets in your Modal dashboard:

| Modal secret name      | Required env vars | How to get it |
|------------------------|-------------------|---------------|
| `huggingface-secret`   | `HF_TOKEN`        | [HuggingFace tokens](https://huggingface.co/settings/tokens) |
| `my-umls-secret`       | `UMLS_API_KEY`    | [UMLS license](https://uts.nlm.nih.gov/uts/signup-login) |

Create them with the Modal CLI:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxx
modal secret create my-umls-secret UMLS_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

The secret **names** referenced in `config.py` (`MODAL_SECRET_HUGGINGFACE`, `MODAL_SECRET_UMLS`) must match what you created above. If you use different names, update `config.py` accordingly.

### Frontend

Update `API_BASE` in `docs/app.js` to match your Modal deployment URL:

```javascript
const API_BASE = 'https://<your-modal-username>--ayurparam-service-fastapi-app.modal.run';
```

## Deployment

### Prerequisites

- Python 3.11+
- [Modal CLI](https://modal.com/docs/guide/getting-started) installed and authenticated
- Modal secrets created (see above)
- A HuggingFace account with access to `bharatgenai/AyurParam`

### Deploy to Modal

```bash
modal deploy main.py
```

This builds two container images (CPU and GPU), deploys the ASGI web endpoint, and outputs your deployment URL.

### Local development

```bash
modal serve main.py
```

This runs the app locally with hot-reload. The terminal will show the temporary URL.

## API

### `GET /warmup`

Triggers GPU container startup in the background. Returns immediately.

**Response:**
```json
{"status": "warming"}
```

### `POST /`

Analyzes a patient narrative and returns Ayurvedic treatment recommendations.

**Request:**
```json
{"text": "patient complains of severe headache and nausea"}
```

**Response:**
```json
{
  "input_text": "patient complains of severe headache and nausea",
  "clinical_entities": [
    {"word": "headache", "score": 0.98, "entity_group": "Symptom"},
    {"word": "nausea", "score": 0.95, "entity_group": "Symptom"}
  ],
  "umls_cui": "C0018681",
  "snomed_code": "25064002",
  "csv_match": {
    "ita_id": "ITA-5.32.1",
    "ayurveda_term": "Headache",
    "sanskrit_iast": "shirorogah",
    "sanskrit": "...",
    "description": "..."
  },
  "results": [{
    "ayurveda_term": "Headache",
    "snomed_code": "25064002",
    "treatment_info": {
      "condition_name": "Shiroroga",
      "sanskrit_name": "shirorogah",
      "brief_description": "...",
      "dosha_involvement": "...",
      "nidana_causes": ["..."],
      "rupa_symptoms": ["..."],
      "ottamooli_single_remedies": [{"medicine_name": "...", "dosage": "..."}],
      "classical_formulations": [{"name": "...", "reference_text": "..."}],
      "pathya_dietary_advice": {"foods_to_favor": ["..."], "foods_to_avoid": ["..."]},
      "vihara_lifestyle": ["..."],
      "yoga_exercises": ["..."],
      "warning_signs": ["..."],
      "disclaimer": "..."
    }
  }]
}
```

## Cost

| Component | When running | When idle |
|-----------|-------------|-----------|
| GPU container (T4) | ~$0.76/hr | $0 (scales to zero) |
| CPU container | ~$0.04/hr | $0 (scales to zero after 5 min) |
| Modal volume | ~$0.07/GB/mo | ~$0.07/GB/mo |

With the warmup strategy, the GPU is only active during user sessions, not 24/7.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `vllm` | High-throughput LLM serving with PagedAttention |
| `transformers` | NER pipeline (`d4data/biomedical-ner-all`) |
| `modal` | Serverless GPU/CPU containers |
| `fastapi` | ASGI web framework |
