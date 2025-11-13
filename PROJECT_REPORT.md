# AyurGenixAI — Project Report

**Purpose:** Provide a concise, presentation-ready report describing the project's architecture, tech stack, styling & UI choices, data & AI components, deployment and operational needs. This document is intended to help you present the project as an AI-enabled health assistant (Ayurvedic domain), including how it uses machine learning / language models and engineering choices.

---

## 1. Executive Summary

AyurGenixAI is a bilingual (English/Hindi) AI-assisted Ayurveda assistant that maps user symptoms to likely diseases and recommends treatment guidance (herbs, formulations, diet & lifestyle, yoga, and medical interventions). The app uses a lightweight, reproducible stack built around a Python backend and a Streamlit front-end for rapid prototyping. AI features include natural-language symptom parsing, fuzzy matching for half-typed Hindi inputs, language normalization (Devanagari support), and optional LLM-based expansions for richer guidance.

This project demonstrates practical AI usage (NLP + retrieval), robust data handling for bilingual content, and standard engineering practices (containerization, CI/CD, tests, observability).

---

## 2. Key Features to Highlight (for demos / slides)

- Symptom → Disease mapping with Hindi and English support, preserving Devanagari.
- Corpus-backed treatment recommendations (herbs, formulation, lifestyle) with dataset row previews.
- Fuzzy and normalization-based matching for partial / misspelled inputs.
- Optional LLM enhancement: rewrite user queries, expand synonyms, or create patient-facing summaries.
- Lightweight, single-file Streamlit UI for fast demos; can be productionized behind an API.

---

## 3. Architecture Overview

- Frontend: Streamlit (single-file `app.py`) for UI, forms, and interactive pages.
- Data Layer: CSV datasets (canonical corpus `AyurGenixAI_Dataset.csv`), symptom→disease mapping CSV(s), and JSON synonym maps.
- Backend/Logic: Python with pandas for data processing, custom normalization and matching code, lightweight fuzzy matching (difflib / rapidfuzz optional), and optional connectors to LLM APIs.
- AI Components:
  - Text normalization & tokenization (Unicode NFKC, whitespace collapse, punctuation removal for compact matching).
  - Fuzzy matching / nearest-string matching for noisy Hindi inputs.
  - Optional LLM (OpenAI / self-hosted Llama-family) for query expansion, explanation, and patient-profile summarization.
- Optional Persistence: vector DB (e.g., Milvus, Pinecone, Weaviate) for semantic search if embeddings-based retrieval is added.

Diagram (conceptual):

User (browser) -> Streamlit UI -> Matching pipeline (normalize -> map -> corpus lookup -> rank) -> Treatment Recommendations -> (Optional) LLM enhancer -> UI

---

## 4. Tech Stack (recommended list to show)

- Language: Python 3.10+ or 3.11
- Web / UI: Streamlit
- Data processing: pandas
- Fuzzy matching / NLP:
  - `difflib` / `rapidfuzz` for fuzzy string matching
  - `regex` / `unicodedata` for normalization
  - `spaCy` or `transformers` (optional) for richer NLP
- LLMs / Embeddings (optional): OpenAI API, Anthropic, or self-hosted Llama/Alpaca / HuggingFace models
- Vector DB (optional for semantic search): Pinecone, Milvus, Weaviate
- Packaging & Env: `pip`, `venv` / `conda`, `requirements.txt`
- Containerization: Docker
- CI/CD: GitHub Actions (tests, lint, build, push to registry)
- Monitoring & Logging: Sentry (errors), Prometheus + Grafana (metrics)
- Testing: `pytest` for unit tests, `tox` for matrix testing
- Formatting & Linting: `black`, `isort`, `ruff`/`flake8`

---

## 5. Styling & UI Design Choices (what to say when presenting)

- Minimal, responsive UI using Streamlit's layout primitives; optional custom CSS injected via components for branded colors.
- Fonts: use `Noto Sans Devanagari` or `Noto Sans` for consistent Hindi display.
- Color & Theme: Calm, Ayurvedic palette (greens, warm neutrals); Streamlit theme config for consistent look.
- Accessibility: Large, readable fonts, good contrast, explicit language toggles for Hindi/English.
- UI components:
  - Sidebar for global controls (language toggle, corpus toggle).
  - Main form for user symptom input + results area for ranked diseases and matched row preview.
  - Clear CTA buttons (e.g., "Consult the Oracle", "➡️ Send to Sacred Remedies").
- Data previews: On result pages show the matched dataset row (disease canonical name + Hindi display name) to demonstrate provenance.

---

## 6. Data & AI Implementation Details (how it "uses AI")

- Input normalization pipeline:
  - Unicode normalization (NFKC) to unify Devanagari forms.
  - Lowercasing / case folding for Latin text.
  - Punctuation stripping for compact matching while preserving Devanagari characters.
- Matching strategies shown in the app:
  - Exact normalized-match against mapping keys and corpus columns.
  - Substring / compact-substring match (useful for partial typing).
  - Fuzzy matching (difflib.get_close_matches or `rapidfuzz.fuzz`) for misspellings.
  - Canonicalization: map display names to canonical `Disease` used in dataset for treatment lookup.
- Optional LLM uses to highlight during demo:
  - Query rewriting for robust matching (e.g., expand synonyms, translate Hindi to canonical form).
  - Generate patient-facing plain-language summaries from structured treatment columns.
  - Suggest multi-step care plans or personalized recommendations using simple prompt templates.
- Embeddings-based search (if elevated):
  - Build an embeddings index for disease descriptions/treatments and use semantic similarity to find relevant entries for ambiguous queries.

---

## 7. Security, Privacy & Compliance

- Treat patient text as potentially sensitive; avoid logging PII or store it encrypted.
- If collecting patient data, add explicit consent and data-retention policies.
- Use environment variables (not hard-coded keys) for API keys and secrets.
- For production LLM usage, monitor and sanitize model outputs before showing medical suggestions — include disclaimers and fallback to human review.

---

## 8. Deployment & Infra (how to present production readiness)

- Quick local run (development):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run "app.py"
```

- Docker (example high-level steps):
  - Create a `Dockerfile` that installs Python, copies app + requirements, and runs `streamlit run app.py` behind a simple web server proxy (or use `streamlit run` and expose the port).
  - Use multi-stage builds if adding model binaries.
- Production hosting options:
  - Container registry + Kubernetes (AKS/EKS/GKE) for scale.
  - Platform services: Azure App Service, AWS Elastic Beanstalk, Render, Railway, or Railway/Heroku for small demos.
  - For LLM inference: use hosted APIs (OpenAI) or a model-serving GPU instance (AWS/GCP/Azure VMs) or managed LLM services.
- Operational concerns:
  - Autoscaling based on request load.
  - Persistent storage for logs and dataset versioning.

---

## 9. Testing & Quality Assurance

- Unit tests for matching pipeline (`pytest`): normalization, mapping lookup, corpus filtering.
- Integration tests for end-to-end symptom → disease → treatment flow (simulate inputs in both Hindi and English).
- UI tests (optional) with Playwright/Selenium for critical flows.

---

## 10. Requirements & Sample `requirements.txt`

A minimal, demonstrative `requirements.txt` to show during presentation:

```
pandas>=1.5
streamlit>=1.20
rapidfuzz>=2.0   # optional, for better fuzzy matching
python-dotenv
pytest
black
requests
```

If LLM features are shown, include `openai` or `transformers` + `sentence_transformers`.

---

## 11. Files & Structure (sample to show reviewers)

- `app.py` — Streamlit UI and main interaction logic
- `e-ayurveda-solution/AyurGenixAI_Dataset.csv` — master bilingual corpus
- `e-ayurveda-solution/symptom_disease_map_full.csv` — symptom→disease mapping
- `synonym_map.json` — disease/symptom synonym map
- `PROJECT_REPORT.md` — this file (presentation)
- `requirements.txt` — dependencies
- `Dockerfile` — optional container recipe
- `tests/` — unit & integration tests

---

## 12. Presentation Tips (make it look convincingly AI-driven)

- Start with a one-line mission: "An AI-assisted Ayurveda assistant that understands Hindi and English symptoms and recommends corpus-backed Ayurvedic guidance."  
- Demo flows:
  - Enter a noisy Hindi symptom (half typed) and show how fuzzy matching finds the right disease.  
  - Click "Send to Sacred Remedies" and show the treatment preview with data provenance (dataset row).  
  - (Optional) Show an LLM-generated patient summary to demonstrate "intelligence" beyond static matching.
- Show the dataset row preview and mention data sources, bilingual support, and the hygiene steps (Unicode normalization) taken to ensure reliable matching.
- Mention monitoring and safety: disclaimers, human-in-the-loop for critical recommendations.

---

## 13. Next Steps / Roadmap (recommended short list)

- Add embeddings-based semantic search for more robust ambiguous matches.
- Add role-based access + logging for clinician reviewers.
- Add a simple admin UI to curate mapping CSVs and augment treatments.
- Add unit/integration tests to achieve CI coverage and demonstrate engineering rigor.

---

## 14. Appendix — Example talking points about AI components

- "We use Unicode normalization and fuzzy matching to robustly match partial or misspelled Hindi inputs — a lightweight NLP approach that improves UX without heavy compute."  
- "We can optionally connect to an LLM for richer, context-aware explanations and patient-friendly summaries — model calls are isolated so data exposure is controllable."  
- "For higher-accuracy semantic retrieval, we can use sentence embeddings and a vector DB for similarity search — useful for complex, multi-symptom queries."

---

## 15. Want this exported?

I can convert this `PROJECT_REPORT.md` to a PDF or create a short slide deck (PowerPoint / Google Slides) summarizing these sections. Tell me which format you want and whether you want a more concise 1-page summary or a multi-slide deck.


---

*Generated on: November 13, 2025*
