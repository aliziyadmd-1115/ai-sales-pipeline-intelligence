# Architecture Notes

## Data lifecycle

1. Generate or ingest synthetic CRM opportunities.
2. Validate required fields and accepted outcomes.
3. Clean duplicates, normalize categories, parse dates, and coerce numeric fields.
4. Redact email PII before analytics or retrieval.
5. Train a win-probability model using text and structured pre-close features.
6. Index historical opportunities with TF-IDF or ChromaDB semantic vector search.
7. Retrieve similar opportunities and prior win/loss reasons.
8. Generate grounded business analysis through an optional local LLM API.
9. Serve model scoring and retrieval through FastAPI and Streamlit.
10. Test and package with pytest, GitHub Actions, and Docker.

## Leakage control

The target `outcome`, `actual_revenue`, and `close_reason` are not model input features. `close_reason` is retained only as historical evidence for retrieval after the opportunity has closed.

## Why two retrieval backends?

TF-IDF is lightweight and deterministic for local development and CI. ChromaDB with sentence-transformer embeddings adds semantic matching when two opportunities describe similar situations using different wording.
