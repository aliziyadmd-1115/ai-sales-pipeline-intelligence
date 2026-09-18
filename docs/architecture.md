# Architecture notes

1. Generate synthetic data deterministically using a local random generator.
2. Normalize headers and validate the required schema.
3. Apply the shared Pydantic feature contract; validate IDs, timestamps, outcomes and revenues.
4. Reject malformed rows with aggregate reasons; detect conflicting duplicate IDs.
5. Redact emails in notes, owner_email and close_reason; drop unexpected source fields.
6. Sort IDs, create disjoint training/validation/test partitions, fit preprocessing only on training data.
7. Compare thresholds on validation data; export final holdout results, predictions, split IDs and calibration evidence.
8. Index historical notes, industry and product using TF-IDF or optional Chroma embeddings.
9. Return sufficiently similar evidence; use close reasons only after retrieval.
10. Optionally call Ollama with separate system instructions and sanitized context; check response/citation format or return retrieval evidence.
11. Serve predictions and evidence through FastAPI and Streamlit; test the default workflow and optional service contracts.
12. Build an unprivileged Docker image and run a CI smoke test against its API.

The model, API, and pipeline share src/schemas.py to prevent training/serving normalization differences. The feature allowlist excludes labels and post-close fields.

TF-IDF uses a 0.10 minimum cosine similarity and rejects all-zero vectors. Chroma uses a 0.25 minimum similarity and content-addressed collection names. These are conservative demo defaults, not calibrated relevance probabilities; backend scores are not interchangeable. Old Chroma snapshots remain on disk until explicitly removed.

Liveness (/health) checks the process; readiness (/ready) loads model and retrieval resources. Cached resources are process-local. Restart workers after updating data/models. Joblib models are loaded only from trusted local build artifacts.

Citation membership does not verify every claim, and retrieved content remains untrusted even when an ID is valid. This is a demonstration, not a production security boundary.
