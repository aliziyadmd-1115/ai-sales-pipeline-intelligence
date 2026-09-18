# Reliability and evaluation update

Based on repository commit d7757bd7569d759d2fdaf46035644e80dc152c8a.

- Replaced permissive coercion with shared data/API/model validation and aggregate quality reporting.
- Corrected string-to-boolean inference, blank/null categories, missing numeric values, infinities, invalid ranges, normalized headers, and conflicting duplicate handling.
- Redacted emails in historical close reasons and before requests reach retrieval/LLM synthesis.
- Made synthetic generation work for small row counts without resetting Python's global random state.
- Removed class balancing from probability estimation; added separate threshold-validation data, confidence intervals, calibration comparison, split manifest, environment/data fingerprint, and holdout predictions.
- Stopped arbitrary all-zero retrieval matches; excluded post-close reasons from retrieval ranking.
- Fingerprinted Chroma snapshots, repaired interrupted indexing, and capped requested results to collection size.
- Added no-evidence abstention, LLM response/citation validation, safe fallback, and bounded output/timeouts.
- Added readiness checks, normalized/bounded API inputs, UI validation feedback, Docker context exclusions, non-root runtime, and CI container smoke checks.
- Made the preview reproducible from actual artifacts and removed static live-health/CI assertions.
- Expanded tests from 15 to 65; updated documentation, model card, data dictionary, and portfolio language.

Local checks passed on Python 3.12: 65 tests, complete generation/evaluation, HTTP API smoke check, and Streamlit loading/prediction/retrieval. Docker, actual Chroma embeddings, and actual Ollama require separate runtime verification. The project remains a synthetic portfolio prototype.
