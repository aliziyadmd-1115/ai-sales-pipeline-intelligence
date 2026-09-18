# Interview talking points

## 30-second explanation

I built a sales-pipeline intelligence prototype that takes synthetic CRM data through validation, probability modeling, historical retrieval, and a usable API and interface. I used a shared input contract, kept post-close fields out of predictions, separated threshold selection from final testing, and added regression tests and a Docker CI workflow. An optional local LLM can summarize retrieved evidence, with citation checks and fallback behavior.

## Technical choices I can explain

- Logistic regression gives a clear, reproducible baseline for mixed text/structured data. I removed class weighting because the application exposes probabilities; unweighted fitting better preserves the observed class prior, while calibration still needs evaluation.
- The data is split 60/15/25 into training, validation, and test sets. Preprocessing is fitted only on training data, threshold candidates are compared on validation data, and final results use the held-out test set.
- The test ROC-AUC is 0.790, with a 0.757–0.822 row-bootstrap interval. Brier score is 0.177 against a constant training-prior baseline of 0.233. These support the synthetic demonstration, not a real-world performance claim.
- A 0.30 threshold maximized validation won-F1 among five candidates. On the final test it yields 56.44% precision and 82.01% recall. The API keeps an explicit, configurable threshold because commercial costs and capacity are not known.
- TF-IDF is deterministic and lightweight. Chroma is optional and adds semantic embeddings, but needs live integration and human relevance evaluation. Content fingerprints prevent changed data from silently reusing stale vector collections.
- No-overlap retrieval returns no evidence. LLM service failures, empty/malformed output, and invalid citation IDs return retrieval-only context. Valid citation IDs alone cannot prove factual correctness.
- Process liveness is distinct from readiness to serve predictions and evidence. Tests exercise both.

## What I would improve with real client data

The synthetic notes repeat templates and the notes-only same-industry retrieval proxy is below random. I would collect richer pre-close snapshots, define a prediction horizon, establish time/account-separated evaluation, and create independently judged retrieval queries. I would validate probability calibration, intervention costs, drift, and operational impact before using the system for decisions.

## Verification scope

The default Python workflow, 65 regression tests, live local HTTP API, and Streamlit test harness were verified. CI includes Docker build/runtime checks. Live Chroma embeddings and Ollama were not exercised in the local review. I would not claim a deployed production system or measured business impact.
