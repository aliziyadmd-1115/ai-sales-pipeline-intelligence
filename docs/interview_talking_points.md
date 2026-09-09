# Interview Talking Points

## 30-second explanation

I built an AI sales-pipeline intelligence platform to show the full path from raw business data to a usable Data & AI application. I generated synthetic CRM opportunities, added validation, cleaning and PII redaction, trained a machine-learning model to estimate win probability using text and structured sales signals, and added retrieval over historical won and lost opportunities. I exposed the workflow through FastAPI and Streamlit, with an optional ChromaDB vector store and local LLM for grounded RAG-style analysis, then added pytest, GitHub Actions, and Docker for engineering quality.

## Why it matters

The project goes beyond descriptive analysis. It turns data quality, predictive analytics, retrieval, and generative AI into one workflow that a business team could review and integrate into other tools.

## Technical tradeoff

TF-IDF keeps development and CI lightweight and deterministic. ChromaDB plus sentence-transformer embeddings supports semantic retrieval for less exact language matches. Keeping both makes the architecture practical to test while still demonstrating vector search.

## Evaluation and business threshold

The logistic-regression pipeline achieved 0.795 ROC-AUC and 0.726 macro F1 on a fixed stratified holdout set, compared with 0.500 ROC-AUC and 0.386 macro F1 for an always-lost majority baseline. I also evaluated thresholds instead of treating 0.50 as automatically correct. A 0.40 threshold increased won-deal recall to 85.3%, while 0.60 increased precision to 65.9%. In a client setting, I would select the threshold using intervention cost, opportunity value, team capacity, and out-of-time validation.

## Honest limitation

The dataset and benchmarks are synthetic, so the results demonstrate a reproducible engineering and evaluation workflow rather than production sales performance. The TF-IDF retrieval hit-at-3 result also shows that retrieval quality needs further domain data and human relevance evaluation before production use.

## Responsible-AI point

I used only synthetic data, redacted emails, kept post-close fields out of model features to avoid leakage, grounded LLM output in retrieved historical opportunities, and returned source opportunity IDs as evidence.
