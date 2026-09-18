from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from .schemas import OpportunityRequest, SearchRequest, AnswerRequest

from .model import load_model, predict_win_probability
from .rag import answer_query
from .retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/opportunities_clean.csv"
MODEL_PATH = ROOT / "artifacts/win_model.joblib"

app = FastAPI(
    title="AI Sales Pipeline Intelligence API",
    version="1.1.0",
    description="Win-probability prediction, historical opportunity retrieval, and grounded AI assistance.",
)


@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run the data pipeline first.")
    from .pipeline import clean_opportunities
    clean, quality = clean_opportunities(pd.read_csv(DATA_PATH))
    if clean.empty or quality["rows_removed_total"]:
        raise RuntimeError("Historical data is invalid; rerun the pipeline")
    return clean


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run python -m src.evaluate first.")
    try:
        model = load_model(MODEL_PATH)
        if set(model.classes_) != {"won", "lost"} or not callable(model.predict_proba):
            raise ValueError("Invalid model")
        return model
    except Exception as exc:
        raise RuntimeError("Model artifact is unreadable; rerun evaluation") from exc


@lru_cache(maxsize=1)
def get_retriever():
    return build_retriever(get_df(), backend=os.getenv("RETRIEVAL_BACKEND", "tfidf"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tfidf")}


@app.get("/ready")
def ready() -> dict:
    try:
        get_model()
        get_retriever()
    except (OSError, RuntimeError, ValueError, KeyError):
        raise HTTPException(status_code=503, detail="Model or retrieval resources are not ready") from None
    return {"status": "ready"}


@app.post("/predict-win")
def predict(req: OpportunityRequest) -> dict:
    try:
        return predict_win_probability(get_model(), **req.model_dump())
    except (OSError, RuntimeError):
        raise HTTPException(status_code=503, detail="Prediction model unavailable; run evaluation") from None


@app.post("/similar-opportunities")
def similar(req: SearchRequest) -> dict:
    try:
        hits = get_retriever().search(req.query, k=req.top_k)
        return {"results": [hit.__dict__ for hit in hits]}
    except (OSError, RuntimeError, ValueError, KeyError):
        raise HTTPException(status_code=503, detail="Retrieval unavailable; check data and backend setup") from None


@app.post("/answer")
def answer(req: AnswerRequest) -> dict:
    try:
        return answer_query(req.query, get_retriever(), use_llm=req.use_llm, k=req.top_k)
    except (OSError, RuntimeError, ValueError, KeyError):
        raise HTTPException(status_code=503, detail="Retrieval unavailable; check data and backend setup") from None
