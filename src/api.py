from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import load_model, predict_win_probability
from .rag import answer_query
from .retrieval import build_retriever

DATA_PATH = Path("data/opportunities_clean.csv")
MODEL_PATH = Path("artifacts/win_model.joblib")

app = FastAPI(
    title="AI Sales Pipeline Intelligence API",
    version="1.0.0",
    description="Win-probability prediction, historical opportunity retrieval, and grounded AI assistance.",
)


class OpportunityRequest(BaseModel):
    notes: str = Field(..., min_length=10)
    region: str = "Northeast"
    industry: str = "technology"
    customer_segment: str = "Mid-Market"
    product_line: str = "analytics"
    sales_stage: str = "discovery"
    estimated_value: float = Field(50000, gt=0)
    days_in_pipeline: int = Field(45, ge=0, le=1000)
    engagement_score: int = Field(60, ge=0, le=100)
    meetings_count: int = Field(3, ge=0, le=100)
    competitor_present: bool = False
    discount_pct: float = Field(0.10, ge=0, le=1)
    proposal_sent: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=10)
    top_k: int = Field(3, ge=1, le=10)


class AnswerRequest(SearchRequest):
    use_llm: bool = False


@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run the data pipeline first.")
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run python -m src.evaluate first.")
    return load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def get_retriever():
    return build_retriever(get_df(), backend=os.getenv("RETRIEVAL_BACKEND", "tfidf"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retrieval_backend": os.getenv("RETRIEVAL_BACKEND", "tfidf")}


@app.post("/predict-win")
def predict(req: OpportunityRequest) -> dict:
    try:
        return predict_win_probability(get_model(), **req.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/similar-opportunities")
def similar(req: SearchRequest) -> dict:
    try:
        hits = get_retriever().search(req.query, k=req.top_k)
        return {"results": [hit.__dict__ for hit in hits]}
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/answer")
def answer(req: AnswerRequest) -> dict:
    try:
        return answer_query(req.query, get_retriever(), use_llm=req.use_llm, k=req.top_k)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
