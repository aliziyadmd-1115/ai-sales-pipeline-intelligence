import pandas as pd
import requests

from src.rag import answer_query
from src.retrieval import TfidfRetriever


def test_llm_failure_returns_retrieval_only_evidence(monkeypatch):
    df = pd.DataFrame([
        {
            "opportunity_id": "OPP-1",
            "industry": "technology",
            "product_line": "analytics",
            "outcome": "won",
            "notes": "Executive sponsor approved the analytics proposal.",
            "close_reason": "Strong sponsorship",
        }
    ])
    retriever = TfidfRetriever(df)

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("local LLM unavailable")

    monkeypatch.setattr("src.rag.requests.post", unavailable)
    result = answer_query(
        "What happened in the analytics opportunity with executive sponsorship?",
        retriever,
        use_llm=True,
        k=1,
    )

    assert result["llm_used"] is False
    assert "retrieval context" in result["answer"]
    assert result["citations"][0]["opportunity_id"] == "OPP-1"
    assert "warning" in result
