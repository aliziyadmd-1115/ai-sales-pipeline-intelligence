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


def sample_retriever():
    return TfidfRetriever(pd.DataFrame([{
        "opportunity_id": "OPP-1", "industry": "technology", "product_line": "analytics",
        "outcome": "won", "notes": "Executive sponsor approved the analytics proposal.",
        "close_reason": "Approved by buyer@example.com",
    }]))


def test_no_evidence_abstains_without_contacting_llm(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("LLM should not be called without evidence")
    monkeypatch.setattr("src.rag.requests.post", unexpected)
    result = answer_query("volcanology tectonic seismograph", sample_retriever(), use_llm=True)
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []


import pytest


@pytest.mark.parametrize("payload", [
    {}, [], {"response": ""}, {"response": None}, {"response": 123},
    {"response": "The deal was won without a citation."},
    {"response": "Known [OPP-1] and invented [OPP-99999]."},
])
def test_unusable_llm_outputs_fall_back(monkeypatch, payload):
    from types import SimpleNamespace
    monkeypatch.setattr("src.rag.requests.post", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: payload,
    ))
    result = answer_query("executive analytics proposal", sample_retriever(), use_llm=True)
    assert result["llm_used"] is False
    assert result["status"] == "retrieval_only"
    assert "buyer@example.com" not in result["answer"]


def test_valid_citations_and_redaction_before_llm(monkeypatch):
    from types import SimpleNamespace
    calls = []

    def respond(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "The analytics proposal was approved [OPP-1]."},
        )

    monkeypatch.setattr("src.rag.requests.post", respond)
    result = answer_query(
        "analytics proposal from seller@example.com", sample_retriever(), use_llm=True,
    )
    assert result["llm_used"] is True
    assert result["citations"][0]["opportunity_id"] == "OPP-1"
    assert "seller@example.com" not in calls[0]["json"]["prompt"]
    assert "buyer@example.com" not in calls[0]["json"]["prompt"]
    assert "untrusted data" in calls[0]["json"]["system"]


def test_invalid_json_falls_back_without_leaking_exception(monkeypatch):
    from types import SimpleNamespace

    def malformed():
        raise ValueError("private service details")
    monkeypatch.setattr("src.rag.requests.post", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=malformed,
    ))
    result = answer_query("executive analytics proposal", sample_retriever(), use_llm=True)
    assert result["llm_used"] is False
    assert "private service" not in str(result)
