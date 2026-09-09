import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.generate_data import generate_dataset
from src.model import train_and_evaluate
from src.pipeline import clean_opportunities
from src.retrieval import TfidfRetriever


@pytest.fixture(scope="module")
def resources():
    raw = generate_dataset(600)
    clean, _ = clean_opportunities(raw)
    model, _ = train_and_evaluate(clean)
    return model, TfidfRetriever(clean)


@pytest.fixture
def client(monkeypatch, resources):
    model, retriever = resources
    monkeypatch.setattr(api, "get_model", lambda: model)
    monkeypatch.setattr(api, "get_retriever", lambda: retriever)
    return TestClient(api.app)


VALID_OPPORTUNITY = {
    "notes": "Executive sponsor is engaged and requested a final analytics proposal.",
    "region": "Northeast",
    "industry": "technology",
    "customer_segment": "Mid-Market",
    "product_line": "analytics",
    "sales_stage": "proposal",
    "estimated_value": 75000,
    "days_in_pipeline": 52,
    "engagement_score": 78,
    "meetings_count": 5,
    "competitor_present": True,
    "discount_pct": 0.12,
    "proposal_sent": True,
    "decision_threshold": 0.50,
}


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_probability_and_threshold(client):
    response = client.post("/predict-win", json=VALID_OPPORTUNITY)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["win_probability"] <= 1
    assert body["predicted_outcome"] in {"won", "lost"}
    assert body["decision_threshold"] == 0.50


def test_predict_endpoint_rejects_invalid_score(client):
    payload = {**VALID_OPPORTUNITY, "engagement_score": 101}
    response = client.post("/predict-win", json=payload)
    assert response.status_code == 422


def test_similar_opportunities_endpoint(client):
    response = client.post(
        "/similar-opportunities",
        json={"query": "analytics proposal with executive sponsorship", "top_k": 3},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert {"opportunity_id", "outcome", "score"}.issubset(results[0])


def test_answer_endpoint_returns_grounded_citations(client):
    response = client.post(
        "/answer",
        json={
            "query": "What patterns affect analytics deals with executive sponsorship?",
            "top_k": 3,
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["llm_used"] is False
    assert len(body["citations"]) == 3


def test_search_endpoint_rejects_invalid_top_k(client):
    response = client.post(
        "/similar-opportunities",
        json={"query": "analytics proposal with executive sponsorship", "top_k": 0},
    )
    assert response.status_code == 422
