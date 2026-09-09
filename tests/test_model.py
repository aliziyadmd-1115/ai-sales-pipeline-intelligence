import pytest

from src.generate_data import generate_dataset
from src.pipeline import clean_opportunities
from src.model import predict_win_probability, train_and_evaluate


def test_model_trains_and_returns_metrics():
    raw = generate_dataset(600)
    clean, _ = clean_opportunities(raw)
    model, metrics = train_and_evaluate(clean)
    assert metrics["roc_auc"] >= 0.65
    assert metrics["roc_auc"] > metrics["baseline"]["roc_auc"]
    assert len(metrics["threshold_analysis"]) == 5
    assert 0 <= metrics["brier_score"] <= 1
    assert hasattr(model, "predict_proba")


def test_prediction_respects_decision_threshold():
    raw = generate_dataset(600)
    clean, _ = clean_opportunities(raw)
    model, _ = train_and_evaluate(clean)
    inputs = {
        "notes": "Decision makers are engaged and requested a final proposal.",
        "engagement_score": 75,
        "proposal_sent": True,
    }
    permissive = predict_win_probability(model, **inputs, decision_threshold=0.0)
    strict = predict_win_probability(model, **inputs, decision_threshold=1.0)
    assert permissive["predicted_outcome"] == "won"
    assert strict["predicted_outcome"] == "lost"


def test_prediction_rejects_invalid_decision_threshold():
    raw = generate_dataset(300)
    clean, _ = clean_opportunities(raw)
    model, _ = train_and_evaluate(clean)
    with pytest.raises(ValueError, match="decision_threshold"):
        predict_win_probability(model, notes="A sufficiently detailed opportunity note.", decision_threshold=1.1)
