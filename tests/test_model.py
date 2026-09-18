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


def test_holdout_is_disjoint_reproducible_and_has_no_postclose_features():
    import pandas as pd
    from src.model import train_with_holdout_predictions, FEATURE_COLUMNS
    clean, _ = clean_opportunities(generate_dataset(300))
    model, metrics, predictions = train_with_holdout_predictions(clean)
    _, shuffled_metrics, shuffled_predictions = train_with_holdout_predictions(
        clean.sample(frac=1, random_state=5),
    )
    manifest = predictions.attrs["split_manifest"]
    assert manifest["opportunity_id"].is_unique
    assert len(manifest) == len(clean)
    assert metrics["threshold_analysis_split"] == "validation"
    assert metrics["split"]["train_rows"] == 180
    test_ids = set(manifest.loc[manifest["split"] == "test", "opportunity_id"])
    assert set(predictions["opportunity_id"]) == test_ids
    assert {"outcome", "actual_revenue", "close_reason"}.isdisjoint(FEATURE_COLUMNS)
    assert metrics["dataset_sha256"] == shuffled_metrics["dataset_sha256"]
    pd.testing.assert_frame_equal(predictions, shuffled_predictions)
    assert metrics["brier_score"] < metrics["baseline"]["brier_score"]


def test_inference_normalizes_false_strings_and_redacts_notes():
    import numpy as np

    class CaptureModel:
        classes_ = np.array(["lost", "won"])

        def predict_proba(self, row):
            assert row.iloc[0]["competitor_present"] == False
            assert row.iloc[0]["industry"] == "technology"
            assert "buyer@example.com" not in row.iloc[0]["notes"]
            return np.array([[0.6, 0.4]])

    result = predict_win_probability(
        CaptureModel(), notes="Discuss proposal with buyer@example.com",
        industry="  TECHNOLOGY ", competitor_present="false", decision_threshold=0.405,
    )
    assert result["decision_threshold"] == 0.405
    assert result["predicted_outcome"] == "lost"
