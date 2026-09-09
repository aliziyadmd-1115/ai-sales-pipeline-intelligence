from src.generate_data import generate_dataset
from src.pipeline import clean_opportunities
from src.model import train_and_evaluate


def test_model_trains_and_returns_metrics():
    raw = generate_dataset(600)
    clean, _ = clean_opportunities(raw)
    model, metrics = train_and_evaluate(clean)
    assert metrics["roc_auc"] >= 0.65
    assert hasattr(model, "predict_proba")
