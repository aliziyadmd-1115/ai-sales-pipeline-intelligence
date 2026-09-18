from __future__ import annotations

from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pipeline import FEATURE_COLUMNS, clean_opportunities
from .schemas import OpportunityRequest


def build_win_model() -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1200), "notes"),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), [
                "region", "industry", "customer_segment", "product_line", "sales_stage",
                "competitor_present", "proposal_sent",
            ]),
            ("numeric", StandardScaler(), [
                "estimated_value", "days_in_pipeline", "engagement_score", "meetings_count", "discount_pct",
            ]),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1500, random_state=42)),
    ])


def _threshold_metrics(y_true, probabilities, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "opportunities_flagged_pct": round(float(predicted.mean() * 100), 2),
    }


def train_with_holdout_predictions(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame]:
    clean, quality = clean_opportunities(df)
    if quality["rows_removed_total"]:
        raise ValueError("Training data must be cleaned with unique opportunity IDs first")
    if set(clean["outcome"]) != {"won", "lost"} or clean["outcome"].value_counts().min() < 8:
        raise ValueError("Training requires at least 8 won and 8 lost opportunities")
    # Sorting makes partitions stable even when the input CSV row order changes.
    clean = clean.sort_values("opportunity_id").reset_index(drop=True)
    development, test = train_test_split(
        clean, test_size=0.25, random_state=42, stratify=clean["outcome"],
    )
    train, validation = train_test_split(
        development, test_size=0.20, random_state=43, stratify=development["outcome"],
    )
    model = build_win_model()
    model.fit(train[FEATURE_COLUMNS], train["outcome"])
    won_index = list(model.classes_).index("won")
    validation_probs = model.predict_proba(validation[FEATURE_COLUMNS])[:, won_index]
    threshold_analysis = [
        _threshold_metrics((validation["outcome"] == "won").astype(int), validation_probs, t)
        for t in (0.30, 0.40, 0.50, 0.60, 0.70)
    ]
    selected = max(threshold_analysis, key=lambda item: (item["f1"], item["threshold"]))["threshold"]

    probs = model.predict_proba(test[FEATURE_COLUMNS])[:, won_index]
    pred = np.where(probs >= 0.50, "won", "lost")
    y_test = test["outcome"]
    y_binary = (y_test == "won").astype(int)
    majority_label = train["outcome"].mode().iloc[0]
    majority_pred = np.repeat(majority_label, len(test))
    majority_probability = float((train["outcome"] == "won").mean())
    prior_probs = np.full(len(test), majority_probability)

    # Fixed-model, row-bootstrap uncertainty; does not estimate training variability.
    rng = np.random.default_rng(42)
    y_array = y_binary.to_numpy()
    bootstrap_auc = []
    for _ in range(500):
        sample = rng.integers(0, len(test), len(test))
        if len(np.unique(y_array[sample])) == 2:
            bootstrap_auc.append(roc_auc_score(y_array[sample], probs[sample]))
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_binary, probs)), 4),
        "roc_auc_ci95": np.round(np.quantile(bootstrap_auc, [0.025, 0.975]), 4).tolist(),
        "brier_score": round(float(brier_score_loss(y_binary, probs)), 4),
        "test_rows": len(test),
        "split": {
            "strategy": "stratified random 60/15/25 train/validation/test; sorted opportunity IDs",
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
            "test_seed": 42, "validation_seed": 43,
        },
        "feature_columns": FEATURE_COLUMNS,
        "dataset_sha256": hashlib.sha256(clean.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "default_decision_threshold": 0.50,
        "baseline": {
            "strategy": f"always predict {majority_label}; probability = training win rate",
            "accuracy": round(float(accuracy_score(y_test, majority_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, majority_pred, average="macro")), 4),
            "roc_auc": 0.5,
            "brier_score": round(float(brier_score_loss(y_binary, prior_probs)), 4),
            "training_win_rate": round(majority_probability, 4),
        },
        "threshold_analysis_split": "validation",
        "threshold_analysis": threshold_analysis,
        "validation_selected_threshold": selected,
        "selected_threshold_test_metrics": _threshold_metrics(y_binary, probs, selected),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    predictions = pd.DataFrame({
        "opportunity_id": test["opportunity_id"].to_numpy(),
        "actual_outcome": y_test.to_numpy(),
        "actual_won": y_binary.to_numpy(),
        "predicted_outcome": pred,
        "win_probability": probs,
    })
    predictions.attrs["split_manifest"] = pd.concat([
        part[["opportunity_id"]].assign(split=label)
        for label, part in (("train", train), ("validation", validation), ("test", test))
    ], ignore_index=True)
    return model, metrics, predictions


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    model, metrics, _ = train_with_holdout_predictions(df)
    return model, metrics


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def predict_win_probability(model: Pipeline, **kwargs) -> dict:
    request = OpportunityRequest(**kwargs)
    decision_threshold = request.decision_threshold
    row = pd.DataFrame([request.model_dump(exclude={"decision_threshold"})])
    probs = model.predict_proba(row)[0]
    classes = list(model.classes_)
    win_prob = float(probs[classes.index("won")])
    return {
        "predicted_outcome": "won" if win_prob >= decision_threshold else "lost",
        "win_probability": round(win_prob, 4),
        "loss_probability": round(1.0 - win_prob, 4),
        "decision_threshold": decision_threshold,
    }
