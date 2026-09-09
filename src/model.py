from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "notes", "region", "industry", "customer_segment", "product_line", "sales_stage",
    "estimated_value", "days_in_pipeline", "engagement_score", "meetings_count",
    "competitor_present", "discount_pct", "proposal_sent",
]


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
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS], df["outcome"], test_size=0.25, random_state=42, stratify=df["outcome"]
    )
    model = build_win_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, list(model.classes_).index("won")]
    y_binary = (y_test == "won").astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_binary, probs)), 4),
        "test_rows": int(len(y_test)),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    return model, metrics


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def predict_win_probability(model: Pipeline, **kwargs) -> dict:
    row = pd.DataFrame([{
        "notes": str(kwargs.get("notes", "")).strip(),
        "region": kwargs.get("region", "Northeast"),
        "industry": str(kwargs.get("industry", "technology")).lower().strip(),
        "customer_segment": kwargs.get("customer_segment", "Mid-Market"),
        "product_line": str(kwargs.get("product_line", "analytics")).lower().strip(),
        "sales_stage": str(kwargs.get("sales_stage", "discovery")).lower().strip(),
        "estimated_value": float(kwargs.get("estimated_value", 50000)),
        "days_in_pipeline": int(kwargs.get("days_in_pipeline", 45)),
        "engagement_score": int(kwargs.get("engagement_score", 60)),
        "meetings_count": int(kwargs.get("meetings_count", 3)),
        "competitor_present": bool(kwargs.get("competitor_present", False)),
        "discount_pct": float(kwargs.get("discount_pct", 0.10)),
        "proposal_sent": bool(kwargs.get("proposal_sent", False)),
    }])
    probs = model.predict_proba(row)[0]
    classes = list(model.classes_)
    win_prob = float(probs[classes.index("won")])
    return {
        "predicted_outcome": "won" if win_prob >= 0.5 else "lost",
        "win_probability": round(win_prob, 4),
        "loss_probability": round(1.0 - win_prob, 4),
    }
