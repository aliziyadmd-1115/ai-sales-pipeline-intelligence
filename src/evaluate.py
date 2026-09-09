from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import save_model, train_and_evaluate
from .retrieval import TfidfRetriever


def retrieval_same_industry_hit_at_k(df: pd.DataFrame, k: int = 3, sample_size: int = 120) -> float:
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    retriever = TfidfRetriever(df)
    hits = 0
    for _, row in sample.iterrows():
        query = str(row["notes"])
        results = retriever.search(query, k=k + 1)
        results = [r for r in results if r.opportunity_id != row.opportunity_id][:k]
        if any(r.industry == row.industry for r in results):
            hits += 1
    return hits / len(sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/opportunities_clean.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/win_model.joblib"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model, metrics = train_and_evaluate(df)
    save_model(model, args.model)
    metrics["retrieval_same_industry_hit_at_3"] = round(retrieval_same_industry_hit_at_k(df), 4)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
