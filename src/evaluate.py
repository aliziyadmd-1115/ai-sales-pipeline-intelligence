from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from .model import save_model, train_with_holdout_predictions
from .retrieval import TfidfRetriever

DELOITTE_GREEN = "#86BC25"
CHARCOAL = "#242424"
SLATE = "#607080"
LIGHT_GRAY = "#E8ECEF"


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


def build_score_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    band_order = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    work = predictions.copy()
    work["score_band"] = pd.cut(
        work["win_probability"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=band_order,
        include_lowest=True,
    )
    summary = (
        work.groupby("score_band", observed=False)
        .agg(
            opportunities=("actual_won", "size"),
            observed_win_rate=("actual_won", "mean"),
            average_predicted_probability=("win_probability", "mean"),
        )
        .reset_index()
    )
    summary["score_band"] = summary["score_band"].astype(str)
    for col in ("observed_win_rate", "average_predicted_probability"):
        summary[col] = summary[col].fillna(0).round(4)
    return summary


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)


def save_evaluation_charts(
    metrics: dict,
    predictions: pd.DataFrame,
    score_bands: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions["actual_won"]
    probabilities = predictions["win_probability"]
    y_pred = (probabilities >= 0.50).astype(int)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(fpr, tpr, color=DELOITTE_GREEN, linewidth=2.5, label=f"Logistic regression (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", label="Random baseline (AUC 0.500)")
    ax.set(title="Win-probability model ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    display = ConfusionMatrixDisplay(matrix, display_labels=["Lost", "Won"])
    display.plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
    ax.set_title("Holdout confusion matrix at 0.50 threshold")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    observed = score_bands["observed_win_rate"] * 100
    bars = ax.bar(score_bands["score_band"], observed, color=DELOITTE_GREEN, width=0.68)
    for bar, rate, count in zip(bars, observed, score_bands["opportunities"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{rate:.1f}%\nn={count}", ha="center", va="bottom", fontsize=9)
    ax.set(
        title="Observed win rate rises across model score bands",
        xlabel="Predicted win-probability band",
        ylabel="Observed win rate",
        ylim=(0, max(100, float(observed.max()) + 12)),
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "score_band_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Macro F1", "ROC-AUC"],
        "Model": [metrics["accuracy"], metrics["macro_f1"], metrics["roc_auc"]],
        "Majority baseline": [
            metrics["baseline"]["accuracy"],
            metrics["baseline"]["macro_f1"],
            metrics["baseline"]["roc_auc"],
        ],
    })
    x = range(len(comparison))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.34
    model_bars = ax.bar([i - width / 2 for i in x], comparison["Model"], width, label="Model", color=DELOITTE_GREEN)
    baseline_bars = ax.bar([i + width / 2 for i in x], comparison["Majority baseline"], width, label="Majority baseline", color=SLATE)
    for bars in (model_bars, baseline_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x), comparison["Metric"])
    ax.set(title="Model performance versus baseline", ylabel="Score", ylim=(0, 1.0))
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "model_vs_baseline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/opportunities_clean.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/win_model.joblib"))
    parser.add_argument("--chart-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--thresholds", type=Path, default=Path("artifacts/threshold_analysis.csv"))
    parser.add_argument("--score-bands", type=Path, default=Path("artifacts/score_band_performance.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model, metrics, predictions = train_with_holdout_predictions(df)
    save_model(model, args.model)
    metrics["retrieval_same_industry_hit_at_3"] = round(retrieval_same_industry_hit_at_k(df), 4)
    score_bands = build_score_bands(predictions)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["threshold_analysis"]).to_csv(args.thresholds, index=False)
    score_bands.to_csv(args.score_bands, index=False)
    save_evaluation_charts(metrics, predictions, score_bands, args.chart_dir)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
