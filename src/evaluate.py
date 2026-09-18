from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from .model import save_model, train_with_holdout_predictions
from .retrieval import TfidfRetriever

DELOITTE_GREEN = "#86BC25"
CHARCOAL = "#242424"
SLATE = "#607080"
LIGHT_GRAY = "#E8ECEF"


def evaluate_retrieval_proxy(df: pd.DataFrame, manifest: pd.DataFrame, k: int = 3) -> dict:
    """A weak same-industry proxy, not human-judged semantic relevance."""
    train_ids = manifest.loc[manifest["split"] == "train", "opportunity_id"]
    test_ids = manifest.loc[manifest["split"] == "test", "opportunity_id"]
    history = df[df["opportunity_id"].isin(train_ids)]
    queries = df[df["opportunity_id"].isin(test_ids)].sample(
        min(120, len(test_ids)), random_state=42,
    )
    retriever = TfidfRetriever(history)
    hits, random_expectations = [], []
    draw = min(k, len(history))
    for row in queries.itertuples():
        results = retriever.search(row.notes, k=k)
        hits.append(any(r.industry == row.industry for r in results))
        irrelevant = int((history["industry"] != row.industry).sum())
        random_expectations.append(1 - comb(irrelevant, draw) / comb(len(history), draw))
    return {
        "metric": f"same_industry_hit_at_{k}",
        "hit_rate": round(sum(hits) / len(hits), 4),
        "random_expected_hit_rate": round(sum(random_expectations) / len(hits), 4),
        "query_rows": len(queries),
        "index_rows": len(history),
        "query_index_overlap": 0,
        "limitation": "Synthetic notes-only queries; same industry is not a human relevance judgment",
    }


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
        summary[col] = summary[col].round(4)
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
        if not count:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{rate:.1f}%\nn={count}", ha="center", va="bottom", fontsize=9)
    ax.set(
        title="Observed win rate by model score band",
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

    observed, predicted = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(predicted, observed, marker="o", color=DELOITTE_GREEN, label="Holdout score bins")
    ax.plot([0, 1], [0, 1], "--", color=SLATE, label="Perfect calibration")
    ax.set(title="Probability calibration on synthetic holdout", xlabel="Mean predicted probability",
           ylabel="Observed win rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/opportunities_clean.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/win_model.joblib"))
    parser.add_argument("--chart-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--thresholds", type=Path, default=Path("artifacts/threshold_analysis.csv"))
    parser.add_argument("--preview", type=Path, default=Path("docs/portfolio_preview.html"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/holdout_predictions.csv"))
    parser.add_argument("--split-manifest", type=Path, default=Path("artifacts/split_manifest.csv"))
    parser.add_argument("--score-bands", type=Path, default=Path("artifacts/score_band_performance.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model, metrics, predictions = train_with_holdout_predictions(df)
    save_model(model, args.model)
    manifest = predictions.attrs["split_manifest"]
    metrics["retrieval_proxy"] = evaluate_retrieval_proxy(df, manifest)
    metrics["environment"] = {
        "python": platform.python_version(),
        **{package: version(package) for package in ("pandas", "numpy", "scipy", "scikit-learn", "joblib")},
    }
    score_bands = build_score_bands(predictions)

    for output in (args.metrics, args.thresholds, args.score_bands, args.predictions, args.split_manifest):
        output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)
    manifest.to_csv(args.split_manifest, index=False)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["threshold_analysis"]).to_csv(args.thresholds, index=False)
    score_bands.to_csv(args.score_bands, index=False)
    save_evaluation_charts(metrics, predictions, score_bands, args.chart_dir)
    from .preview import write_preview
    write_preview(model, df, metrics, args.preview)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
