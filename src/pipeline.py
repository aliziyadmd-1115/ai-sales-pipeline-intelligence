from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "opportunity_id", "created_at", "owner_email", "region", "industry",
    "customer_segment", "product_line", "sales_stage", "estimated_value",
    "days_in_pipeline", "engagement_score", "meetings_count", "competitor_present",
    "discount_pct", "proposal_sent", "notes", "outcome", "actual_revenue", "close_reason",
}
VALID_OUTCOMES = {"won", "lost"}
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    invalid = set(df["outcome"].dropna().astype(str).str.lower()) - VALID_OUTCOMES
    if invalid:
        raise ValueError(f"Unexpected outcome values: {sorted(invalid)}")


def redact_pii(text: str) -> str:
    return EMAIL_PATTERN.sub("[REDACTED_EMAIL]", str(text))


def clean_opportunities(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_schema(df)
    before = len(df)
    work = df.copy()
    work.columns = [c.strip().lower() for c in work.columns]
    work = work.drop_duplicates(subset=["opportunity_id"], keep="first")

    text_cols = [
        "region", "industry", "customer_segment", "product_line", "sales_stage",
        "notes", "outcome", "close_reason",
    ]
    for col in text_cols:
        work[col] = work[col].fillna("").astype(str).str.strip()

    work["industry"] = work["industry"].str.lower()
    work["product_line"] = work["product_line"].str.lower()
    work["sales_stage"] = work["sales_stage"].str.lower()
    work["outcome"] = work["outcome"].str.lower()
    work["notes"] = work["notes"].map(redact_pii)
    work["owner_email"] = "[REDACTED_EMAIL]"
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")

    numeric_cols = [
        "estimated_value", "days_in_pipeline", "engagement_score", "meetings_count",
        "discount_pct", "actual_revenue",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["competitor_present"] = work["competitor_present"].astype(str).str.lower().isin({"true", "1", "yes"})
    work["proposal_sent"] = work["proposal_sent"].astype(str).str.lower().isin({"true", "1", "yes"})

    work = work.dropna(subset=[
        "opportunity_id", "created_at", "industry", "customer_segment", "product_line",
        "sales_stage", "estimated_value", "engagement_score", "notes", "outcome",
    ])
    work = work[work["notes"].str.len() >= 20]
    work = work[work["outcome"].isin(VALID_OUTCOMES)]
    work = work[(work["estimated_value"] > 0) & work["engagement_score"].between(0, 100)]

    metrics = {
        "rows_raw": before,
        "rows_clean": len(work),
        "duplicates_removed": before - len(df.drop_duplicates(subset=["opportunity_id"])),
        "rows_removed_total": before - len(work),
        "pii_redacted": True,
    }
    return work.reset_index(drop=True), metrics


def run_pipeline(input_path: Path, output_path: Path) -> dict:
    raw = pd.read_csv(input_path)
    clean, metrics = clean_opportunities(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    return metrics
