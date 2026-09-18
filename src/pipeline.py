from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .schemas import OpportunityRequest, redact_pii

FEATURE_COLUMNS = [name for name in OpportunityRequest.model_fields if name != "decision_threshold"]
REQUIRED_COLUMNS = set(FEATURE_COLUMNS) | {
    "opportunity_id", "created_at", "owner_email", "outcome", "actual_revenue", "close_reason",
}
VALID_OUTCOMES = {"won", "lost"}


def validate_schema(df: pd.DataFrame) -> None:
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_opportunities(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    validate_schema(work)
    # Drop unrecognized columns so extra source fields cannot bypass redaction.
    work = work[[c for c in work.columns if c in REQUIRED_COLUMNS]]
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce", format="mixed", utc=True)
    work["actual_revenue"] = pd.to_numeric(work["actual_revenue"], errors="coerce")
    invalid_fields: Counter = Counter()
    rows = []
    invalid_rows = 0
    for record in work.to_dict(orient="records"):
        errors = set()
        try:
            normalized = OpportunityRequest(**{name: record[name] for name in FEATURE_COLUMNS})
            record.update(normalized.model_dump(exclude={"decision_threshold"}))
        except ValidationError as exc:
            errors.update(str(error["loc"][0]) for error in exc.errors())
        opp_id = record["opportunity_id"]
        if pd.isna(opp_id) or not str(opp_id).strip():
            errors.add("opportunity_id")
        record["opportunity_id"] = str(opp_id).strip()
        record["outcome"] = str(record["outcome"]).strip().lower()
        if record["outcome"] not in VALID_OUTCOMES:
            errors.add("outcome")
        if pd.isna(record["created_at"]):
            errors.add("created_at")
        revenue = record["actual_revenue"]
        if not np.isfinite(revenue) or revenue < 0 or (record["outcome"] == "lost" and revenue != 0):
            errors.add("actual_revenue")
        if errors:
            invalid_fields.update(errors)
            invalid_rows += 1
            continue
        record["owner_email"] = "[REDACTED_EMAIL]"
        reason = record["close_reason"]
        record["close_reason"] = "" if pd.isna(reason) else redact_pii(" ".join(str(reason).split()))
        rows.append(record)

    clean = pd.DataFrame(rows, columns=work.columns)
    duplicate_rows = clean[clean.duplicated("opportunity_id", keep=False)]
    if not duplicate_rows.empty:
        conflicts = duplicate_rows.groupby("opportunity_id").nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError("Conflicting records share opportunity_id; resolve them before training")
    duplicates_removed = int(clean.duplicated("opportunity_id").sum())
    clean = clean.drop_duplicates("opportunity_id").reset_index(drop=True)
    metrics = {
        "rows_raw": len(df),
        "rows_clean": len(clean),
        "duplicates_removed": duplicates_removed,
        "invalid_rows_removed": invalid_rows,
        "invalid_field_counts": dict(sorted(invalid_fields.items())),
        "rows_removed_total": len(df) - len(clean),
        "pii_redacted": True,
        "redaction_scope": "Email addresses in notes, close_reason and owner_email; not all PII types",
    }
    return clean, metrics


def run_pipeline(input_path: Path, output_path: Path) -> dict:
    raw = pd.read_csv(input_path)
    clean, metrics = clean_opportunities(raw)
    if clean.empty:
        raise ValueError("No valid opportunities remain after cleaning")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    return metrics
