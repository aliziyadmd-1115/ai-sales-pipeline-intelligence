import pandas as pd
import pytest

from src.pipeline import clean_opportunities, redact_pii


def test_redact_pii():
    assert redact_pii("contact buyer@example.com") == "contact [REDACTED_EMAIL]"


def test_clean_opportunities_removes_duplicate_ids():
    row = {
        "opportunity_id": "OPP-1", "created_at": "2026-01-01", "owner_email": "a@example.com",
        "region": "Northeast", "industry": "technology", "customer_segment": "Mid-Market",
        "product_line": "analytics", "sales_stage": "proposal", "estimated_value": 50000,
        "days_in_pipeline": 40, "engagement_score": 75, "meetings_count": 4,
        "competitor_present": False, "discount_pct": 0.10, "proposal_sent": True,
        "notes": "Decision makers are engaged. Contact buyer@example.com for follow-up.",
        "outcome": "won", "actual_revenue": 49000, "close_reason": "Strong sponsorship."
    }
    df = pd.DataFrame([row, row.copy()])
    clean, metrics = clean_opportunities(df)
    assert len(clean) == 1
    assert metrics["duplicates_removed"] == 1
    assert clean.loc[0, "owner_email"] == "[REDACTED_EMAIL]"
    assert "buyer@example.com" not in clean.loc[0, "notes"]


def test_clean_opportunities_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_opportunities(pd.DataFrame({"opportunity_id": ["OPP-1"]}))


@pytest.mark.parametrize("field,value", [
    ("opportunity_id", "  "), ("created_at", "not a date"), ("industry", None),
    ("customer_segment", ""), ("sales_stage", "closed_won"), ("estimated_value", float("inf")),
    ("days_in_pipeline", -1), ("meetings_count", None), ("meetings_count", 1.5),
    ("engagement_score", 101), ("discount_pct", 1.2), ("proposal_sent", "perhaps"),
    ("competitor_present", None), ("outcome", "pending"), ("actual_revenue", -1),
    ("notes", "                     "),
])
def test_invalid_values_are_removed_and_counted(field, value):
    from src.generate_data import generate_dataset
    raw = generate_dataset(20).drop_duplicates("opportunity_id").head(2).copy().astype(object)
    raw.loc[raw.index[0], field] = value
    clean, metrics = clean_opportunities(raw)
    assert len(clean) == 1
    assert metrics["invalid_rows_removed"] == 1
    assert metrics["invalid_field_counts"][field] == 1


def test_headers_categories_and_email_redaction_are_consistent():
    from src.generate_data import generate_dataset
    raw = generate_dataset(20).drop_duplicates("opportunity_id").head(1).copy()
    raw["industry"] = "  TECHNOLOGY  "
    raw["close_reason"] = "Approved by someone@example.com"
    raw["competitor_present"] = "false"
    raw.columns = [" " + c.upper() + " " for c in raw.columns]
    clean, metrics = clean_opportunities(raw)
    assert metrics["rows_clean"] == 1
    assert clean.iloc[0].industry == "technology"
    assert not clean.iloc[0].competitor_present
    assert "someone@example.com" not in clean.iloc[0].close_reason


def test_conflicting_duplicates_do_not_silently_discard_evidence():
    from src.generate_data import generate_dataset
    raw = generate_dataset(20).drop_duplicates("opportunity_id").head(1)
    copy = raw.copy()
    copy["estimated_value"] += 100
    with pytest.raises(ValueError, match="Conflicting"):
        clean_opportunities(pd.concat([raw, copy]))


def test_empty_dataset_does_not_overwrite_existing_output(tmp_path):
    from src.generate_data import generate_dataset
    from src.pipeline import run_pipeline
    raw = generate_dataset(20)
    raw["notes"] = ""
    source, target = tmp_path / "raw.csv", tmp_path / "clean.csv"
    raw.to_csv(source, index=False)
    target.write_text("previous valid output")
    with pytest.raises(ValueError, match="No valid"):
        run_pipeline(source, target)
    assert target.read_text() == "previous valid output"
