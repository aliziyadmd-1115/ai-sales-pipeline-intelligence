from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
REGIONS = ["Northeast", "South", "Midwest", "West"]
INDUSTRIES = ["technology", "financial_services", "healthcare", "retail", "manufacturing"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
PRODUCTS = ["analytics", "automation", "data_platform", "security"]
STAGES = ["qualification", "discovery", "proposal", "negotiation"]

POSITIVE_NOTES = [
    "Executive sponsor is engaged and the buying team confirmed a clear business need.",
    "Technical evaluation is progressing well and the customer requested implementation details.",
    "Budget is approved and procurement has provided a target decision date.",
    "Decision makers attended the latest meeting and asked for a final proposal.",
]
NEUTRAL_NOTES = [
    "Customer is still comparing alternatives and gathering internal feedback.",
    "Discovery is ongoing and the team requested additional product information.",
    "The opportunity remains active but timing depends on the next planning cycle.",
    "Stakeholders are interested but the business case is still being refined.",
]
NEGATIVE_NOTES = [
    "Budget is uncertain and the main decision maker has not joined recent meetings.",
    "A strong competitor is already deployed and switching costs are a concern.",
    "The customer delayed the project and has not confirmed a decision timeline.",
    "Engagement has slowed and the proposal has not received internal sponsorship.",
]
WIN_REASONS = [
    "Strong executive sponsorship and clear ROI case.",
    "Successful technical evaluation with approved budget.",
    "Good product fit and timely stakeholder engagement.",
]
LOSS_REASONS = [
    "Budget or timing constraints prevented the purchase.",
    "Competitor preference and switching costs reduced win likelihood.",
    "Insufficient stakeholder engagement and unclear business case.",
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_dataset(rows: int = 3000) -> pd.DataFrame:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("rows must be a positive integer")
    rng = random.Random(RANDOM_SEED)
    start = datetime(2025, 1, 1, 8, 0, 0)
    records = []

    for i in range(1, rows + 1):
        region = rng.choice(REGIONS)
        industry = rng.choice(INDUSTRIES)
        segment = rng.choices(SEGMENTS, weights=[45, 35, 20], k=1)[0]
        product = rng.choice(PRODUCTS)
        stage = rng.choices(STAGES, weights=[26, 29, 27, 18], k=1)[0]
        value = round(rng.lognormvariate(10.25, 0.7), 2)
        value = min(max(value, 5000), 450000)
        days = rng.randint(5, 180)
        engagement = rng.randint(15, 100)
        meetings = rng.randint(0, 10)
        competitor = rng.random() < 0.48
        discount = round(rng.uniform(0, 0.28), 3)
        proposal = stage in {"proposal", "negotiation"} or rng.random() < 0.28

        latent = -1.2
        latent += (engagement - 55) / 28
        latent += min(meetings, 6) * 0.13
        latent += 0.65 if proposal else -0.15
        latent += 0.35 if stage == "negotiation" else (0.15 if stage == "proposal" else 0)
        latent += 0.20 if segment == "Mid-Market" else (0.10 if segment == "Enterprise" else 0)
        latent += 0.18 if product in {"analytics", "data_platform"} else 0
        latent -= 0.70 if competitor else 0
        latent -= max(0, discount - 0.18) * 5.0
        latent -= max(0, days - 100) / 85
        latent += rng.gauss(0, 0.65)

        win_prob = _sigmoid(latent * 1.35)
        outcome = "won" if rng.random() < win_prob else "lost"

        if engagement >= 72 and proposal and not competitor:
            note_pool = POSITIVE_NOTES
        elif engagement <= 42 or (competitor and days > 95):
            note_pool = NEGATIVE_NOTES
        else:
            note_pool = NEUTRAL_NOTES
        notes = rng.choice(note_pool)
        if rng.random() < 0.25:
            notes += f" Primary contact: buyer{i % 211:03d}@example.com."

        close_reason = rng.choice(WIN_REASONS if outcome == "won" else LOSS_REASONS)
        actual_revenue = round(value * rng.uniform(0.92, 1.03), 2) if outcome == "won" else 0.0
        created = start + timedelta(minutes=rng.randint(0, 620000))

        records.append({
            "opportunity_id": f"OPP-{i:05d}",
            "created_at": created.isoformat(),
            "owner_email": f"analyst{i % 57:02d}@example.com",
            "region": region,
            "industry": industry,
            "customer_segment": segment,
            "product_line": product,
            "sales_stage": stage,
            "estimated_value": value,
            "days_in_pipeline": days,
            "engagement_score": engagement,
            "meetings_count": meetings,
            "competitor_present": competitor,
            "discount_pct": discount,
            "proposal_sent": proposal,
            "notes": notes,
            "outcome": outcome,
            "actual_revenue": actual_revenue,
            "close_reason": close_reason,
        })

    # Deliberate data-quality issues for the pipeline to detect.
    df = pd.DataFrame(records)
    duplicate_sample = df.sample(min(6, rows), random_state=7)
    df = pd.concat([df, duplicate_sample], ignore_index=True)
    bad_idx = df.sample(min(8, len(df)), random_state=11).index
    df.loc[bad_idx[:4], "industry"] = "  " + df.loc[bad_idx[:4], "industry"].str.upper() + "  "
    df.loc[bad_idx[4:], "notes"] = "  " + df.loc[bad_idx[4:], "notes"] + "  "
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("data/opportunities_raw.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} raw rows to {args.output}")


if __name__ == "__main__":
    main()
