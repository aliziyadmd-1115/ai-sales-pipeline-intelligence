"""Regenerate the synthetic sample and its data-quality report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate_data import generate_dataset
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    args = parser.parse_args()
    Path("data").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    raw_path, clean_path = Path("data/opportunities_raw.csv"), Path("data/opportunities_clean.csv")
    generate_dataset(args.rows).to_csv(raw_path, index=False)
    quality = run_pipeline(raw_path, clean_path)
    Path("artifacts/data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
