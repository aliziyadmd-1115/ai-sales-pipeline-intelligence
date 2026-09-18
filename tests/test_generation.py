import random

import pandas as pd
import pytest

from src.generate_data import generate_dataset
from src.pipeline import clean_opportunities


@pytest.mark.parametrize("rows", [1, 3, 8, 600])
def test_generation_supports_small_datasets_and_clean_duplicate_rows(rows):
    raw = generate_dataset(rows)
    clean, metrics = clean_opportunities(raw)
    assert len(clean) == rows
    assert metrics["duplicates_removed"] == min(6, rows)


def test_generation_is_repeatable_without_changing_global_rng():
    state = random.getstate()
    first = generate_dataset(20)
    assert random.getstate() == state
    pd.testing.assert_frame_equal(first, generate_dataset(20))
    with pytest.raises(ValueError, match="positive"):
        generate_dataset(0)
