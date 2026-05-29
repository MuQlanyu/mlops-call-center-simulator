"""Tests for data preprocessing utilities (MTHR/OCEAN dataset)."""

import numpy as np
import pandas as pd

from call_center_simulator.data.preprocessing import (
    OCEAN_AXIS_ORDER,
    OCEAN_VALUE_MAX,
    OCEAN_VALUE_MIN,
    build_ocean_pairs,
    normalize_ocean,
    row_based_split,
)

# MTHR/OCEAN column names (canonical O, C, E, A, N order)
OCEAN_COLS = OCEAN_AXIS_ORDER  # ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _make_ocean_df(n: int = 20) -> pd.DataFrame:
    """Create a synthetic MTHR/OCEAN-style DataFrame with Likert [1,5] values."""
    rng = np.random.default_rng(42)
    data = {
        "Text": [f"Sample text number {i}." for i in range(n)],
    }
    for col in OCEAN_COLS:
        # Simulate Likert scale 1.0-5.0 (float, as in MTHR/OCEAN)
        data[col] = rng.uniform(1.0, 5.0, size=n).tolist()
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# normalize_ocean
# ---------------------------------------------------------------------------


def test_normalize_ocean_range():
    """All OCEAN values should be in [0, 1] after normalization."""
    df = _make_ocean_df(10)
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"


def test_normalize_ocean_dtype():
    """OCEAN columns should be float after normalization."""
    df = _make_ocean_df(10)
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].dtype == float


def test_normalize_ocean_min_maps_to_zero():
    """Raw minimum value (1.0) should map to exactly 0.0."""
    df = pd.DataFrame({"Text": ["x"], **{col: [OCEAN_VALUE_MIN] for col in OCEAN_COLS}})
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].iloc[0] == 0.0, f"{col}: min should map to 0.0"


def test_normalize_ocean_max_maps_to_one():
    """Raw maximum value (5.0) should map to exactly 1.0."""
    df = pd.DataFrame({"Text": ["x"], **{col: [OCEAN_VALUE_MAX] for col in OCEAN_COLS}})
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].iloc[0] == 1.0, f"{col}: max should map to 1.0"


def test_normalize_ocean_midpoint():
    """Raw midpoint (3.0 for [1,5]) should map to 0.5."""
    mid = (OCEAN_VALUE_MIN + OCEAN_VALUE_MAX) / 2.0  # 3.0
    df = pd.DataFrame({"Text": ["x"], **{col: [mid] for col in OCEAN_COLS}})
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert (
            abs(result[col].iloc[0] - 0.5) < 1e-9
        ), f"{col}: midpoint should map to 0.5"


# ---------------------------------------------------------------------------
# row_based_split
# ---------------------------------------------------------------------------


def test_row_based_split_sizes():
    """Total rows preserved; approximate split sizes match ratios."""
    df = _make_ocean_df(100)
    train, val, test = row_based_split(df, 0.8, 0.1, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) - 80) <= 5
    assert abs(len(val) - 10) <= 5


def test_row_based_split_no_overlap():
    """No row should appear in more than one split."""
    df = _make_ocean_df(60)
    # Add a unique id to track rows
    df["_id"] = range(len(df))
    train, val, test = row_based_split(df, 0.8, 0.1, seed=42)
    train_ids = set(train["_id"])
    val_ids = set(val["_id"])
    test_ids = set(test["_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_row_based_split_reproducible():
    """Same seed produces identical splits."""
    df = _make_ocean_df(50)
    train1, _, _ = row_based_split(df, 0.8, 0.1, seed=42)
    train2, _, _ = row_based_split(df, 0.8, 0.1, seed=42)
    assert list(train1.index) == list(train2.index)


def test_row_based_split_different_seeds():
    """Different seeds produce different row orderings."""
    df = _make_ocean_df(50)
    train1, _, _ = row_based_split(df, 0.8, 0.1, seed=42)
    train2, _, _ = row_based_split(df, 0.8, 0.1, seed=99)
    # Compare actual text content (not reset index) — very unlikely to be identical
    assert list(train1["Text"]) != list(train2["Text"])


# ---------------------------------------------------------------------------
# build_ocean_pairs
# ---------------------------------------------------------------------------


def test_build_ocean_pairs_structure():
    """Each pair has 'text' and 'ocean_profile' with 5 values in [0,1]."""
    df = _make_ocean_df(5)
    df = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    pairs = build_ocean_pairs(df, text_col="Text", ocean_cols=OCEAN_COLS)
    assert len(pairs) == 5
    for pair in pairs:
        assert "text" in pair
        assert "ocean_profile" in pair
        assert len(pair["ocean_profile"]) == 5
        for v in pair["ocean_profile"]:
            assert 0.0 <= v <= 1.0, f"ocean_profile value {v} out of [0,1]"


def test_build_ocean_pairs_text_preserved():
    """Text values are preserved correctly."""
    df = _make_ocean_df(3)
    df = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    pairs = build_ocean_pairs(df, text_col="Text", ocean_cols=OCEAN_COLS)
    for i, pair in enumerate(pairs):
        assert pair["text"] == df["Text"].iloc[i]
