"""Tests for data preprocessing utilities."""

import numpy as np
import pandas as pd

from call_center_simulator.data.preprocessing import (
    build_dialog_pairs,
    build_essay_pairs,
    normalize_ocean,
    user_based_split,
)

OCEAN_COLS = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]


def _make_essays_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "#AUTHID": [f"user_{i:03d}" for i in range(n)],
        "TEXT": [f"Sample essay text number {i}." for i in range(n)],
    }
    for col in OCEAN_COLS:
        data[col] = rng.integers(0, 2, size=n).tolist()
    return pd.DataFrame(data)


def test_normalize_ocean_range():
    df = _make_essays_df(10)
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].between(0.0, 1.0).all()


def test_normalize_ocean_dtype():
    df = _make_essays_df(10)
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].dtype == float


def test_user_based_split_sizes():
    df = _make_essays_df(100)
    train, val, test = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) - 80) <= 5
    assert abs(len(val) - 10) <= 5


def test_user_based_split_no_leakage():
    df = _make_essays_df(60)
    train, val, test = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    assert set(train["#AUTHID"]).isdisjoint(set(val["#AUTHID"]))
    assert set(train["#AUTHID"]).isdisjoint(set(test["#AUTHID"]))
    assert set(val["#AUTHID"]).isdisjoint(set(test["#AUTHID"]))


def test_user_based_split_reproducible():
    df = _make_essays_df(50)
    train1, _, _ = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    train2, _, _ = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    assert list(train1["#AUTHID"]) == list(train2["#AUTHID"])


def test_build_essay_pairs_structure():
    df = _make_essays_df(5)
    df = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    pairs = build_essay_pairs(df, text_col="TEXT", ocean_cols=OCEAN_COLS)
    assert len(pairs) == 5
    for pair in pairs:
        assert "text" in pair
        assert "ocean_profile" in pair
        assert len(pair["ocean_profile"]) == 5
        for v in pair["ocean_profile"]:
            assert 0.0 <= v <= 1.0


def test_build_dialog_pairs_structure():
    raw_dialogs = [
        {"utterances": [{"history": ["Hi", "Hello"], "candidates": ["How are you?"]}]},
        {"utterances": [{"history": ["Bye"], "candidates": ["Goodbye!"]}]},
    ]
    pairs = build_dialog_pairs(raw_dialogs, max_history=10)
    assert len(pairs) == 2
    for pair in pairs:
        assert "history" in pair
        assert "response" in pair
        assert isinstance(pair["history"], list)
        assert isinstance(pair["response"], str)
