"""Data preprocessing utilities for MTHR/OCEAN dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# OCEAN axis order throughout the codebase: O, C, E, A, N
# Maps to MTHR/OCEAN column names: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
OCEAN_AXIS_ORDER = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

# Raw value range in MTHR/OCEAN dataset (Likert 1-5)
OCEAN_VALUE_MIN = 1.0
OCEAN_VALUE_MAX = 5.0


def normalize_ocean(
    df: pd.DataFrame,
    ocean_cols: list[str],
    value_min: float = OCEAN_VALUE_MIN,
    value_max: float = OCEAN_VALUE_MAX,
) -> pd.DataFrame:
    """Normalize OCEAN columns from [value_min, value_max] to [0, 1].

    Formula: ``(x - value_min) / (value_max - value_min)``

    For MTHR/OCEAN the raw range is [1.0, 5.0] (Likert scale).

    Args:
        df: DataFrame containing OCEAN columns.
        ocean_cols: List of column names to normalize.
        value_min: Minimum raw value (default 1.0 for MTHR/OCEAN).
        value_max: Maximum raw value (default 5.0 for MTHR/OCEAN).

    Returns:
        Copy of df with OCEAN columns normalized to float in [0, 1].
    """
    df = df.copy()
    span = float(value_max - value_min)
    for col in ocean_cols:
        df[col] = ((df[col].astype(float) - value_min) / span).clip(0.0, 1.0)
    return df


def row_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by rows (no user-id column available).

    Shuffles rows with a fixed seed and splits into train/val/test.
    Ratios: train=train_ratio, val=val_ratio, test=1-train_ratio-val_ratio.

    Args:
        df: DataFrame to split.
        train_ratio: Fraction of rows for training (default 0.8).
        val_ratio: Fraction of rows for validation (default 0.1).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple of (train_df, val_df, test_df), each reset_index(drop=True).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def build_ocean_pairs(
    df: pd.DataFrame,
    text_col: str,
    ocean_cols: list[str],
) -> list[dict[str, Any]]:
    """Build list of {text, ocean_profile} dicts from OCEAN DataFrame.

    Args:
        df: DataFrame with text and OCEAN columns (already normalized to [0,1]).
        text_col: Column name containing text.
        ocean_cols: Ordered list of OCEAN column names (canonical order O,C,E,A,N).

    Returns:
        List of dicts with keys 'text' (str) and 'ocean_profile' (list[float] len=5).
    """
    return [
        {
            "text": str(row[text_col]),
            "ocean_profile": [float(row[col]) for col in ocean_cols],
        }
        for _, row in df.iterrows()
    ]
