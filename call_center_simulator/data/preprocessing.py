"""Data preprocessing utilities for Essays and PersonaChat datasets."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

# OCEAN axis order throughout the codebase: O, C, E, A, N
OCEAN_AXIS_ORDER = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]


def normalize_ocean(df: pd.DataFrame, ocean_cols: list[str]) -> pd.DataFrame:
    """Normalize OCEAN columns to float in [0, 1]. Essays uses binary {0,1}.

    Args:
        df: DataFrame containing OCEAN columns.
        ocean_cols: List of column names to normalize.

    Returns:
        Copy of df with OCEAN columns cast to float and clipped to [0, 1].
    """
    df = df.copy()
    for col in ocean_cols:
        df[col] = df[col].astype(float).clip(0.0, 1.0)
    return df


def user_based_split(
    df: pd.DataFrame,
    user_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by unique users to avoid data leakage.

    Guarantees no user appears in more than one split.
    Ratios: train=train_ratio, val=val_ratio, test=1-train_ratio-val_ratio.

    Args:
        df: DataFrame with a user identifier column.
        user_col: Column name containing user IDs.
        train_ratio: Fraction of users for training (default 0.8).
        val_ratio: Fraction of users for validation (default 0.1).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple of (train_df, val_df, test_df), each reset_index(drop=True).
    """
    users = list(df[user_col].unique())
    rng = random.Random(seed)
    rng.shuffle(users)
    n = len(users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_users = set(users[:n_train])
    val_users = set(users[n_train : n_train + n_val])
    test_users = set(users[n_train + n_val :])
    return (
        df[df[user_col].isin(train_users)].reset_index(drop=True),
        df[df[user_col].isin(val_users)].reset_index(drop=True),
        df[df[user_col].isin(test_users)].reset_index(drop=True),
    )


def build_essay_pairs(
    df: pd.DataFrame,
    text_col: str,
    ocean_cols: list[str],
) -> list[dict[str, Any]]:
    """Build list of {text, ocean_profile} dicts from Essays DataFrame.

    Args:
        df: DataFrame with text and OCEAN columns (already normalized).
        text_col: Column name containing essay text.
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


def build_dialog_pairs(
    raw_dialogs: list[dict[str, Any]],
    max_history: int = 10,
) -> list[dict[str, Any]]:
    """Build {history, response} pairs from PersonaChat raw dialogs.

    Args:
        raw_dialogs: List of dialog dicts, each with 'utterances' key.
            Each utterance has 'history' (list[str]) and 'candidates' (list[str]).
            The last candidate is the gold response.
        max_history: Maximum number of history turns to keep (default 10).

    Returns:
        List of dicts with keys 'history' (list[str]) and 'response' (str).
    """
    pairs = []
    for dialog in raw_dialogs:
        for utterance in dialog.get("utterances", []):
            candidates = utterance.get("candidates", [])
            if not candidates:
                continue
            history = utterance.get("history", [])
            pairs.append(
                {
                    "history": history[-max_history:],
                    "response": candidates[-1],
                }
            )
    return pairs
