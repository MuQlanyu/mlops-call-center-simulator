"""PyTorch Lightning DataModule for MTHR/OCEAN dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from omegaconf import DictConfig

from call_center_simulator.data.preprocessing import (
    OCEAN_AXIS_ORDER,
    OCEAN_VALUE_MAX,
    OCEAN_VALUE_MIN,
    build_ocean_pairs,
    normalize_ocean,
    row_based_split,
)


class OceanDataModule(LightningDataModule):
    """DataModule for the MTHR/OCEAN dataset.

    Reads pre-processed CSV files (ocean_train.csv, ocean_val.csv, ocean_test.csv)
    from ``processed_dir``, tokenizes text, and returns
    ``TensorDataset(input_ids, attention_mask, ocean_labels)`` — the same
    batch format expected by ``OceanClassifierModule`` and ``SteeringModule``.
    """

    def __init__(
        self,
        processed_dir: Path | str,
        tokenizer_name: str,
        ocean_cols: list[str] | None = None,
        text_col: str = "Text",
        batch_size: int = 16,
        num_workers: int = 0,
        max_length: int = 256,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        value_min: float = OCEAN_VALUE_MIN,
        value_max: float = OCEAN_VALUE_MAX,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.tokenizer_name = tokenizer_name
        self.ocean_cols = ocean_cols or OCEAN_AXIS_ORDER
        self.text_col = text_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.value_min = value_min
        self.value_max = value_max
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None
        self._setup_done = False

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> OceanDataModule:
        """Construct from a Hydra DictConfig."""
        value_range = list(cfg.data.ocean_value_range)
        return cls(
            processed_dir=cfg.paths.processed_data_dir,
            tokenizer_name=cfg.model.backbone_name,
            ocean_cols=list(cfg.data.ocean_columns),
            text_col=cfg.data.text_column,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            max_length=cfg.data.max_length,
            train_ratio=cfg.data.train_split,
            val_ratio=cfg.data.val_split,
            seed=cfg.seed,
            value_min=float(value_range[0]),
            value_max=float(value_range[1]),
        )

    def prepare_data(self) -> None:
        """Check that processed CSVs exist."""
        for split in ("train", "val", "test"):
            p = self.processed_dir / f"ocean_{split}.csv"
            if not p.exists():
                raise FileNotFoundError(
                    f"Processed CSV not found: {p}. Run: uv run dvc repro preprocess"
                )

    def setup(self, stage: str | None = None) -> None:
        """Load and tokenize datasets (idempotent)."""
        if self._setup_done:
            return
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.train_dataset = self._load_split("train", tokenizer)
        self.val_dataset = self._load_split("val", tokenizer)
        self.test_dataset = self._load_split("test", tokenizer)
        self._setup_done = True

    def _load_split(self, split: str, tokenizer: Any) -> TensorDataset:
        csv_path = self.processed_dir / f"ocean_{split}.csv"
        df = pd.read_csv(csv_path)
        # Values are already normalized to [0,1] in the processed CSVs
        pairs = build_ocean_pairs(df, self.text_col, self.ocean_cols)
        texts = [p["text"] for p in pairs]
        labels = [p["ocean_profile"] for p in pairs]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return TensorDataset(
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(labels, dtype=torch.float32),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before test_dataloader()")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def preprocess_and_save(
    raw_csv: Path | str,
    output_dir: Path | str,
    ocean_cols: list[str] | None = None,
    value_min: float = OCEAN_VALUE_MIN,
    value_max: float = OCEAN_VALUE_MAX,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Normalize and split raw OCEAN CSV, saving three processed CSVs.

    Outputs: ``ocean_train.csv``, ``ocean_val.csv``, ``ocean_test.csv``
    in ``output_dir``.

    Args:
        raw_csv: Path to the raw CSV (e.g. ``data/raw/ocean/ocean_raw.csv``).
        output_dir: Directory to write processed CSVs.
        ocean_cols: OCEAN column names (default: OCEAN_AXIS_ORDER).
        value_min: Raw minimum value for normalization (default 1.0).
        value_max: Raw maximum value for normalization (default 5.0).
        train_ratio: Fraction for training split (default 0.8).
        val_ratio: Fraction for validation split (default 0.1).
        seed: Random seed (default 42).
    """
    cols = ocean_cols or OCEAN_AXIS_ORDER
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv)
    df = normalize_ocean(df, cols, value_min=value_min, value_max=value_max)
    train_df, val_df, test_df = row_based_split(df, train_ratio, val_ratio, seed)
    train_df.to_csv(output_dir / "ocean_train.csv", index=False)
    val_df.to_csv(output_dir / "ocean_val.csv", index=False)
    test_df.to_csv(output_dir / "ocean_test.csv", index=False)
