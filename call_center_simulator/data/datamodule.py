"""PyTorch Lightning DataModules for Essays and PersonaChat datasets."""

from __future__ import annotations

import logging
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
    build_dialog_pairs,
    build_essay_pairs,
    normalize_ocean,
    user_based_split,
)

logger = logging.getLogger(__name__)


class EssaysDataModule(LightningDataModule):
    """DataModule for the Essays (Mairesse/Pennebaker) dataset."""

    def __init__(
        self,
        csv_path: Path | str,
        tokenizer_name: str,
        ocean_cols: list[str] | None = None,
        text_col: str = "TEXT",
        user_col: str = "#AUTHID",
        batch_size: int = 16,
        num_workers: int = 0,
        max_length: int = 512,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.tokenizer_name = tokenizer_name
        self.ocean_cols = ocean_cols or OCEAN_AXIS_ORDER
        self.text_col = text_col
        self.user_col = user_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> EssaysDataModule:
        return cls(
            csv_path=cfg.data.raw_path,
            tokenizer_name=cfg.model.backbone_name,
            ocean_cols=cfg.data.ocean_order,
            text_col=cfg.data.text_column,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            max_length=cfg.data.max_length,
            seed=cfg.seed,
        )

    def prepare_data(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Essays CSV not found: {self.csv_path}. Run: uv run dvc repro download"
            )

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.csv_path)
        df = normalize_ocean(df, self.ocean_cols)
        train_df, val_df, test_df = user_based_split(
            df, self.user_col, self.train_ratio, self.val_ratio, self.seed
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.train_dataset = self._make_dataset(train_df, tokenizer)
        self.val_dataset = self._make_dataset(val_df, tokenizer)
        self.test_dataset = self._make_dataset(test_df, tokenizer)

    def _make_dataset(self, df: pd.DataFrame, tokenizer: Any) -> TensorDataset:
        pairs = build_essay_pairs(df, self.text_col, self.ocean_cols)
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
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class PersonaChatDataModule(LightningDataModule):
    """DataModule for PersonaChat (dialog pairs for BLEU/ROUGE-L eval)."""

    def __init__(
        self,
        dataset_dir: Path | str,
        tokenizer_name: str,
        batch_size: int = 8,
        num_workers: int = 0,
        max_length: int = 256,
        max_history_turns: int = 10,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.max_history_turns = max_history_turns
        self.val_dataset: Dataset[Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        from datasets import load_from_disk  # type: ignore[import-untyped]

        dataset = load_from_disk(str(self.dataset_dir / "personachat"))
        pairs = build_dialog_pairs(list(dataset["validation"]), self.max_history_turns)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        contexts = [" ".join(p["history"]) for p in pairs]
        responses = [p["response"] for p in pairs]
        ctx_enc = tokenizer(
            contexts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        resp_enc = tokenizer(
            responses,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.val_dataset = TensorDataset(
            ctx_enc["input_ids"], ctx_enc["attention_mask"], resp_enc["input_ids"]
        )

    def val_dataloader(self) -> DataLoader[Any]:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
