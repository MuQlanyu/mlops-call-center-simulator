"""PyTorch Lightning module for pre-training the OCEAN classifier head."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from transformers import AutoModel

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead
from call_center_simulator.utils.metrics import compute_mape_ocean

logger = logging.getLogger(__name__)


class OceanClassifierModule(LightningModule):
    """Trains OceanClassifierHead on frozen Qwen3-0.6B backbone. Loss: BCE."""

    def __init__(
        self,
        backbone_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.backbone = AutoModel.from_pretrained(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # hidden_size read from model.config — never hardcoded
        hidden_size: int = self.backbone.config.hidden_size
        self.classifier = OceanClassifierHead(
            input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=5, dropout=dropout
        )
        self.loss_fn = nn.BCELoss()

    def _pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self._pool(out.last_hidden_state, attention_mask))

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ids, mask, labels = batch
        loss = self.loss_fn(self(ids, mask), labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        ids, mask, labels = batch
        preds = self(ids, mask)
        loss = self.loss_fn(preds, labels)
        mape, _ = compute_mape_ocean(preds.detach().cpu(), labels.detach().cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mape_ocean", mape, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
