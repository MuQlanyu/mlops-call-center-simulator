"""PyTorch Lightning module for training steering vectors."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from transformers import AutoModelForCausalLM

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead
from call_center_simulator.models.components.steering_vectors import SteeringVectors
from call_center_simulator.utils.metrics import compute_mape_ocean, compute_perplexity

logger = logging.getLogger(__name__)


class SteeringModule(LightningModule):
    """Trains steering vectors on frozen Qwen3-0.6B backbone.

    Trainable parameters: only SteeringVectors (5 * hidden_size = 5120 params).
    Backbone: frozen (requires_grad=False on all params).
    OCEAN classifier: frozen (loaded from checkpoint, requires_grad=False).

    Loss: CE_LM + lambda_steering * BCE(ocean_classifier(pooled_hidden), target_profile)
    """

    def __init__(
        self,
        backbone_name: str,
        ocean_classifier_ckpt: str | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lambda_steering: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        model_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lambda_steering = lambda_steering
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Frozen backbone — backbone params are frozen but steering vector
        # contribution at the hook layer must remain in the autograd graph.
        # We do NOT wrap the backbone in torch.no_grad(); we only set
        # requires_grad=False on its parameters.
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name, **(model_kwargs or {})
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        # hidden_size and num_layers read from model.config — never hardcoded
        hidden_size: int = self.backbone.config.hidden_size
        num_layers: int = self.backbone.config.num_hidden_layers
        target_layer: int = num_layers // 2

        # Trainable steering vectors (only trainable params in this module)
        self.steering = SteeringVectors(hidden_size=hidden_size)
        self.steering.register(self.backbone, target_layer=target_layer)

        # Frozen OCEAN classifier (loaded from Task 6 checkpoint)
        self.ocean_classifier = OceanClassifierHead(
            input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=5, dropout=dropout
        )
        if ocean_classifier_ckpt is not None:
            state = torch.load(ocean_classifier_ckpt, map_location="cpu")
            # Support both raw state_dict and Lightning checkpoint formats
            if "state_dict" in state:
                # Lightning checkpoint: extract classifier sub-keys
                prefix = "classifier."
                classifier_state = {
                    k[len(prefix) :]: v
                    for k, v in state["state_dict"].items()
                    if k.startswith(prefix)
                }
                if not classifier_state:
                    raise RuntimeError(
                        f"No keys with prefix '{prefix}' found in Lightning checkpoint. "
                        f"Available top-level prefixes: {sorted({k.split('.')[0] for k in state['state_dict']})}"
                    )
                self.ocean_classifier.load_state_dict(classifier_state)
            else:
                self.ocean_classifier.load_state_dict(state)
        for param in self.ocean_classifier.parameters():
            param.requires_grad = False

        self.bce_loss = nn.BCELoss()

    def _pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Mean-pool hidden states over non-padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        ocean_profile: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass with steering injection.

        The hook registered in __init__ injects the steering delta at the
        target layer during backbone.forward(). Gradients flow:
            loss → later layers → hook addition → steering.vectors

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            ocean_profile: [B, 5] target OCEAN profile, values in [0, 1]
            labels: [B, seq_len] for causal LM loss (None → ce_lm = 0)

        Returns:
            dict with keys: ce_lm, ocean_preds, pooled
        """
        self.steering.set_ocean_profile(ocean_profile)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        ce_lm = (
            outputs.loss
            if labels is not None
            else torch.tensor(0.0, device=self.device)
        )
        # Pool last hidden state for OCEAN classifier
        last_hidden = outputs.hidden_states[-1]
        pooled = self._pool(last_hidden, attention_mask)
        ocean_preds = self.ocean_classifier(pooled)
        # Clear stale profile to prevent leakage if backbone.generate() is called
        # without set_ocean_profile
        self.steering._ocean_profile = None
        return {"ce_lm": ce_lm, "ocean_preds": ocean_preds, "pooled": pooled}

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        input_ids, attention_mask, ocean_labels = batch
        # Shift labels for causal LM: mask padding positions
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        result = self(input_ids, attention_mask, ocean_labels, labels)
        bce = self.bce_loss(result["ocean_preds"], ocean_labels)
        loss = result["ce_lm"] + self.lambda_steering * bce
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ce_lm", result["ce_lm"], on_step=True, on_epoch=True)
        self.log("train_bce_ocean", bce, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, attention_mask, ocean_labels = batch
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        result = self(input_ids, attention_mask, ocean_labels, labels)
        bce = self.bce_loss(result["ocean_preds"], ocean_labels)
        loss = result["ce_lm"] + self.lambda_steering * bce
        mape, _ = compute_mape_ocean(
            result["ocean_preds"].detach().cpu(), ocean_labels.detach().cpu()
        )
        ppl = compute_perplexity(float(result["ce_lm"].detach().cpu()))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mape_ocean", mape, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", ppl, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Only steering vectors are trainable — verify this is the only param group
        trainable = list(self.steering.parameters())
        if not all(p.requires_grad for p in trainable):
            raise RuntimeError("Steering params must be trainable")
        return torch.optim.Adam(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay
        )
