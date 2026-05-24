"""OCEAN personality classifier head (MLP)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class OceanClassifierHead(nn.Module):
    """MLP head: Linear(input_dim->hidden_dim)->ReLU->Dropout->Linear(->5)->Sigmoid.

    Input: mean-pooled hidden state from backbone.
    Output: 5 values in [0,1], axis order O, C, E, A, N.
    Pass model.config.hidden_size as input_dim in production code.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: x shape [B, input_dim]. Returns: shape [B, 5]."""
        return self.net(x)
