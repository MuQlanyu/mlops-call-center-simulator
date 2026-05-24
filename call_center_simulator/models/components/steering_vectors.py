"""Steering vectors for OCEAN-conditioned generation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SteeringVectors(nn.Module):
    """5 learnable steering vectors, one per OCEAN axis.

    Shape: nn.Parameter([5, hidden_size]), initialized to zeros.
    Applied via forward hook on the target transformer layer.

    Usage:
        sv = SteeringVectors(hidden_size=model.config.hidden_size)
        sv.register(model, target_layer=model.config.num_hidden_layers // 2)
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize SteeringVectors.

        Args:
            hidden_size: Backbone hidden dimension.
                         Pass model.config.hidden_size in production code.
        """
        super().__init__()
        # Shape [5, hidden_size]: 5 OCEAN axes, axis order O, C, E, A, N
        self.vectors = nn.Parameter(torch.zeros(5, hidden_size))
        self._ocean_profile: Tensor | None = None
        self._hook_handle: torch.utils.hooks.RemovableHook | None = None

    def set_ocean_profile(self, ocean_profile: Tensor) -> None:
        """Set the OCEAN profile for the next forward pass.

        Args:
            ocean_profile: Tensor of shape [B, 5], values in [0, 1].
        """
        self._ocean_profile = ocean_profile

    def apply_delta(self, hidden_states: Tensor, ocean_profile: Tensor) -> Tensor:
        """Compute and add steering delta to hidden states.

        delta = ocean_profile @ vectors  -> shape [B, hidden_size]
        Broadcast over sequence length: hidden_states += delta.unsqueeze(1)

        Args:
            hidden_states: Tensor [B, seq_len, hidden_size].
            ocean_profile: Tensor [B, 5], values in [0, 1].

        Returns:
            Modified hidden states, same shape as input.
        """
        # Cast to match hidden_states dtype (backbone may use bfloat16/float16)
        vectors = self.vectors.to(dtype=hidden_states.dtype)
        profile = ocean_profile.to(dtype=hidden_states.dtype)
        # [B, 5] @ [5, hidden_size] -> [B, hidden_size]
        delta = profile @ vectors
        # Broadcast over seq_len: [B, 1, hidden_size]
        return hidden_states + delta.unsqueeze(1)

    def _make_hook(self, ocean_profile: Tensor):
        """Create a forward hook that injects the steering delta."""

        def hook(module, input, output):
            # output is a tuple; first element is hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            modified = self.apply_delta(hidden, ocean_profile)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    def register(self, model: nn.Module, target_layer: int) -> None:
        """Register forward hook on the target transformer layer.

        Args:
            model: The backbone transformer (Qwen3-0.6B).
            target_layer: Layer index = int(model.config.num_hidden_layers * target_layer_fraction).
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()

        # Access transformer layers (Qwen3 uses model.model.layers)
        layers = model.model.layers if hasattr(model, "model") else model.layers
        layer = layers[target_layer]

        def hook(module, input, output):
            if self._ocean_profile is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            modified = self.apply_delta(hidden, self._ocean_profile)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        self._hook_handle = layer.register_forward_hook(hook)

    def remove_hook(self) -> None:
        """Remove the registered forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
