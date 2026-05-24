"""Tests for SteeringVectors."""

import torch

from call_center_simulator.models.components.steering_vectors import SteeringVectors


def test_parameter_shape():
    # hidden_size=64 is a tiny proxy; real value is 1024 (derived from Qwen3-0.6B config)
    sv = SteeringVectors(hidden_size=64)
    assert sv.vectors.shape == (5, 64)


def test_parameter_is_trainable():
    sv = SteeringVectors(hidden_size=64)
    assert sv.vectors.requires_grad


def test_initial_values_zero():
    sv = SteeringVectors(hidden_size=64)
    assert torch.all(sv.vectors == 0.0)


def test_apply_delta_shape():
    sv = SteeringVectors(hidden_size=64)
    hidden = torch.randn(2, 10, 64)  # [B, seq_len, hidden_size]
    ocean_profile = torch.tensor([[0.3, 0.7, 0.2, 0.4, 0.8], [0.1, 0.5, 0.9, 0.3, 0.6]])
    result = sv.apply_delta(hidden, ocean_profile)
    assert result.shape == hidden.shape


def test_apply_delta_modifies_hidden():
    """apply_delta changes hidden states when vectors are non-zero."""
    sv = SteeringVectors(hidden_size=64)
    with torch.no_grad():
        sv.vectors.fill_(1.0)
    hidden = torch.zeros(2, 5, 64)
    ocean_profile = torch.ones(2, 5)
    result = sv.apply_delta(hidden, ocean_profile)
    # delta = ocean_profile @ vectors = [2,5] @ [5,64] = [2,64]
    # broadcast over seq_len: result != 0
    assert not torch.all(result == 0.0)


def test_gradient_flows_through_delta():
    """Gradients flow through apply_delta to steering vectors."""
    sv = SteeringVectors(hidden_size=64)
    hidden = torch.randn(2, 5, 64)
    ocean_profile = torch.rand(2, 5)
    result = sv.apply_delta(hidden, ocean_profile)
    result.sum().backward()
    assert sv.vectors.grad is not None
