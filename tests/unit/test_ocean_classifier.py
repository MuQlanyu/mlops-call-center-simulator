"""Tests for OceanClassifierHead."""

import torch

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead


def test_output_shape():
    # hidden_size=64 is a tiny proxy; real value is 1024 (derived from Qwen3-0.6B config)
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    out = model(torch.randn(4, 64))
    assert out.shape == (4, 5)


def test_output_range():
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    out = model(torch.randn(8, 64))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_dim_5():
    model = OceanClassifierHead(input_dim=128, hidden_dim=64, output_dim=5)
    out = model(torch.randn(1, 128))
    assert out.shape[-1] == 5


def test_no_nan():
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    out = model(torch.randn(16, 64))
    assert not torch.isnan(out).any()


def test_gradient_flows():
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    x = torch.randn(4, 64, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
