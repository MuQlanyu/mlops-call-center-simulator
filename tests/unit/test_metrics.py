"""Tests for metrics utilities."""

import math

import pytest
import torch

from call_center_simulator.utils.metrics import (
    compute_bleu,
    compute_distinct,
    compute_mape_ocean,
    compute_perplexity,
    compute_rouge_l,
)


def test_mape_ocean_perfect():
    preds = torch.tensor([[0.3, 0.7, 0.2, 0.4, 0.8]])
    targets = torch.tensor([[0.3, 0.7, 0.2, 0.4, 0.8]])
    mape, per_axis = compute_mape_ocean(preds, targets)
    assert mape == pytest.approx(0.0, abs=1e-5)
    assert len(per_axis) == 5


def test_mape_ocean_known_value():
    """pred=0.5, target=1.0 -> MAPE=0.5 per axis."""
    preds = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    targets = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    mape, per_axis = compute_mape_ocean(preds, targets)
    assert mape == pytest.approx(0.5, abs=1e-5)
    assert all(v == pytest.approx(0.5, abs=1e-5) for v in per_axis)


def test_mape_ocean_batch():
    """MAPE averages correctly over batch."""
    preds = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    mape, _ = compute_mape_ocean(preds, targets)
    assert mape == pytest.approx(0.5, abs=1e-5)


def test_perplexity_known():
    assert compute_perplexity(2.0) == pytest.approx(math.exp(2.0), abs=1e-4)


def test_perplexity_zero_loss():
    assert compute_perplexity(0.0) == pytest.approx(1.0, abs=1e-5)


def test_bleu_identical():
    # Need >=4 words for BLEU-4 to compute all n-gram orders (1-4)
    score = compute_bleu(["hello world foo bar"], ["hello world foo bar"])
    assert score > 0.99


def test_bleu_empty():
    # No overlapping n-grams -> BLEU ~= 0 (method1 smoothing keeps it near 0)
    score = compute_bleu(["foo bar baz qux"], ["aaa bbb ccc ddd"])
    assert score == pytest.approx(0.0, abs=1e-5)


def test_rouge_l_identical():
    score = compute_rouge_l(["hello world"], ["hello world"])
    assert score == pytest.approx(1.0, abs=1e-3)


def test_rouge_l_empty():
    score = compute_rouge_l(["foo bar"], ["qux quux"])
    assert score == pytest.approx(0.0, abs=1e-3)


def test_distinct_range():
    texts = ["hello world foo bar", "baz qux hello world"]
    d1, d2 = compute_distinct(texts)
    assert 0.0 <= d1 <= 1.0
    assert 0.0 <= d2 <= 1.0


def test_distinct_all_same():
    """All tokens identical: distinct-1 = 1/4, distinct-2 = 1/3."""
    texts = ["a a a a"]
    d1, d2 = compute_distinct(texts)
    assert d1 == pytest.approx(1 / 4, abs=1e-5)
    assert d2 == pytest.approx(1 / 3, abs=1e-5)
