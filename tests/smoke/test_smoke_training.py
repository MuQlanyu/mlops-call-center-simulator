"""Smoke tests for training pipeline using tiny-random model.

All tests use a tiny random transformer (num_hidden_layers=2, hidden_size=64)
to avoid downloading Qwen3-0.6B. Tests run on CPU in < 30 s total.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from lightning import LightningModule
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead
from call_center_simulator.models.components.steering_vectors import SteeringVectors
from call_center_simulator.models.ocean_classifier_module import OceanClassifierModule
from call_center_simulator.models.steering_module import SteeringModule

# Tiny model config — derived from Qwen3-0.6B config but shrunk for speed
TINY_HIDDEN = 64  # real Qwen3-0.6B: 1024
TINY_LAYERS = 2  # real Qwen3-0.6B: 28
TINY_VOCAB = 256  # real Qwen3-0.6B: 151936
BATCH = 2
SEQ_LEN = 16


def _make_tiny_causal_lm():
    """Create a tiny random causal LM for smoke tests."""
    config = AutoConfig.for_model(
        "qwen2",  # Qwen3 uses qwen2 architecture
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,  # must equal num_attention_heads for tiny config
        intermediate_size=128,
        vocab_size=TINY_VOCAB,
        max_position_embeddings=64,
    )
    return AutoModelForCausalLM.from_config(config)


def _make_tiny_encoder():
    """Create a tiny random encoder model for smoke tests."""
    config = AutoConfig.for_model(
        "qwen2",
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,  # must equal num_attention_heads for tiny config
        intermediate_size=128,
        vocab_size=TINY_VOCAB,
        max_position_embeddings=64,
    )
    return AutoModel.from_config(config)


def _make_dummy_batch():
    """Create a dummy batch: (input_ids, attention_mask, ocean_labels)."""
    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, SEQ_LEN))
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    ocean_labels = torch.rand(BATCH, 5)
    return input_ids, attention_mask, ocean_labels


def _make_patched_ocean_module() -> OceanClassifierModule:
    """Build an OceanClassifierModule with a tiny backbone, bypassing HF download."""
    # Call LightningModule.__init__ so all Lightning internals (_fabric, etc.) are set up
    module = OceanClassifierModule.__new__(OceanClassifierModule)
    LightningModule.__init__(module)
    module.backbone = _make_tiny_encoder()
    for p in module.backbone.parameters():
        p.requires_grad = False
    module.classifier = OceanClassifierHead(
        input_dim=TINY_HIDDEN, hidden_dim=32, output_dim=5
    )
    module.loss_fn = nn.BCELoss()
    module.learning_rate = 1e-3
    module.weight_decay = 0.0
    return module


def _make_patched_steering_module() -> SteeringModule:
    """Build a SteeringModule with a tiny backbone, bypassing HF download."""
    module = SteeringModule.__new__(SteeringModule)
    LightningModule.__init__(module)
    module.backbone = _make_tiny_causal_lm()
    for p in module.backbone.parameters():
        p.requires_grad = False

    module.steering = SteeringVectors(hidden_size=TINY_HIDDEN)
    module.steering.register(module.backbone, target_layer=TINY_LAYERS // 2)

    module.ocean_classifier = OceanClassifierHead(
        input_dim=TINY_HIDDEN, hidden_dim=32, output_dim=5
    )
    for p in module.ocean_classifier.parameters():
        p.requires_grad = False

    module.bce_loss = nn.BCELoss()
    module.lambda_steering = 0.1
    module.learning_rate = 1e-3
    module.weight_decay = 0.0
    return module


def test_ocean_classifier_head_smoke():
    """OceanClassifierHead forward pass with tiny hidden size."""
    head = OceanClassifierHead(input_dim=TINY_HIDDEN, hidden_dim=32, output_dim=5)
    x = torch.randn(BATCH, TINY_HIDDEN)
    out = head(x)
    assert out.shape == (BATCH, 5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_steering_vectors_hook_applied():
    """SteeringVectors hook modifies hidden states in a tiny model."""
    model = _make_tiny_causal_lm()
    for p in model.parameters():
        p.requires_grad = False

    sv = SteeringVectors(hidden_size=TINY_HIDDEN)
    with torch.no_grad():
        sv.vectors.fill_(0.1)  # non-zero so delta is detectable

    # Register hook on layer 1 (= TINY_LAYERS // 2)
    sv.register(model, target_layer=TINY_LAYERS // 2)

    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, SEQ_LEN))
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    ocean_profile = torch.ones(BATCH, 5) * 0.5
    sv.set_ocean_profile(ocean_profile)

    with torch.no_grad():
        out_with_hook = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    sv.remove_hook()
    with torch.no_grad():
        out_without_hook = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Hidden states at target layer should differ
    h_with = out_with_hook.hidden_states[TINY_LAYERS // 2 + 1]
    h_without = out_without_hook.hidden_states[TINY_LAYERS // 2 + 1]
    assert not torch.allclose(h_with, h_without), "Hook did not modify hidden states"


def test_ocean_classifier_module_train_step():
    """OceanClassifierModule.training_step runs without error on tiny model."""
    module = _make_patched_ocean_module()
    batch = _make_dummy_batch()
    loss = module.training_step(batch, 0)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_steering_module_train_step():
    """SteeringModule.training_step runs without error on tiny model."""
    module = _make_patched_steering_module()
    batch = _make_dummy_batch()
    loss = module.training_step(batch, 0)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_steering_vectors_gradient_update():
    """Steering vectors receive gradient after backward pass."""
    model = _make_tiny_causal_lm()
    for p in model.parameters():
        p.requires_grad = False

    sv = SteeringVectors(hidden_size=TINY_HIDDEN)
    sv.register(model, target_layer=TINY_LAYERS // 2)

    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, SEQ_LEN))
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    ocean_profile = torch.rand(BATCH, 5)
    sv.set_ocean_profile(ocean_profile)

    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    assert sv.vectors.grad is not None, "No gradient on steering vectors"
    assert not torch.isnan(sv.vectors.grad).any()
