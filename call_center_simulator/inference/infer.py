"""CLI inference: generate a client reply from command line."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from call_center_simulator.models.components.steering_vectors import SteeringVectors

logger = logging.getLogger(__name__)


def load_model(
    backbone_name: str,
    steering_ckpt: str | None = None,
) -> tuple[Any, Any, SteeringVectors]:
    """Load backbone, tokenizer, and steering vectors.

    Returns:
        Tuple of (model, tokenizer, steering_vectors).
    """
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        backbone_name, torch_dtype=torch.float16
    )
    model.eval()

    # hidden_size and num_layers read from model.config — never hardcoded
    hidden_size: int = model.config.hidden_size
    num_layers: int = model.config.num_hidden_layers
    target_layer: int = num_layers // 2

    steering = SteeringVectors(hidden_size=hidden_size)
    if steering_ckpt and Path(steering_ckpt).exists():
        state = torch.load(steering_ckpt, map_location="cpu")
        if "state_dict" in state:
            prefix = "steering."
            sv_state = {
                k[len(prefix) :]: v
                for k, v in state["state_dict"].items()
                if k.startswith(prefix)
            }
            if not sv_state:
                raise RuntimeError(
                    f"No keys with prefix '{prefix}' found in Lightning checkpoint. "
                    f"Available top-level prefixes: {sorted({k.split('.')[0] for k in state['state_dict']})}"
                )
            steering.load_state_dict(sv_state)
        else:
            steering.load_state_dict(state)
    steering.register(model, target_layer=target_layer)

    return model, tokenizer, steering


def generate_reply(
    model: Any,
    tokenizer: Any,
    steering: SteeringVectors,
    situation: str,
    history: list[dict[str, str]],
    ocean_profile: list[float],
    max_new_tokens: int = 128,
) -> str:
    """Generate a client reply.

    Args:
        model: Loaded backbone model.
        tokenizer: Loaded tokenizer.
        steering: SteeringVectors with hook registered.
        situation: Situation description string.
        history: List of {role, text} dicts.
        ocean_profile: List of 5 floats [O, C, E, A, N].
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated reply string.
    """
    prompt_parts = []
    if situation:
        prompt_parts.append(f"Situation: {situation}")
    for turn in history[-10:]:
        prompt_parts.append(f"{turn['role']}: {turn['text']}")
    prompt_parts.append("client:")
    prompt = "\n".join(prompt_parts)

    inputs = tokenizer(prompt, return_tensors="pt")
    ocean_tensor = torch.tensor([ocean_profile], dtype=torch.float32)
    steering.set_ocean_profile(ocean_tensor)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
