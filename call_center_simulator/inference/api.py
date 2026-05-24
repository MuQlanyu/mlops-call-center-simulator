"""FastAPI backend for call-center simulator inference."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from call_center_simulator.models.components.steering_vectors import SteeringVectors

logger = logging.getLogger(__name__)

# Module-level state (loaded on startup)
_backbone: Any = None
_tokenizer: Any = None
_steering: SteeringVectors | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load backbone and steering vectors on startup; clean up on shutdown."""
    global _backbone, _tokenizer, _steering

    backbone_name = os.environ.get("BACKBONE_NAME", "Qwen/Qwen3-0.6B")
    steering_ckpt = os.environ.get("STEERING_CKPT", "")

    logger.info("Loading backbone: %s", backbone_name)
    _tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _backbone = AutoModelForCausalLM.from_pretrained(
        backbone_name, torch_dtype=torch.float16
    )
    _backbone.eval()

    # hidden_size read from model.config — never hardcoded
    hidden_size: int = _backbone.config.hidden_size
    num_layers: int = _backbone.config.num_hidden_layers
    target_layer: int = num_layers // 2

    _steering = SteeringVectors(hidden_size=hidden_size)
    if steering_ckpt and Path(steering_ckpt).exists():
        state = torch.load(steering_ckpt, map_location="cpu")
        _steering.load_state_dict(state)
        logger.info("Loaded steering vectors from %s", steering_ckpt)
    _steering.register(_backbone, target_layer=target_layer)
    logger.info(
        "Model ready. hidden_size=%d, target_layer=%d", hidden_size, target_layer
    )

    yield

    # shutdown — remove hook if present
    if _steering is not None:
        try:
            _steering.remove_hook()
        except Exception:
            pass


app = FastAPI(title="Call-Center Simulator API", version="0.1.0", lifespan=lifespan)


class OceanProfile(BaseModel):
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)

    def to_tensor(self) -> torch.Tensor:
        """Return [1, 5] tensor in O, C, E, A, N order."""
        return torch.tensor(
            [
                [
                    self.openness,
                    self.conscientiousness,
                    self.extraversion,
                    self.agreeableness,
                    self.neuroticism,
                ]
            ],
            dtype=torch.float32,
        )


class DialogTurn(BaseModel):
    role: str
    text: str


class GenerateRequest(BaseModel):
    history: list[DialogTurn] = Field(default_factory=list)
    situation: str = ""
    ocean_profile: OceanProfile
    max_new_tokens: int = Field(default=128, ge=1, le=512)


class GenerateResponse(BaseModel):
    reply: str


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "call-center-simulator"}


@app.get("/health")
async def health() -> dict[str, str]:
    if _backbone is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate a client reply conditioned on OCEAN profile."""
    if _backbone is None or _tokenizer is None or _steering is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build prompt
    prompt_parts = []
    if request.situation:
        prompt_parts.append(f"Situation: {request.situation}")
    for turn in request.history[-10:]:
        prompt_parts.append(f"{turn.role}: {turn.text}")
    prompt_parts.append("client:")
    prompt = "\n".join(prompt_parts)

    inputs = _tokenizer(prompt, return_tensors="pt")
    ocean_tensor = request.ocean_profile.to_tensor()
    _steering.set_ocean_profile(ocean_tensor)

    with torch.no_grad():
        output_ids = _backbone.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    reply = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return GenerateResponse(reply=reply)
