"""Smoke tests for FastAPI /generate endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.testclient import TestClient

from call_center_simulator.inference.api import (
    GenerateRequest,
    GenerateResponse,
)


def _make_test_app() -> FastAPI:
    """Create FastAPI app with a stub model (no real Qwen3-0.6B)."""
    test_app = FastAPI()

    @test_app.get("/health")
    def health():
        return {"status": "healthy"}

    @test_app.post("/generate", response_model=GenerateResponse)
    def generate(request: Annotated[GenerateRequest, Body()]) -> GenerateResponse:
        # Stub: return a fixed reply without loading any model
        return GenerateResponse(reply="Stub reply for smoke test.")

    return test_app


def test_health_endpoint():
    """GET /health returns 200 with status healthy."""
    app = _make_test_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_generate_endpoint_returns_reply():
    """POST /generate returns a non-empty reply."""
    app = _make_test_app()
    client = TestClient(app)
    payload = {
        "history": [{"role": "operator", "text": "Hello, how can I help?"}],
        "situation": "Client calls about delayed delivery.",
        "ocean_profile": {
            "openness": 0.3,
            "conscientiousness": 0.7,
            "extraversion": 0.2,
            "agreeableness": 0.4,
            "neuroticism": 0.8,
        },
        "max_new_tokens": 64,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert len(data["reply"]) > 0


def test_generate_validates_ocean_range():
    """POST /generate rejects OCEAN values outside [0, 1]."""
    app = _make_test_app()
    client = TestClient(app)
    payload = {
        "history": [],
        "situation": "",
        "ocean_profile": {
            "openness": 1.5,  # invalid: > 1.0
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        },
        "max_new_tokens": 32,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 422  # Pydantic validation error
