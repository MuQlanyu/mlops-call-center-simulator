"""Typer CLI entry-point for call-center simulator."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import torch
import typer
import uvicorn
from transformers import AutoModel

from call_center_simulator.data.datamodule import preprocess_and_save
from call_center_simulator.data.download import download_ocean
from call_center_simulator.inference.app import main as inference_main
from call_center_simulator.inference.export_onnx import (
    export_ocean_classifier_onnx,
    verify_onnx,
)
from call_center_simulator.inference.infer import generate_reply, load_model
from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead

app = typer.Typer(help="Call-Center Simulator CLI")
logger = logging.getLogger(__name__)


@app.command()
def download_data() -> None:
    """Download MTHR/OCEAN dataset from HuggingFace Hub."""

    download_ocean(Path("data/raw/ocean"))
    typer.echo("Data downloaded successfully.")


@app.command("preprocess-data")
def preprocess_data() -> None:
    """Preprocess raw MTHR/OCEAN CSV → train/val/test CSVs."""

    preprocess_and_save(
        raw_csv=Path("data/raw/ocean/ocean_raw.csv"),
        output_dir=Path("data/processed"),
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )
    typer.echo("Preprocessing done.")


@app.command("train-ocean")
def train_ocean(
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Hydra overrides, e.g. train=smoke"),
    ] = None,
) -> None:
    """Train the OCEAN classifier head (Task 6)."""
    cmd = [
        sys.executable,
        "-m",
        "call_center_simulator.training.train_ocean_classifier",
    ]
    if overrides:
        cmd.extend(overrides)
    subprocess.run(cmd, check=True)


@app.command("train-steering")
def train_steering(
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Hydra overrides, e.g. train=smoke"),
    ] = None,
) -> None:
    """Train the steering vectors (Task 7)."""
    cmd = [sys.executable, "-m", "call_center_simulator.training.train"]
    if overrides:
        cmd.extend(overrides)
    subprocess.run(cmd, check=True)


@app.command()
def export_onnx(
    ckpt_path: str | None = typer.Option(None, help="Path to classifier checkpoint"),
    output: str = typer.Option("models/ocean_classifier.onnx", help="Output ONNX path"),
    backbone: str = typer.Option("Qwen/Qwen3-0.6B", help="Backbone model name"),
) -> None:
    """Export OCEAN classifier to ONNX format."""

    backbone_model = AutoModel.from_pretrained(backbone)
    hidden_size: int = backbone_model.config.hidden_size

    classifier = OceanClassifierHead(
        input_dim=hidden_size, hidden_dim=256, output_dim=5
    )
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        classifier.load_state_dict(state)

    onnx_path = Path(output)
    export_ocean_classifier_onnx(classifier, onnx_path, input_dim=hidden_size)
    verify_onnx(onnx_path, input_dim=hidden_size)
    typer.echo(f"ONNX exported to {onnx_path}")


@app.command()
def infer(
    situation: str = typer.Option("", help="Situation description"),
    history_json: str = typer.Option("[]", help="JSON list of {role, text} dicts"),
    openness: float = typer.Option(0.5),
    conscientiousness: float = typer.Option(0.5),
    extraversion: float = typer.Option(0.5),
    agreeableness: float = typer.Option(0.5),
    neuroticism: float = typer.Option(0.5),
    backbone: str = typer.Option("Qwen/Qwen3-0.6B"),
    steering_ckpt: str | None = typer.Option(None),
    max_tokens: int = typer.Option(128),
) -> None:
    """Generate a client reply from CLI."""

    history = json.loads(history_json)
    ocean_profile = [
        openness,
        conscientiousness,
        extraversion,
        agreeableness,
        neuroticism,
    ]
    model, tokenizer, steering = load_model(backbone, steering_ckpt)
    reply = generate_reply(
        model, tokenizer, steering, situation, history, ocean_profile, max_tokens
    )
    typer.echo(reply)


@app.command()
def serve_api(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Start FastAPI inference server."""

    uvicorn.run(
        "call_center_simulator.inference.api:app", host=host, port=port, reload=False
    )


@app.command()
def serve_ui(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(7860),
) -> None:
    """Start Gradio UI."""

    inference_main()


if __name__ == "__main__":
    app()
