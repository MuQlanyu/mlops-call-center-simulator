"""Export OceanClassifierHead to ONNX format."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead

logger = logging.getLogger(__name__)


def export_ocean_classifier_onnx(
    model: OceanClassifierHead,
    output_path: Path,
    input_dim: int,
    opset_version: int = 17,
) -> None:
    """Export OceanClassifierHead to ONNX.

    Args:
        model: Trained OceanClassifierHead instance.
        output_path: Destination .onnx file path.
        input_dim: Input dimension (= backbone hidden_size).
        opset_version: ONNX opset version.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.zeros(1, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["hidden_states"],
        output_names=["ocean_scores"],
        dynamic_axes={
            "hidden_states": {0: "batch_size"},
            "ocean_scores": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    logger.info("ONNX model exported to %s", output_path)


def verify_onnx(onnx_path: Path, input_dim: int) -> None:
    """Verify ONNX model loads and produces correct output shape.

    Args:
        onnx_path: Path to the .onnx file.
        input_dim: Input dimension for dummy inference.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    dummy = np.zeros((2, input_dim), dtype=np.float32)
    outputs = sess.run(None, {"hidden_states": dummy})
    assert outputs[0].shape == (2, 5), f"Expected (2, 5), got {outputs[0].shape}"
    logger.info("ONNX verification passed: output shape %s", outputs[0].shape)


def main() -> None:
    """CLI entry point for ONNX export."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        version_base=None,
        config_path=str(Path(__file__).resolve().parent.parent.parent / "configs"),
        config_name="config",
    )
    def _main(cfg: DictConfig) -> None:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(cfg.model.backbone_name)
        hidden_size: int = backbone.config.hidden_size

        classifier = OceanClassifierHead(
            input_dim=hidden_size,
            hidden_dim=cfg.model.ocean_classifier.hidden_dim,
            output_dim=5,
        )
        ckpt_path = cfg.model.ocean_classifier.get("ckpt_path", None)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            classifier.load_state_dict(state)

        onnx_path = Path(cfg.model.ocean_onnx_path)
        export_ocean_classifier_onnx(classifier, onnx_path, input_dim=hidden_size)
        verify_onnx(onnx_path, input_dim=hidden_size)

    _main()


if __name__ == "__main__":
    main()
