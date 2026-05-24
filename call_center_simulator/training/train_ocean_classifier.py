"""Entry point: pre-train OCEAN classifier head on Essays dataset."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import hydra
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from call_center_simulator.data.datamodule import EssaysDataModule
from call_center_simulator.models.ocean_classifier_module import OceanClassifierModule

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent.parent.parent / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Train OCEAN classifier head."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=(cfg.mlflow.run_name or "ocean-classifier"),
    )
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        git_commit = "unknown"
    mlflow_logger.log_hyperparams({"git_commit": git_commit})
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    datamodule = EssaysDataModule.from_hydra_config(cfg)
    model = OceanClassifierModule(
        backbone_name=cfg.model.backbone_name,
        hidden_dim=cfg.model.ocean_classifier.hidden_dim,
        dropout=cfg.model.ocean_classifier.dropout,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.models_dir,
            filename=cfg.train.checkpoint.filename,
            monitor=cfg.train.checkpoint.monitor,
            mode=cfg.train.checkpoint.mode,
            save_top_k=cfg.train.checkpoint.save_top_k,
        ),
        EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
            mode=cfg.train.early_stopping.mode,
        ),
    ]
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=mlflow_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
