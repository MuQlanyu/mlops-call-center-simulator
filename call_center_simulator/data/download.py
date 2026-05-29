"""Data download utilities for MTHR/OCEAN dataset."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "MTHR/OCEAN"


def download_ocean(output_dir: Path) -> None:
    """Download MTHR/OCEAN dataset from HuggingFace Hub and save as CSV.

    Saves the dataset split to ``output_dir/ocean_raw.csv``.
    Skips download if the file already exists.

    Args:
        output_dir: Directory to save the raw dataset
                    (e.g. ``Path("data/raw/ocean")``).
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "ocean_raw.csv"
    if out_csv.exists():
        logger.info("MTHR/OCEAN already at %s, skipping.", out_csv)
        return
    logger.info("Downloading %s from HuggingFace Hub...", HF_DATASET_NAME)
    dataset = load_dataset(HF_DATASET_NAME)
    # Dataset has only a "train" split (1160 rows)
    df = dataset["train"].to_pandas()
    df.to_csv(out_csv, index=False)
    logger.info(
        "Saved %d rows to %s.",
        len(df),
        out_csv,
    )


def main() -> None:
    """Download MTHR/OCEAN dataset (called by DVC download stage)."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        version_base=None,
        config_path=str(Path(__file__).resolve().parent.parent.parent / "configs"),
        config_name="config",
    )
    def _main(cfg: DictConfig) -> None:
        download_ocean(Path(cfg.data.raw_path))

    _main()


if __name__ == "__main__":
    main()
