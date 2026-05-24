"""Data download utilities for Essays and PersonaChat datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ESSAYS_PRIMARY_URL = (
    "https://raw.githubusercontent.com/SenticNet/personality-detection"
    "/master/data/essays.csv"
)
ESSAYS_FALLBACK = (
    "Manual download: http://farm2.user.srcf.net/research/personality/recognizer.html"
    " -> place essays.csv at data/raw/essays.csv"
)


def download_essays(output_path: Path, timeout: int = 60) -> None:
    """Download Essays CSV from GitHub mirror. Falls back with instructions.

    Args:
        output_path: Destination file path (e.g. Path("data/raw/essays.csv")).
        timeout: HTTP request timeout in seconds (default 60).

    Raises:
        RuntimeError: If download fails, with fallback instructions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info("Essays already at %s, skipping.", output_path)
        return
    logger.info("Downloading Essays from %s ...", ESSAYS_PRIMARY_URL)
    try:
        r = requests.get(ESSAYS_PRIMARY_URL, timeout=timeout)
        r.raise_for_status()
        output_path.write_bytes(r.content)
        logger.info("Saved %d bytes to %s.", len(r.content), output_path)
    except Exception as exc:
        logger.error("Download failed: %s\n%s", exc, ESSAYS_FALLBACK)
        raise RuntimeError(f"Essays download failed. {ESSAYS_FALLBACK}") from exc


def download_personachat(output_dir: Path) -> None:
    """Download PersonaChat via HuggingFace datasets library.

    Saves dataset to disk at output_dir/personachat using HF datasets
    save_to_disk format. Skips if already present.

    Args:
        output_dir: Directory to save the dataset (e.g. Path("data/raw")).
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "personachat"
    if cache_path.exists():
        logger.info("PersonaChat already at %s, skipping.", cache_path)
        return
    logger.info("Downloading PersonaChat (bavard/personachat_truecased)...")
    dataset = load_dataset("bavard/personachat_truecased")
    dataset.save_to_disk(str(cache_path))
    logger.info("PersonaChat saved to %s.", cache_path)


def main() -> None:
    """Download all datasets (called by DVC download stage)."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        version_base=None,
        config_path=str(Path(__file__).resolve().parent.parent.parent / "configs"),
        config_name="config",
    )
    def _main(cfg: DictConfig) -> None:
        download_essays(Path(cfg.data.raw_path))
        download_personachat(Path(cfg.paths.raw_data_dir))

    _main()


if __name__ == "__main__":
    main()
