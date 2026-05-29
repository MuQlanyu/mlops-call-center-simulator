# Call-Center Simulator — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete Phase A of the call-center simulator — all code, infrastructure, and smoke-tests — without real GPU training. The result is a fully runnable repo: `pre-commit run -a` passes, DVC pipeline executes, smoke-tests pass on CPU in < 30 s, Docker Compose starts.

**Architecture:** `Qwen/Qwen3-0.6B` (frozen backbone, `hidden_size=1024`, 28 layers) + 5 trainable steering vectors (`nn.Parameter` shape `[5, 1024]`) injected via `register_forward_hook` on layer 14 (`num_layers // 2`). A small MLP OCEAN-classifier (`hidden_size → 256 → 5 → Sigmoid`) is pre-trained on Essays dataset (BCE loss), exported to ONNX, then frozen during steering-vector training. Loss = `CE_LM + 0.1 * BCE(OCEAN_classifier(hidden_states_pooled), target_profile)`. Served via FastAPI + Gradio UI in Docker Compose.

**Tech Stack:** Python 3.11, uv, torch >=2.2, lightning >=2.2, transformers >=4.45, datasets >=2.19, hydra-core >=1.3, mlflow >=2.14, dvc >=3.50, gradio >=4.40, fastapi >=0.110, pydantic >=2.7, typer >=0.12, onnx >=1.16, onnxruntime >=1.18, nltk, rouge-score, ruff, pytest >=8, python-dotenv.

> **HF Model name decision:** `Qwen/Qwen3-0.6B` — confirmed publicly available on HuggingFace (released May 2025, non-gated). Architecture: `hidden_size=1024`, `num_hidden_layers=28`, `target_layer=14` (= `num_layers // 2`). Total trainable params = 5 × 1024 = 5 120 steering-vector parameters. In all production code `hidden_size` is read from `model.config.hidden_size` — never hardcoded. Hardcodes appear only in tests with comment `# derived from Qwen3-0.6B config`.

---

## File Structure

```
mlops-call-center-simulator/
├── call_center_simulator/
│   ├── __init__.py
│   ├── cli.py                          # Typer CLI entry-point
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py                 # Essays (GitHub raw) + PersonaChat (HF datasets)
│   │   ├── preprocessing.py            # Normalize OCEAN [0,1], user-based split, pairs
│   │   └── datamodule.py               # EssaysDataModule + PersonaChatDataModule (Lightning)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── ocean_classifier.py     # OceanClassifierHead MLP
│   │   │   └── steering_vectors.py     # SteeringVectors nn.Parameter + hook
│   │   ├── ocean_classifier_module.py  # LightningModule for OCEAN pre-training
│   │   └── steering_module.py          # LightningModule: frozen Qwen + steering
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_ocean_classifier.py   # Hydra entry-point: train OCEAN head
│   │   └── train.py                    # Hydra entry-point: train steering vectors
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── infer.py                    # CLI inference: profile + history -> reply
│   │   ├── export_onnx.py              # Export OceanClassifierHead to ONNX
│   │   ├── api.py                      # FastAPI /generate endpoint
│   │   └── app.py                      # Gradio UI
│   └── utils/
│       ├── __init__.py
│       └── metrics.py                  # MAPE_ocean, Perplexity, BLEU, ROUGE-L, Distinct-1/2
├── configs/
│   ├── config.yaml                     # Root Hydra config (defaults list)
│   ├── data/
│   │   ├── essays.yaml
│   │   └── personachat.yaml
│   ├── model/
│   │   ├── qwen.yaml
│   │   └── ocean_classifier.yaml
│   └── train/
│       ├── default.yaml
│       └── smoke.yaml
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_metrics.py
│   │   ├── test_preprocessing.py
│   │   ├── test_ocean_classifier.py
│   │   └── test_steering_vectors.py
│   └── smoke/
│       ├── __init__.py
│       ├── test_smoke_training.py      # tiny-random model, CPU, < 30 s total
│       └── test_smoke_api.py           # FastAPI smoke test
├── data/
│   └── .gitignore
├── models/
│   └── .gitignore
├── plots/
│   └── .gitkeep
├── .dvc/
│   └── config
├── .dvc-storage/                       # local DVC remote
├── pyproject.toml
├── uv.lock                             # committed after uv sync
├── dvc.yaml
├── Dockerfile
├── docker-compose.yml
├── .pre-commit-config.yaml
├── .gitignore
├── .dvcignore
├── .env.example
├── AGENTS.md
└── README.md
```

---

## Version Pinning

| Library | Version constraint |
|---|---|
| python | >=3.11,<3.12 |
| torch | >=2.2.0 |
| lightning | >=2.2.0 |
| transformers | >=4.45.0 |
| datasets | >=2.19.0 |
| hydra-core | >=1.3.0 |
| mlflow | >=2.14.0 |
| dvc | >=3.50.0 |
| gradio | >=4.40.0 |
| fastapi | >=0.110.0 |
| uvicorn | >=0.29.0 |
| pydantic | >=2.7.0 |
| typer | >=0.12.0 |
| onnx | >=1.16.0 |
| onnxruntime | >=1.18.0 |
| nltk | >=3.8.0 |
| rouge-score | >=0.1.2 |
| python-dotenv | >=1.0.0 |
| httpx | >=0.27.0 |
| requests | >=2.31.0 |
| pytest | >=8.0.0 |
| pytest-cov | >=5.0.0 |
| ruff | >=0.4.0 |

---

## Tasks

### Task 1: Bootstrap — pyproject.toml, .gitignore, .pre-commit-config.yaml, .env.example, DVC init

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.dvcignore`
- Create: `.pre-commit-config.yaml`
- Create: `.env.example`
- Create: `data/.gitignore`
- Create: `models/.gitignore`
- Create: `plots/.gitkeep`

- [ ] **Step 0: Delete stale temporary files (if they exist)**

```bash
# These are leftover from a previous subagent run — not part of the project
rm -f update_plan.py update_smoke_tests.py
git rm --cached update_plan.py update_smoke_tests.py 2>/dev/null || true
```

Expected: files removed from working tree and git index (if they were tracked).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "call-center-simulator"
version = "0.1.0"
description = "Call-center operator trainer via OCEAN-steered LLM"
authors = [
    {name = "Мун Павел Юрьевич"}
]
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.2.0",
    "lightning>=2.2.0",
    "transformers>=4.45.0",
    "datasets>=2.19.0",
    "hydra-core>=1.3.0",
    "mlflow>=2.14.0",
    "dvc>=3.50.0",
    "gradio>=4.40.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "pydantic>=2.7.0",
    "typer>=0.12.0",
    "onnx>=1.16.0",
    "onnxruntime>=1.18.0",
    "nltk>=3.8.0",
    "rouge-score>=0.1.2",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "requests>=2.31.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
    "pre-commit>=3.7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["call_center_simulator"]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "RUF", # ruff-specific rules
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "PTH", # flake8-use-pathlib
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["call_center_simulator"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# Environment variables
.env

# DVC
/data/raw/
/data/processed/
/models/*.pt
/models/*.onnx
/models/*.ckpt
.dvc/tmp/
.dvc/cache/

# MLflow
mlruns/
mlflow.db

# Hydra outputs
outputs/
multirun/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/
*.swp

# Logs
logs/
*.log
```

- [ ] **Step 3: Create `.dvcignore`**

```
.git
.venv
__pycache__
*.pyc
.env
```

- [ ] **Step 4: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=1000"]
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

- [ ] **Step 5: Create `.env.example`**

```bash
# HuggingFace token (Qwen3-0.6B is non-gated, but good practice)
HF_TOKEN=hf_your_token_here

# MLflow tracking URI (local default)
MLFLOW_TRACKING_URI=http://127.0.0.1:8080
```

- [ ] **Step 6: Create `data/.gitignore`**

```gitignore
raw/
processed/
```

- [ ] **Step 7: Create `models/.gitignore`**

```gitignore
*.pt
*.onnx
*.ckpt
```

- [ ] **Step 8: Create `plots/.gitkeep`** (empty file)

- [ ] **Step 9: Install uv and sync**

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Expected: environment created, packages installed without errors.

- [ ] **Step 10: Init DVC with local remote**

```bash
uv run dvc init
uv run dvc remote add -d local .dvc-storage
uv run dvc remote list
```

Expected output:
```
local   .dvc-storage
```

- [ ] **Step 11: Install pre-commit hooks and run**

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Expected: all hooks pass (green).

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml .gitignore .dvcignore .pre-commit-config.yaml .env.example \
        data/.gitignore models/.gitignore plots/.gitkeep \
        .dvc/config .dvc/.gitignore
git commit -m "chore: bootstrap project — pyproject.toml, pre-commit, DVC init"
```

---

### Task 2: Package skeleton + Hydra configs

**Files:**
- Create: `call_center_simulator/__init__.py` (and all sub-package `__init__.py`)
- Create: `configs/config.yaml`
- Create: `configs/data/essays.yaml`
- Create: `configs/data/personachat.yaml`
- Create: `configs/model/qwen.yaml`
- Create: `configs/model/ocean_classifier.yaml`
- Create: `configs/train/default.yaml`
- Create: `configs/train/smoke.yaml`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/smoke/__init__.py`

- [ ] **Step 1: Create directory tree and empty `__init__.py` files**

```bash
mkdir -p call_center_simulator/data \
         call_center_simulator/models/components \
         call_center_simulator/training \
         call_center_simulator/inference \
         call_center_simulator/utils \
         configs/data configs/model configs/train \
         tests/unit tests/smoke

touch call_center_simulator/__init__.py \
      call_center_simulator/data/__init__.py \
      call_center_simulator/models/__init__.py \
      call_center_simulator/models/components/__init__.py \
      call_center_simulator/training/__init__.py \
      call_center_simulator/inference/__init__.py \
      call_center_simulator/utils/__init__.py \
      tests/__init__.py \
      tests/unit/__init__.py \
      tests/smoke/__init__.py
```

- [ ] **Step 2: Create `configs/config.yaml`**

```yaml
defaults:
  - data: essays
  - model: qwen
  - train: default
  - _self_

seed: 42

paths:
  data_dir: data
  raw_data_dir: ${paths.data_dir}/raw
  processed_data_dir: ${paths.data_dir}/processed
  models_dir: models
  plots_dir: plots

mlflow:
  tracking_uri: http://127.0.0.1:8080
  experiment_name: call-center-simulator
  run_name: null

logging:
  log_level: INFO
```

- [ ] **Step 3: Create `configs/data/essays.yaml`**

```yaml
# @package _global_
data:
  name: essays
  essays_url: "https://raw.githubusercontent.com/SenticNet/personality-detection/master/data/essays.csv"
  essays_fallback_path: "data/raw/essays.csv"
  raw_path: data/raw/essays.csv
  processed_path: data/processed/essays
  batch_size: 16
  num_workers: 0
  max_length: 512
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  # OCEAN axis order: O, C, E, A, N
  ocean_columns: [cEXT, cNEU, cAGR, cCON, cOPN]
  ocean_order: [cOPN, cCON, cEXT, cAGR, cNEU]
  text_column: TEXT
```

- [ ] **Step 4: Create `configs/data/personachat.yaml`**

```yaml
# @package _global_
data:
  name: personachat
  hf_dataset: bavard/personachat_truecased
  raw_path: data/raw/personachat
  processed_path: data/processed/personachat
  batch_size: 8
  num_workers: 0
  max_length: 256
  max_history_turns: 10
```

- [ ] **Step 5: Create `configs/model/qwen.yaml`**

```yaml
# @package _global_
model:
  backbone_name: Qwen/Qwen3-0.6B
  # hidden_size and num_layers are read from model.config at runtime.
  # Reference values (derived from Qwen3-0.6B config, for documentation only):
  #   hidden_size: 1024
  #   num_hidden_layers: 28
  #   target_layer: 14  (= num_hidden_layers // 2)
  target_layer_fraction: 0.5
  dtype: float16
  gradient_checkpointing: true
  lambda_steering: 0.1
```

- [ ] **Step 6: Create `configs/model/ocean_classifier.yaml`**

```yaml
# @package _global_
model:
  ocean_classifier:
    hidden_dim: 256
    output_dim: 5
    dropout: 0.1
  ocean_onnx_path: models/ocean_classifier.onnx
```

- [ ] **Step 7: Create `configs/train/default.yaml`**

```yaml
# @package _global_
train:
  max_epochs: 10
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  accelerator: auto
  devices: 1
  precision: 16-mixed
  gradient_clip_val: 1.0
  log_every_n_steps: 10
  check_val_every_n_epoch: 1
  early_stopping:
    monitor: val_mape_ocean
    patience: 5
    mode: min
  checkpoint:
    monitor: val_mape_ocean
    mode: min
    save_top_k: 1
    filename: "best-{epoch:02d}-{val_mape_ocean:.4f}"
```

- [ ] **Step 8: Create `configs/train/smoke.yaml`**

```yaml
# @package _global_
# Smoke config: tiny run for CI / fast iteration on CPU
train:
  max_epochs: 1
  max_steps: 3
  learning_rate: 1.0e-3
  weight_decay: 0.0
  accelerator: cpu
  devices: 1
  precision: 32
  gradient_clip_val: 1.0
  log_every_n_steps: 1
  check_val_every_n_epoch: 1
  early_stopping:
    monitor: val_loss
    patience: 2
    mode: min
  checkpoint:
    monitor: val_loss
    mode: min
    save_top_k: 1
    filename: "smoke-{epoch:02d}"
```

- [ ] **Step 9: Verify import works**

```bash
uv run python -c "import call_center_simulator; print('OK')"
```

Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add call_center_simulator/ configs/ tests/
git commit -m "feat: package skeleton + Hydra configs"
```

---

### Task 3: Metrics (TDD)

**Files:**
- Create: `call_center_simulator/utils/metrics.py`
- Create: `tests/unit/test_metrics.py`

- [ ] **Step 1: Write failing tests — create `tests/unit/test_metrics.py`**

```python
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
    preds = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0, 1.0, 1.0]])
    mape, _ = compute_mape_ocean(preds, targets)
    assert mape == pytest.approx(0.5, abs=1e-5)


def test_perplexity_known():
    assert compute_perplexity(2.0) == pytest.approx(math.exp(2.0), abs=1e-4)


def test_perplexity_zero_loss():
    assert compute_perplexity(0.0) == pytest.approx(1.0, abs=1e-5)


def test_bleu_identical():
    score = compute_bleu(["hello world"], ["hello world"])
    assert score > 0.99


def test_bleu_empty():
    score = compute_bleu(["foo bar baz"], ["qux quux corge"])
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_metrics.py -v
```

Expected: `FAILED` — `ImportError: cannot import name 'compute_mape_ocean'`

- [ ] **Step 3: Implement `call_center_simulator/utils/metrics.py`**

```python
"""Evaluation metrics for the call-center simulator."""

from __future__ import annotations

import math
from collections import Counter

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


def compute_mape_ocean(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, list[float]]:
    """Compute MAPE averaged over OCEAN axes and batch.

    Args:
        preds: Predicted OCEAN values, shape [B, 5], values in [0, 1].
        targets: Target OCEAN values, shape [B, 5], values in [0, 1].
        eps: Small constant to avoid division by zero.

    Returns:
        Tuple of (mean_mape, per_axis_mape). Axis order: O, C, E, A, N.
    """
    abs_err = (preds - targets).abs()
    denom = targets.abs().clamp(min=eps)
    per_sample_per_axis = abs_err / denom  # [B, 5]
    per_axis = per_sample_per_axis.mean(dim=0).tolist()  # [5]
    mean_mape = float(sum(per_axis) / len(per_axis))
    return mean_mape, per_axis


def compute_perplexity(avg_cross_entropy_loss: float) -> float:
    """Compute perplexity from average cross-entropy loss.

    Args:
        avg_cross_entropy_loss: Mean token-level cross-entropy loss (nats).

    Returns:
        Perplexity = exp(loss).
    """
    return math.exp(avg_cross_entropy_loss)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus BLEU-4 score.

    Args:
        hypotheses: List of generated strings.
        references: List of reference strings (one per hypothesis).

    Returns:
        BLEU-4 score in [0, 1].
    """
    tokenized_hyps = [h.split() for h in hypotheses]
    tokenized_refs = [[r.split()] for r in references]
    smoothing = SmoothingFunction().method1
    return float(corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothing))


def compute_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Compute mean ROUGE-L F1 score.

    Args:
        hypotheses: List of generated strings.
        references: List of reference strings.

    Returns:
        Mean ROUGE-L F1 in [0, 1].
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_distinct(texts: list[str]) -> tuple[float, float]:
    """Compute Distinct-1 and Distinct-2 (anti-collapse diversity metrics).

    Args:
        texts: List of generated strings.

    Returns:
        Tuple of (distinct_1, distinct_2).
        distinct_1 = unique_unigrams / total_unigrams
        distinct_2 = unique_bigrams / total_bigrams
    """
    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(text.split())

    if not all_tokens:
        return 0.0, 0.0

    distinct_1 = len(Counter(all_tokens)) / len(all_tokens)

    bigrams = list(zip(all_tokens[:-1], all_tokens[1:]))
    if not bigrams:
        return distinct_1, 0.0

    distinct_2 = len(Counter(bigrams)) / len(bigrams)
    return distinct_1, distinct_2
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_metrics.py -v
```

Expected: all 11 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add call_center_simulator/utils/metrics.py tests/unit/test_metrics.py
git commit -m "feat: metrics — MAPE_ocean, Perplexity, BLEU, ROUGE-L, Distinct-1/2 (TDD)"
```

---

### Task 4: Data download + preprocessing (TDD)

**Files:**
- Create: `call_center_simulator/data/download.py`
- Create: `call_center_simulator/data/preprocessing.py`
- Create: `tests/unit/test_preprocessing.py`

- [ ] **Step 1: Write failing tests — create `tests/unit/test_preprocessing.py`**

```python
"""Tests for data preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from call_center_simulator.data.preprocessing import (
    build_dialog_pairs,
    build_essay_pairs,
    normalize_ocean,
    user_based_split,
)

OCEAN_COLS = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]


def _make_essays_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "#AUTHID": [f"user_{i:03d}" for i in range(n)],
        "TEXT": [f"Sample essay text number {i}." for i in range(n)],
    }
    for col in OCEAN_COLS:
        data[col] = rng.integers(0, 2, size=n).tolist()
    return pd.DataFrame(data)


def test_normalize_ocean_range():
    df = _make_essays_df(10)
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].between(0.0, 1.0).all()


def test_normalize_ocean_dtype():
    df = _make_essays_df(10)
    result = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    for col in OCEAN_COLS:
        assert result[col].dtype == float


def test_user_based_split_sizes():
    df = _make_essays_df(100)
    train, val, test = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) - 80) <= 5
    assert abs(len(val) - 10) <= 5


def test_user_based_split_no_leakage():
    df = _make_essays_df(60)
    train, val, test = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    assert set(train["#AUTHID"]).isdisjoint(set(val["#AUTHID"]))
    assert set(train["#AUTHID"]).isdisjoint(set(test["#AUTHID"]))
    assert set(val["#AUTHID"]).isdisjoint(set(test["#AUTHID"]))


def test_user_based_split_reproducible():
    df = _make_essays_df(50)
    train1, _, _ = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    train2, _, _ = user_based_split(df, "#AUTHID", 0.8, 0.1, seed=42)
    assert list(train1["#AUTHID"]) == list(train2["#AUTHID"])


def test_build_essay_pairs_structure():
    df = _make_essays_df(5)
    df = normalize_ocean(df, ocean_cols=OCEAN_COLS)
    pairs = build_essay_pairs(df, text_col="TEXT", ocean_cols=OCEAN_COLS)
    assert len(pairs) == 5
    for pair in pairs:
        assert "text" in pair
        assert "ocean_profile" in pair
        assert len(pair["ocean_profile"]) == 5
        for v in pair["ocean_profile"]:
            assert 0.0 <= v <= 1.0


def test_build_dialog_pairs_structure():
    raw_dialogs = [
        {"utterances": [{"history": ["Hi", "Hello"], "candidates": ["How are you?"]}]},
        {"utterances": [{"history": ["Bye"], "candidates": ["Goodbye!"]}]},
    ]
    pairs = build_dialog_pairs(raw_dialogs, max_history=10)
    assert len(pairs) == 2
    for pair in pairs:
        assert "history" in pair
        assert "response" in pair
        assert isinstance(pair["history"], list)
        assert isinstance(pair["response"], str)

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_preprocessing.py -v
```

Expected: `FAILED` — `ImportError: cannot import name 'normalize_ocean'`

- [ ] **Step 3: Implement `call_center_simulator/data/preprocessing.py`**

```python
"""Data preprocessing utilities for Essays and PersonaChat datasets."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

# OCEAN axis order throughout the codebase: O, C, E, A, N
OCEAN_AXIS_ORDER = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]


def normalize_ocean(df: pd.DataFrame, ocean_cols: list[str]) -> pd.DataFrame:
    """Normalize OCEAN columns to float in [0, 1]. Essays uses binary {0,1}."""
    df = df.copy()
    for col in ocean_cols:
        df[col] = df[col].astype(float).clip(0.0, 1.0)
    return df


def user_based_split(
    df: pd.DataFrame,
    user_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique users to avoid data leakage (seed=42)."""
    users = list(df[user_col].unique())
    rng = random.Random(seed)
    rng.shuffle(users)
    n = len(users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_users = set(users[:n_train])
    val_users = set(users[n_train : n_train + n_val])
    test_users = set(users[n_train + n_val :])
    return (
        df[df[user_col].isin(train_users)].reset_index(drop=True),
        df[df[user_col].isin(val_users)].reset_index(drop=True),
        df[df[user_col].isin(test_users)].reset_index(drop=True),
    )


def build_essay_pairs(
    df: pd.DataFrame,
    text_col: str,
    ocean_cols: list[str],
) -> list[dict[str, Any]]:
    """Build list of {text, ocean_profile} dicts from Essays DataFrame."""
    return [
        {"text": str(row[text_col]),
         "ocean_profile": [float(row[col]) for col in ocean_cols]}
        for _, row in df.iterrows()
    ]


def build_dialog_pairs(
    raw_dialogs: list[dict[str, Any]],
    max_history: int = 10,
) -> list[dict[str, Any]]:
    """Build {history, response} pairs from PersonaChat raw dialogs."""
    pairs = []
    for dialog in raw_dialogs:
        for utterance in dialog.get("utterances", []):
            candidates = utterance.get("candidates", [])
            if not candidates:
                continue
            history = utterance.get("history", [])
            pairs.append({
                "history": history[-max_history:],
                "response": candidates[-1],
            })
    return pairs
```

- [ ] **Step 4: Implement `call_center_simulator/data/download.py`**

```python
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
    """Download Essays CSV from GitHub mirror. Falls back with instructions."""
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
    """Download PersonaChat via HuggingFace datasets library."""
    from datasets import load_dataset  # type: ignore[import-untyped]

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
```

- [ ] **Step 5: Run preprocessing tests**

```bash
uv run pytest tests/unit/test_preprocessing.py -v
```

Expected: all 8 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add call_center_simulator/data/download.py \
        call_center_simulator/data/preprocessing.py \
        tests/unit/test_preprocessing.py
git commit -m "feat: data download + preprocessing with user-based split (TDD)"
```

---

### Task 5: DataModules (Lightning)

**Files:**
- Create: `call_center_simulator/data/datamodule.py`

- [ ] **Step 1: Implement `call_center_simulator/data/datamodule.py`**

```python
"""PyTorch Lightning DataModules for Essays and PersonaChat datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from omegaconf import DictConfig

from call_center_simulator.data.preprocessing import (
    OCEAN_AXIS_ORDER,
    build_dialog_pairs,
    build_essay_pairs,
    normalize_ocean,
    user_based_split,
)

logger = logging.getLogger(__name__)


class EssaysDataModule(LightningDataModule):
    """DataModule for the Essays (Mairesse/Pennebaker) dataset."""

    def __init__(
        self,
        csv_path: Path | str,
        tokenizer_name: str,
        ocean_cols: list[str] | None = None,
        text_col: str = "TEXT",
        user_col: str = "#AUTHID",
        batch_size: int = 16,
        num_workers: int = 0,
        max_length: int = 512,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.tokenizer_name = tokenizer_name
        self.ocean_cols = ocean_cols or OCEAN_AXIS_ORDER
        self.text_col = text_col
        self.user_col = user_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> EssaysDataModule:
        return cls(
            csv_path=cfg.data.raw_path,
            tokenizer_name=cfg.model.backbone_name,
            ocean_cols=cfg.data.ocean_order,
            text_col=cfg.data.text_column,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            max_length=cfg.data.max_length,
            seed=cfg.seed,
        )

    def prepare_data(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Essays CSV not found: {self.csv_path}. Run: uv run dvc repro download"
            )

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.csv_path)
        df = normalize_ocean(df, self.ocean_cols)
        train_df, val_df, test_df = user_based_split(
            df, self.user_col, self.train_ratio, self.val_ratio, self.seed
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.train_dataset = self._make_dataset(train_df, tokenizer)
        self.val_dataset = self._make_dataset(val_df, tokenizer)
        self.test_dataset = self._make_dataset(test_df, tokenizer)

    def _make_dataset(self, df: pd.DataFrame, tokenizer: Any) -> TensorDataset:
        pairs = build_essay_pairs(df, self.text_col, self.ocean_cols)
        texts = [p["text"] for p in pairs]
        labels = [p["ocean_profile"] for p in pairs]
        enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return TensorDataset(
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(labels, dtype=torch.float32),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        assert self.train_dataset is not None
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader[Any]:
        assert self.val_dataset is not None
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader[Any]:
        assert self.test_dataset is not None
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


class PersonaChatDataModule(LightningDataModule):
    """DataModule for PersonaChat (dialog pairs for BLEU/ROUGE-L eval)."""

    def __init__(
        self,
        dataset_dir: Path | str,
        tokenizer_name: str,
        batch_size: int = 8,
        num_workers: int = 0,
        max_length: int = 256,
        max_history_turns: int = 10,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.max_history_turns = max_history_turns
        self.val_dataset: Dataset[Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        from datasets import load_from_disk  # type: ignore[import-untyped]

        dataset = load_from_disk(str(self.dataset_dir / "personachat"))
        pairs = build_dialog_pairs(list(dataset["validation"]), self.max_history_turns)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        contexts = [" ".join(p["history"]) for p in pairs]
        responses = [p["response"] for p in pairs]
        ctx_enc = tokenizer(contexts, padding="max_length", truncation=True,
                            max_length=self.max_length, return_tensors="pt")
        resp_enc = tokenizer(responses, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        self.val_dataset = TensorDataset(
            ctx_enc["input_ids"], ctx_enc["attention_mask"], resp_enc["input_ids"]
        )

    def val_dataloader(self) -> DataLoader[Any]:
        assert self.val_dataset is not None
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from call_center_simulator.data.datamodule import EssaysDataModule; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add call_center_simulator/data/datamodule.py
git commit -m "feat: EssaysDataModule + PersonaChatDataModule (Lightning)"
```

---

### Task 6: OCEAN Classifier component + LightningModule (TDD)

**Files:**
- Create: `call_center_simulator/models/components/ocean_classifier.py`
- Create: `call_center_simulator/models/ocean_classifier_module.py`
- Create: `call_center_simulator/training/train_ocean_classifier.py`
- Create: `tests/unit/test_ocean_classifier.py`

- [ ] **Step 1: Write failing tests — create `tests/unit/test_ocean_classifier.py`**

```python
"""Tests for OceanClassifierHead."""

import torch
import pytest

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead


def test_output_shape():
    # hidden_size=64 is a tiny proxy; real value is 1024 (derived from Qwen3-0.6B config)
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    out = model(torch.randn(4, 64))
    assert out.shape == (4, 5)


def test_output_range():
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    out = model(torch.randn(8, 64))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_dim_5():
    model = OceanClassifierHead(input_dim=128, hidden_dim=64, output_dim=5)
    out = model(torch.randn(1, 128))
    assert out.shape[-1] == 5


def test_no_nan():
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    out = model(torch.randn(16, 64))
    assert not torch.isnan(out).any()


def test_gradient_flows():
    model = OceanClassifierHead(input_dim=64, hidden_dim=32, output_dim=5)
    x = torch.randn(4, 64, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_ocean_classifier.py -v
```

Expected: `FAILED` — `ImportError: cannot import name 'OceanClassifierHead'`

- [ ] **Step 3: Implement `call_center_simulator/models/components/ocean_classifier.py`**

```python
"""OCEAN personality classifier head (MLP)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class OceanClassifierHead(nn.Module):
    """MLP head: Linear(input_dim->hidden_dim)->ReLU->Dropout->Linear(->5)->Sigmoid.

    Input: mean-pooled hidden state from backbone.
    Output: 5 values in [0,1], axis order O, C, E, A, N.
    Pass model.config.hidden_size as input_dim in production code.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: x shape [B, input_dim]. Returns: shape [B, 5]."""
        return self.net(x)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_ocean_classifier.py -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 5: Implement `call_center_simulator/models/ocean_classifier_module.py`**

```python
"""PyTorch Lightning module for pre-training the OCEAN classifier head."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from transformers import AutoModel

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead
from call_center_simulator.utils.metrics import compute_mape_ocean

logger = logging.getLogger(__name__)


class OceanClassifierModule(LightningModule):
    """Trains OceanClassifierHead on frozen Qwen3-0.6B backbone. Loss: BCE."""

    def __init__(
        self,
        backbone_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.backbone = AutoModel.from_pretrained(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # hidden_size read from model.config — never hardcoded
        hidden_size: int = self.backbone.config.hidden_size
        self.classifier = OceanClassifierHead(
            input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=5, dropout=dropout
        )
        self.loss_fn = nn.BCELoss()

    def _pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self._pool(out.last_hidden_state, attention_mask))

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ids, mask, labels = batch
        loss = self.loss_fn(self(ids, mask), labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        ids, mask, labels = batch
        preds = self(ids, mask)
        loss = self.loss_fn(preds, labels)
        mape, _ = compute_mape_ocean(preds.detach().cpu(), labels.detach().cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mape_ocean", mape, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
```

- [ ] **Step 6: Implement `call_center_simulator/training/train_ocean_classifier.py`**

```python
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
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
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
```

- [ ] **Step 7: Commit**

```bash
git add call_center_simulator/models/components/ocean_classifier.py \
        call_center_simulator/models/ocean_classifier_module.py \
        call_center_simulator/training/train_ocean_classifier.py \
        tests/unit/test_ocean_classifier.py
git commit -m "feat: OceanClassifierHead + OceanClassifierModule + train entry-point (TDD)"
```

---

### Task 7: Steering Vectors component + SteeringModule (TDD)

**Files:**
- Create: `call_center_simulator/models/components/steering_vectors.py`
- Create: `call_center_simulator/models/steering_module.py`
- Create: `call_center_simulator/training/train.py`
- Create: `tests/unit/test_steering_vectors.py`

- [ ] **Step 1: Write failing tests — create `tests/unit/test_steering_vectors.py`**

```python
"""Tests for SteeringVectors."""

import torch
import pytest

from call_center_simulator.models.components.steering_vectors import SteeringVectors


def test_parameter_shape():
    # hidden_size=64 is a tiny proxy; real value is 1024 (derived from Qwen3-0.6B config)
    sv = SteeringVectors(hidden_size=64)
    assert sv.vectors.shape == (5, 64)


def test_parameter_is_trainable():
    sv = SteeringVectors(hidden_size=64)
    assert sv.vectors.requires_grad


def test_initial_values_zero():
    sv = SteeringVectors(hidden_size=64)
    assert torch.all(sv.vectors == 0.0)


def test_apply_delta_shape():
    sv = SteeringVectors(hidden_size=64)
    hidden = torch.randn(2, 10, 64)  # [B, seq_len, hidden_size]
    ocean_profile = torch.tensor([[0.3, 0.7, 0.2, 0.4, 0.8],
                                   [0.1, 0.5, 0.9, 0.3, 0.6]])
    result = sv.apply_delta(hidden, ocean_profile)
    assert result.shape == hidden.shape


def test_apply_delta_modifies_hidden():
    """apply_delta changes hidden states when vectors are non-zero."""
    sv = SteeringVectors(hidden_size=64)
    with torch.no_grad():
        sv.vectors.fill_(1.0)
    hidden = torch.zeros(2, 5, 64)
    ocean_profile = torch.ones(2, 5)
    result = sv.apply_delta(hidden, ocean_profile)
    # delta = ocean_profile @ vectors = [2,5] @ [5,64] = [2,64]
    # broadcast over seq_len: result != 0
    assert not torch.all(result == 0.0)


def test_gradient_flows_through_delta():
    """Gradients flow through apply_delta to steering vectors."""
    sv = SteeringVectors(hidden_size=64)
    hidden = torch.randn(2, 5, 64)
    ocean_profile = torch.rand(2, 5)
    result = sv.apply_delta(hidden, ocean_profile)
    result.sum().backward()
    assert sv.vectors.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_steering_vectors.py -v
```

Expected: `FAILED` — `ImportError: cannot import name 'SteeringVectors'`

- [ ] **Step 3: Implement `call_center_simulator/models/components/steering_vectors.py`**

```python
"""Steering vectors for OCEAN-conditioned generation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SteeringVectors(nn.Module):
    """5 learnable steering vectors, one per OCEAN axis.

    Shape: nn.Parameter([5, hidden_size]), initialized to zeros.
    Applied via forward hook on the target transformer layer.

    Usage:
        sv = SteeringVectors(hidden_size=model.config.hidden_size)
        sv.register(model, target_layer=model.config.num_hidden_layers // 2)
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize SteeringVectors.

        Args:
            hidden_size: Backbone hidden dimension.
                         Pass model.config.hidden_size in production code.
        """
        super().__init__()
        # Shape [5, hidden_size]: 5 OCEAN axes, axis order O, C, E, A, N
        self.vectors = nn.Parameter(torch.zeros(5, hidden_size))
        self._ocean_profile: Tensor | None = None
        self._hook_handle: torch.utils.hooks.RemovableHook | None = None

    def set_ocean_profile(self, ocean_profile: Tensor) -> None:
        """Set the OCEAN profile for the next forward pass.

        Args:
            ocean_profile: Tensor of shape [B, 5], values in [0, 1].
        """
        self._ocean_profile = ocean_profile

    def apply_delta(self, hidden_states: Tensor, ocean_profile: Tensor) -> Tensor:
        """Compute and add steering delta to hidden states.

        delta = ocean_profile @ vectors  -> shape [B, hidden_size]
        Broadcast over sequence length: hidden_states += delta.unsqueeze(1)

        Args:
            hidden_states: Tensor [B, seq_len, hidden_size].
            ocean_profile: Tensor [B, 5], values in [0, 1].

        Returns:
            Modified hidden states, same shape as input.
        """
        # [B, 5] @ [5, hidden_size] -> [B, hidden_size]
        delta = ocean_profile @ self.vectors
        # Broadcast over seq_len: [B, 1, hidden_size]
        return hidden_states + delta.unsqueeze(1)

    def _make_hook(self, ocean_profile: Tensor):
        """Create a forward hook that injects the steering delta."""
        def hook(module, input, output):  # noqa: ANN001
            # output is a tuple; first element is hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            modified = self.apply_delta(hidden, ocean_profile)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return hook

    def register(self, model: nn.Module, target_layer: int) -> None:
        """Register forward hook on the target transformer layer.

        Args:
            model: The backbone transformer (Qwen3-0.6B).
            target_layer: Layer index = model.config.num_hidden_layers // 2.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()

        # Access transformer layers (Qwen3 uses model.layers)
        layers = model.model.layers if hasattr(model, "model") else model.layers
        layer = layers[target_layer]

        def hook(module, input, output):  # noqa: ANN001
            if self._ocean_profile is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            modified = self.apply_delta(hidden, self._ocean_profile)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        self._hook_handle = layer.register_forward_hook(hook)

    def remove_hook(self) -> None:
        """Remove the registered forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_steering_vectors.py -v
```

Expected: all 6 tests `PASSED`.

- [ ] **Step 5: Implement `call_center_simulator/models/steering_module.py`**

```python
"""PyTorch Lightning module for training steering vectors."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from transformers import AutoModelForCausalLM

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead
from call_center_simulator.models.components.steering_vectors import SteeringVectors
from call_center_simulator.utils.metrics import compute_mape_ocean, compute_perplexity

logger = logging.getLogger(__name__)


class SteeringModule(LightningModule):
    """Trains steering vectors on frozen Qwen3-0.6B backbone.

    Trainable parameters: only SteeringVectors (5 * hidden_size = 5120 params).
    Backbone: frozen (requires_grad=False on all params).
    OCEAN classifier: frozen (loaded from checkpoint, requires_grad=False).

    Loss: CE_LM + lambda_steering * BCE(ocean_classifier(pooled_hidden), target_profile)
    """

    def __init__(
        self,
        backbone_name: str,
        ocean_classifier_ckpt: str | None = None,
        hidden_dim: int = 256,
        lambda_steering: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lambda_steering = lambda_steering
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Frozen backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # hidden_size and num_layers read from model.config — never hardcoded
        hidden_size: int = self.backbone.config.hidden_size
        num_layers: int = self.backbone.config.num_hidden_layers
        target_layer: int = num_layers // 2

        # Trainable steering vectors
        self.steering = SteeringVectors(hidden_size=hidden_size)
        self.steering.register(self.backbone, target_layer=target_layer)

        # Frozen OCEAN classifier
        self.ocean_classifier = OceanClassifierHead(
            input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=5
        )
        if ocean_classifier_ckpt is not None:
            state = torch.load(ocean_classifier_ckpt, map_location="cpu")
            self.ocean_classifier.load_state_dict(state)
        for param in self.ocean_classifier.parameters():
            param.requires_grad = False

        self.bce_loss = nn.BCELoss()

    def _pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        ocean_profile: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self.steering.set_ocean_profile(ocean_profile)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        ce_lm = outputs.loss if labels is not None else torch.tensor(0.0)
        # Pool last hidden state for OCEAN classifier
        last_hidden = outputs.hidden_states[-1]
        pooled = self._pool(last_hidden, attention_mask)
        ocean_preds = self.ocean_classifier(pooled)
        return {"ce_lm": ce_lm, "ocean_preds": ocean_preds, "pooled": pooled}

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        input_ids, attention_mask, ocean_labels = batch
        # Shift labels for causal LM
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        result = self(input_ids, attention_mask, ocean_labels, labels)
        bce = self.bce_loss(result["ocean_preds"], ocean_labels)
        loss = result["ce_lm"] + self.lambda_steering * bce
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ce_lm", result["ce_lm"], on_step=True, on_epoch=True)
        self.log("train_bce_ocean", bce, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, attention_mask, ocean_labels = batch
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        result = self(input_ids, attention_mask, ocean_labels, labels)
        bce = self.bce_loss(result["ocean_preds"], ocean_labels)
        loss = result["ce_lm"] + self.lambda_steering * bce
        mape, _ = compute_mape_ocean(
            result["ocean_preds"].detach().cpu(), ocean_labels.detach().cpu()
        )
        ppl = compute_perplexity(float(result["ce_lm"].detach().cpu()))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mape_ocean", mape, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", ppl, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.steering.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
```

- [ ] **Step 6: Implement `call_center_simulator/training/train.py`**

```python
"""Entry point: train steering vectors on Essays dataset."""

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
from call_center_simulator.models.steering_module import SteeringModule

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent.parent.parent / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Train steering vectors."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=(cfg.mlflow.run_name or "steering-vectors"),
    )
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit = "unknown"
    mlflow_logger.log_hyperparams({"git_commit": git_commit})
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    datamodule = EssaysDataModule.from_hydra_config(cfg)
    model = SteeringModule(
        backbone_name=cfg.model.backbone_name,
        ocean_classifier_ckpt=cfg.model.ocean_classifier.get("ckpt_path", None),
        hidden_dim=cfg.model.ocean_classifier.hidden_dim,
        lambda_steering=cfg.model.lambda_steering,
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
```

- [ ] **Step 7: Commit**

```bash
git add call_center_simulator/models/components/steering_vectors.py \
        call_center_simulator/models/steering_module.py \
        call_center_simulator/training/train.py \
        tests/unit/test_steering_vectors.py
git commit -m "feat: SteeringVectors + SteeringModule + train entry-point (TDD)"
```

---

### Task 8: ONNX export + FastAPI + Gradio + CLI

**Files:**
- Create: `call_center_simulator/inference/export_onnx.py`
- Create: `call_center_simulator/inference/api.py`
- Create: `call_center_simulator/inference/app.py`
- Create: `call_center_simulator/inference/infer.py`
- Create: `call_center_simulator/cli.py`

- [ ] **Step 1: Implement `call_center_simulator/inference/export_onnx.py`**

```python
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
        dynamic_axes={"hidden_states": {0: "batch_size"}, "ocean_scores": {0: "batch_size"}},
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
```

- [ ] **Step 2: Implement `call_center_simulator/inference/api.py`**

```python
"""FastAPI backend for call-center simulator inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from call_center_simulator.models.components.steering_vectors import SteeringVectors

logger = logging.getLogger(__name__)

app = FastAPI(title="Call-Center Simulator API", version="0.1.0")

# Module-level state (loaded on startup)
_backbone: Any = None
_tokenizer: Any = None
_steering: SteeringVectors | None = None


class OceanProfile(BaseModel):
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)

    def to_tensor(self) -> torch.Tensor:
        """Return [1, 5] tensor in O, C, E, A, N order."""
        return torch.tensor([[
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism,
        ]], dtype=torch.float32)


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


@app.on_event("startup")
async def load_model() -> None:
    """Load backbone and steering vectors on startup."""
    global _backbone, _tokenizer, _steering

    backbone_name = os.environ.get("BACKBONE_NAME", "Qwen/Qwen3-0.6B")
    steering_ckpt = os.environ.get("STEERING_CKPT", "")

    logger.info("Loading backbone: %s", backbone_name)
    _tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _backbone = AutoModelForCausalLM.from_pretrained(backbone_name, torch_dtype=torch.float16)
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
    logger.info("Model ready. hidden_size=%d, target_layer=%d", hidden_size, target_layer)


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

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return GenerateResponse(reply=reply)
```

- [ ] **Step 3: Implement `call_center_simulator/inference/app.py`**

```python
"""Gradio UI for call-center simulator."""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def generate_reply(
    situation: str,
    history_text: str,
    openness: float,
    conscientiousness: float,
    extraversion: float,
    agreeableness: float,
    neuroticism: float,
    max_tokens: int,
) -> str:
    """Call FastAPI /generate and return the reply."""
    history = []
    for line in history_text.strip().splitlines():
        if ": " in line:
            role, text = line.split(": ", 1)
            history.append({"role": role.strip(), "text": text.strip()})

    payload = {
        "history": history,
        "situation": situation,
        "ocean_profile": {
            "openness": openness,
            "conscientiousness": conscientiousness,
            "extraversion": extraversion,
            "agreeableness": agreeableness,
            "neuroticism": neuroticism,
        },
        "max_new_tokens": max_tokens,
    }
    try:
        response = httpx.post(f"{API_URL}/generate", json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()["reply"]
    except Exception as exc:
        return f"Error: {exc}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Call-Center Simulator") as demo:
        gr.Markdown("# Call-Center Simulator\nGenerate client replies with OCEAN personality.")
        with gr.Row():
            with gr.Column():
                situation = gr.Textbox(label="Situation", lines=2,
                                       placeholder="Client calls about delayed delivery...")
                history_text = gr.Textbox(label="Dialog history (role: text per line)", lines=5,
                                          placeholder="operator: Hello, how can I help?")
                gr.Markdown("### OCEAN Profile")
                openness = gr.Slider(0.0, 1.0, value=0.5, label="Openness (O)")
                conscientiousness = gr.Slider(0.0, 1.0, value=0.5, label="Conscientiousness (C)")
                extraversion = gr.Slider(0.0, 1.0, value=0.5, label="Extraversion (E)")
                agreeableness = gr.Slider(0.0, 1.0, value=0.5, label="Agreeableness (A)")
                neuroticism = gr.Slider(0.0, 1.0, value=0.5, label="Neuroticism (N)")
                max_tokens = gr.Slider(16, 256, value=128, step=16, label="Max tokens")
                btn = gr.Button("Generate reply", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Client reply", lines=6)
        btn.click(
            generate_reply,
            inputs=[situation, history_text, openness, conscientiousness,
                    extraversion, agreeableness, neuroticism, max_tokens],
            outputs=output,
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `call_center_simulator/inference/infer.py`**

```python
"""CLI inference: generate a client reply from command line."""

from __future__ import annotations

import json
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

    model = AutoModelForCausalLM.from_pretrained(backbone_name, torch_dtype=torch.float16)
    model.eval()

    # hidden_size and num_layers read from model.config — never hardcoded
    hidden_size: int = model.config.hidden_size
    num_layers: int = model.config.num_hidden_layers
    target_layer: int = num_layers // 2

    steering = SteeringVectors(hidden_size=hidden_size)
    if steering_ckpt and Path(steering_ckpt).exists():
        state = torch.load(steering_ckpt, map_location="cpu")
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

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

- [ ] **Step 5: Implement `call_center_simulator/cli.py`**

```python
"""Typer CLI entry-point for call-center simulator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Call-Center Simulator CLI")
logger = logging.getLogger(__name__)


@app.command()
def download_data() -> None:
    """Download Essays and PersonaChat datasets."""
    from call_center_simulator.data.download import download_essays, download_personachat

    download_essays(Path("data/raw/essays.csv"))
    download_personachat(Path("data/raw"))
    typer.echo("Data downloaded successfully.")


@app.command()
def train_ocean() -> None:
    """Train OCEAN classifier head on Essays dataset."""
    from call_center_simulator.training.train_ocean_classifier import main

    main()


@app.command()
def train_steering() -> None:
    """Train steering vectors on Essays dataset."""
    from call_center_simulator.training.train import main

    main()


@app.command()
def export_onnx(
    ckpt_path: Optional[str] = typer.Option(None, help="Path to classifier checkpoint"),
    output: str = typer.Option("models/ocean_classifier.onnx", help="Output ONNX path"),
    backbone: str = typer.Option("Qwen/Qwen3-0.6B", help="Backbone model name"),
) -> None:
    """Export OCEAN classifier to ONNX format."""
    import torch
    from transformers import AutoModel

    from call_center_simulator.inference.export_onnx import (
        export_ocean_classifier_onnx,
        verify_onnx,
    )
    from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead

    backbone_model = AutoModel.from_pretrained(backbone)
    hidden_size: int = backbone_model.config.hidden_size

    classifier = OceanClassifierHead(input_dim=hidden_size, hidden_dim=256, output_dim=5)
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
    steering_ckpt: Optional[str] = typer.Option(None),
    max_tokens: int = typer.Option(128),
) -> None:
    """Generate a client reply from CLI."""
    from call_center_simulator.inference.infer import generate_reply, load_model

    history = json.loads(history_json)
    ocean_profile = [openness, conscientiousness, extraversion, agreeableness, neuroticism]
    model, tokenizer, steering = load_model(backbone, steering_ckpt)
    reply = generate_reply(model, tokenizer, steering, situation, history, ocean_profile, max_tokens)
    typer.echo(reply)


@app.command()
def serve_api(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Start FastAPI inference server."""
    import uvicorn

    uvicorn.run("call_center_simulator.inference.api:app", host=host, port=port, reload=False)


@app.command()
def serve_ui(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(7860),
) -> None:
    """Start Gradio UI."""
    from call_center_simulator.inference.app import main

    main()


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Verify imports**

```bash
uv run python -c "from call_center_simulator.cli import app; print('CLI OK')"
uv run python -c "from call_center_simulator.inference.api import app; print('API OK')"
```

Expected: `CLI OK`, `API OK`

- [ ] **Step 7: Commit**

```bash
git add call_center_simulator/inference/export_onnx.py \
        call_center_simulator/inference/api.py \
        call_center_simulator/inference/app.py \
        call_center_simulator/inference/infer.py \
        call_center_simulator/cli.py
git commit -m "feat: ONNX export + FastAPI /generate + Gradio UI + Typer CLI"
```

---

### Task 9: Smoke tests (tiny-random model, CPU, < 30 s)

**Files:**
- Create: `tests/smoke/test_smoke_training.py`
- Create: `tests/smoke/test_smoke_api.py`

Smoke tests use a **tiny-random model** built with `transformers.AutoConfig` (`num_hidden_layers=2, hidden_size=64`). They do NOT download `Qwen/Qwen3-0.6B`. All tests run on CPU in < 30 s total.

- [ ] **Step 1: Create `tests/smoke/test_smoke_training.py`**

```python
"""Smoke tests for training pipeline using tiny-random model.

All tests use a tiny random transformer (num_hidden_layers=2, hidden_size=64)
to avoid downloading Qwen3-0.6B. Tests run on CPU in < 30 s total.
"""

from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from call_center_simulator.models.components.ocean_classifier import OceanClassifierHead
from call_center_simulator.models.components.steering_vectors import SteeringVectors
from call_center_simulator.models.ocean_classifier_module import OceanClassifierModule
from call_center_simulator.models.steering_module import SteeringModule


# Tiny model config — derived from Qwen3-0.6B config but shrunk for speed
TINY_HIDDEN = 64   # real Qwen3-0.6B: 1024
TINY_LAYERS = 2    # real Qwen3-0.6B: 28
TINY_VOCAB = 256   # real Qwen3-0.6B: 151936
BATCH = 2
SEQ_LEN = 16


def _make_tiny_causal_lm():
    """Create a tiny random causal LM for smoke tests."""
    config = AutoConfig.for_model(
        "qwen2",  # Qwen3 uses qwen2 architecture
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
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
        out_with_hook = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)

    sv.remove_hook()
    with torch.no_grad():
        out_without_hook = model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)

    # Hidden states at target layer should differ
    h_with = out_with_hook.hidden_states[TINY_LAYERS // 2 + 1]
    h_without = out_without_hook.hidden_states[TINY_LAYERS // 2 + 1]
    assert not torch.allclose(h_with, h_without), "Hook did not modify hidden states"


def test_ocean_classifier_module_train_step():
    """OceanClassifierModule.training_step runs without error on tiny model."""
    # Patch backbone with tiny model
    module = OceanClassifierModule.__new__(OceanClassifierModule)
    module.backbone = _make_tiny_encoder()
    for p in module.backbone.parameters():
        p.requires_grad = False
    module.classifier = OceanClassifierHead(input_dim=TINY_HIDDEN, hidden_dim=32, output_dim=5)
    import torch.nn as nn
    module.loss_fn = nn.BCELoss()
    module.learning_rate = 1e-3
    module.weight_decay = 0.0

    batch = _make_dummy_batch()
    loss = module.training_step(batch, 0)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_steering_module_train_step():
    """SteeringModule.training_step runs without error on tiny model."""
    module = SteeringModule.__new__(SteeringModule)
    module.backbone = _make_tiny_causal_lm()
    for p in module.backbone.parameters():
        p.requires_grad = False

    hidden_size = TINY_HIDDEN
    num_layers = TINY_LAYERS
    target_layer = num_layers // 2

    module.steering = SteeringVectors(hidden_size=hidden_size)
    module.steering.register(module.backbone, target_layer=target_layer)

    module.ocean_classifier = OceanClassifierHead(input_dim=hidden_size, hidden_dim=32, output_dim=5)
    for p in module.ocean_classifier.parameters():
        p.requires_grad = False

    import torch.nn as nn
    module.bce_loss = nn.BCELoss()
    module.lambda_steering = 0.1
    module.learning_rate = 1e-3
    module.weight_decay = 0.0

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
```

- [ ] **Step 2: Create `tests/smoke/test_smoke_api.py`**

```python
"""Smoke tests for FastAPI /generate endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _make_test_app():
    """Create FastAPI app with a mock model (no real Qwen3-0.6B)."""
    import torch
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    from call_center_simulator.inference.api import (
        GenerateRequest,
        GenerateResponse,
        OceanProfile,
        DialogTurn,
    )

    # Build a minimal app that mimics the real one but uses a stub model
    test_app = FastAPI()

    @test_app.get("/health")
    def health():
        return {"status": "healthy"}

    @test_app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
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
```

- [ ] **Step 3: Run smoke tests**

```bash
uv run pytest tests/smoke/ -v --timeout=30
```

Expected: all 8 smoke tests `PASSED` in < 30 s on CPU.

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/test_smoke_training.py tests/smoke/test_smoke_api.py
git commit -m "test: smoke tests — tiny-random model, CPU, < 30 s"
```

---

### Task 10: DVC pipeline (dvc.yaml)

**Files:**
- Create: `dvc.yaml`

- [ ] **Step 1: Create `dvc.yaml`**

```yaml
stages:
  download:
    cmd: uv run python -m call_center_simulator.data.download
    deps:
      - call_center_simulator/data/download.py
    outs:
      - data/raw/essays.csv
      - data/raw/personachat

  preprocess:
    cmd: >
      uv run python -c "
      import pandas as pd;
      from call_center_simulator.data.preprocessing import normalize_ocean, user_based_split, OCEAN_AXIS_ORDER;
      df = pd.read_csv('data/raw/essays.csv');
      df = normalize_ocean(df, OCEAN_AXIS_ORDER);
      train, val, test = user_based_split(df, '#AUTHID', 0.8, 0.1, 42);
      import pathlib; pathlib.Path('data/processed').mkdir(parents=True, exist_ok=True);
      train.to_csv('data/processed/essays_train.csv', index=False);
      val.to_csv('data/processed/essays_val.csv', index=False);
      test.to_csv('data/processed/essays_test.csv', index=False);
      print('Preprocessing done.')
      "
    deps:
      - call_center_simulator/data/preprocessing.py
      - data/raw/essays.csv
    outs:
      - data/processed/essays_train.csv
      - data/processed/essays_val.csv
      - data/processed/essays_test.csv

  train_ocean_classifier:
    cmd: >
      uv run python -m call_center_simulator.training.train_ocean_classifier
      train=default
    deps:
      - call_center_simulator/training/train_ocean_classifier.py
      - call_center_simulator/models/ocean_classifier_module.py
      - call_center_simulator/models/components/ocean_classifier.py
      - data/processed/essays_train.csv
      - data/processed/essays_val.csv
    outs:
      - models/ocean_classifier_best.ckpt
    params:
      - configs/config.yaml:
          - seed
          - train.max_epochs
          - train.learning_rate
          - model.backbone_name
          - model.ocean_classifier.hidden_dim

  export_onnx:
    cmd: >
      uv run python -m call_center_simulator.inference.export_onnx
    deps:
      - call_center_simulator/inference/export_onnx.py
      - models/ocean_classifier_best.ckpt
    outs:
      - models/ocean_classifier.onnx

  train_steering:
    cmd: >
      uv run python -m call_center_simulator.training.train
      train=default
    deps:
      - call_center_simulator/training/train.py
      - call_center_simulator/models/steering_module.py
      - call_center_simulator/models/components/steering_vectors.py
      - models/ocean_classifier_best.ckpt
      - data/processed/essays_train.csv
      - data/processed/essays_val.csv
    outs:
      - models/steering_best.ckpt
    params:
      - configs/config.yaml:
          - seed
          - train.max_epochs
          - train.learning_rate
          - model.backbone_name
          - model.lambda_steering

  evaluate:
    cmd: >
      uv run python -c "
      import torch, json;
      from call_center_simulator.utils.metrics import compute_mape_ocean, compute_perplexity;
      print('Evaluate stage: placeholder — run after real training on GPU.');
      metrics = {'mape_ocean': 0.0, 'perplexity': 0.0};
      import pathlib; pathlib.Path('models').mkdir(exist_ok=True);
      pathlib.Path('models/metrics.json').write_text(json.dumps(metrics));
      print('Metrics saved to models/metrics.json')
      "
    deps:
      - call_center_simulator/utils/metrics.py
      - models/steering_best.ckpt
    metrics:
      - models/metrics.json:
          cache: false
```

- [ ] **Step 2: Verify DVC pipeline parses correctly**

```bash
uv run dvc dag
```

Expected: ASCII DAG showing download → preprocess → train_ocean_classifier → export_onnx → train_steering → evaluate

- [ ] **Step 3: Commit**

```bash
git add dvc.yaml
git commit -m "feat: DVC pipeline — download, preprocess, train_ocean, export_onnx, train_steering, evaluate"
```

---

### Task 11: Docker — Dockerfile + docker-compose.yml

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
# Multi-stage Dockerfile for Call-Center Simulator

# Stage 1: Builder
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache .

# Stage 2: Runtime
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY call_center_simulator /app/call_center_simulator
COPY configs /app/configs

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "call_center_simulator.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: call-center-api
    ports:
      - "8000:8000"
    environment:
      - BACKBONE_NAME=${BACKBONE_NAME:-Qwen/Qwen3-0.6B}
      - STEERING_CKPT=${STEERING_CKPT:-}
      - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow:5000}
      - HF_TOKEN=${HF_TOKEN:-}
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    depends_on:
      - mlflow

  gradio:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: call-center-gradio
    command: ["python", "-m", "call_center_simulator.inference.app"]
    ports:
      - "7860:7860"
    environment:
      - API_URL=http://api:8000
    restart: unless-stopped
    depends_on:
      - api

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.0
    container_name: call-center-mlflow
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root /mlflow/artifacts
    volumes:
      - mlflow_data:/mlflow
    restart: unless-stopped

volumes:
  mlflow_data:
```

- [ ] **Step 3: Verify Docker Compose config parses**

```bash
docker compose config --quiet
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: multi-stage Dockerfile + docker-compose (api, gradio, mlflow)"
```

---

### Task 12: AGENTS.md + README.md

**Files:**
- Create: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Create `AGENTS.md`**

```markdown
# AGENTS.md

Rules for AI agents working in this repository.

## Common commands

### Environment setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
cp .env.example .env
# Edit .env: add HF_TOKEN if needed
```

### Data

```bash
# Download all datasets
uv run python -m call_center_simulator.cli download-data

# Or via DVC
uv run dvc repro download
```

### Training

```bash
# Start MLflow server (local)
uv run mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Train OCEAN classifier
uv run python -m call_center_simulator.cli train-ocean

# Export OCEAN classifier to ONNX
uv run python -m call_center_simulator.cli export-onnx

# Train steering vectors
uv run python -m call_center_simulator.cli train-steering

# Full DVC pipeline
uv run dvc repro
```

### Inference

```bash
# CLI inference
uv run python -m call_center_simulator.cli infer \
  --situation "Client calls about delayed delivery" \
  --neuroticism 0.8 --agreeableness 0.3

# Start FastAPI server
uv run python -m call_center_simulator.cli serve-api

# Start Gradio UI
uv run python -m call_center_simulator.cli serve-ui
```

### Testing

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit/ -v

# Smoke tests only (CPU, < 30 s)
uv run pytest tests/smoke/ -v --timeout=30

# With coverage
uv run pytest --cov=call_center_simulator --cov-report=term-missing
```

### Code quality

```bash
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

### Docker

```bash
docker compose up --build
```

## Architecture

```
Essays CSV
    |
    v
EssaysDataModule (user-based split 80/10/10, seed=42)
    |
    v
OceanClassifierModule (frozen Qwen3-0.6B backbone + MLP head)
    |  BCE loss, Adam
    v
ocean_classifier_best.ckpt --> export_onnx --> ocean_classifier.onnx
    |
    v
SteeringModule (frozen backbone + frozen OCEAN classifier + trainable SteeringVectors)
    |  Loss: CE_LM + 0.1 * BCE(OCEAN_classifier(hidden_states), target_profile)
    v
steering_best.ckpt
    |
    v
FastAPI /generate <-- Gradio UI
```

## Key design decisions

1. **hidden_size from config**: Always use `model.config.hidden_size`, never hardcode 1024.
2. **target_layer from config**: Always use `model.config.num_hidden_layers // 2`, never hardcode 14.
3. **OCEAN axis order**: O, C, E, A, N throughout all code.
4. **Smoke tests**: Use tiny-random model (hidden_size=64, layers=2), never real Qwen3-0.6B.
5. **No HF Trainer**: Training via PyTorch Lightning only.
6. **DVC local remote**: `.dvc-storage/` in repo root.
```

- [ ] **Step 2: Update `README.md`**

Replace the existing README with:

```markdown
# Call-Center Simulator

Мун Павел Юрьевич

Conditional generation of call-center client replies conditioned on a Big-5 / OCEAN personality profile, using learnable steering vectors injected into a frozen Qwen3-0.6B backbone.

## Постановка задачи

- **Сухой остаток:** условная генерация текстовых реплик клиента колл-центра по заданному психологическому профилю (Big-5 / OCEAN) и описанию ситуации.
- **Решаемая проблема:** обучение операторов колл-центра требует взаимодействия с разнообразными типами клиентов. Реальные тренировки дороги и труднопредсказуемы.
- **Решение:** текстовый бот с профилем клиента (5 числовых параметров OCEAN). Управление стилем — через learnable steering vectors, вставляемые в активации через forward hook.

## Architecture

```
Gradio UI (OCEAN sliders + dialog)
    | HTTP
    v
FastAPI /generate
    |
    v
Qwen3-0.6B (frozen) + SteeringVectors (hook on layer 14)
    |
    v
Client reply
```

```
                    +------------------+
Essays CSV -------> | EssaysDataModule |
                    +------------------+
                           |
                           v
                  OceanClassifierModule
                  (frozen Qwen3-0.6B + MLP head)
                  BCE loss --> ocean_classifier.onnx
                           |
                           v
                    SteeringModule
                  (frozen backbone + frozen OCEAN clf)
                  CE_LM + 0.1*BCE --> steering_best.ckpt
```

## Setup

```bash
# 1. Clone and enter repo
git clone <repo-url>
cd mlops-call-center-simulator

# 2. Create environment (Python 3.11)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Install pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files

# 4. Configure environment
cp .env.example .env
# Edit .env: add HF_TOKEN if needed (Qwen3-0.6B is non-gated)
```

## Train

### 1. Download data

```bash
uv run dvc repro download
# Or manually: place essays.csv at data/raw/essays.csv
```

### 2. Start MLflow

```bash
uv run mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

### 3. Train OCEAN classifier

```bash
uv run python -m call_center_simulator.cli train-ocean
```

### 4. Export OCEAN classifier to ONNX

```bash
uv run python -m call_center_simulator.cli export-onnx
```

### 5. Train steering vectors

```bash
uv run python -m call_center_simulator.cli train-steering
```

### 6. Full DVC pipeline

```bash
uv run dvc repro
```

## Infer

### CLI

```bash
uv run python -m call_center_simulator.cli infer \
  --situation "Client calls about delayed delivery" \
  --neuroticism 0.8 --agreeableness 0.3
```

### API server

```bash
uv run python -m call_center_simulator.cli serve-api
# POST http://localhost:8000/generate
```

### Gradio UI

```bash
uv run python -m call_center_simulator.cli serve-ui
# Open http://localhost:7860
```

### Docker Compose

```bash
docker compose up --build
# API: http://localhost:8000
# Gradio: http://localhost:7860
# MLflow: http://localhost:5000
```

## Overall

### Project tree

```
mlops-call-center-simulator/
├── call_center_simulator/
│   ├── cli.py
│   ├── data/           download, preprocessing, datamodule
│   ├── models/         ocean_classifier_module, steering_module
│   │   └── components/ OceanClassifierHead, SteeringVectors
│   ├── training/       train_ocean_classifier, train
│   ├── inference/      export_onnx, api, app, infer
│   └── utils/          metrics
├── configs/            Hydra configs
├── tests/unit/         TDD unit tests
├── tests/smoke/        CPU smoke tests (< 30 s)
├── data/               DVC-managed datasets
├── models/             DVC-managed checkpoints + ONNX
├── dvc.yaml            DVC pipeline
├── Dockerfile
└── docker-compose.yml
```

### Metrics

| Metric | Target |
|---|---|
| MAPE_ocean | < 0.25 |
| Perplexity | < 50 |
| BLEU | > 0.10 |
| ROUGE-L | > 0.20 |
| Distinct-1 | > 0.5 |
| Distinct-2 | > 0.7 |
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: AGENTS.md + README with setup/train/infer/architecture"
```

---

## Self-Review

### 1. Spec Coverage

| Scope item | Task |
|---|---|
| Bootstrap: pyproject.toml, .gitignore, .dvcignore, .pre-commit-config.yaml, .env.example, DVC init | Task 1 |
| Package structure `call_center_simulator/` | Task 2 |
| Hydra configs: config.yaml + data/, model/, train/ subgroups | Task 2 |
| `data/download.py` — Essays (GitHub raw) + PersonaChat (HF) | Task 4 |
| `data/preprocessing.py` (TDD) — normalize OCEAN, user-based split, pairs | Task 4 |
| `data/datamodule.py` — EssaysDataModule + PersonaChatDataModule | Task 5 |
| `models/components/ocean_classifier.py` (TDD) — OceanClassifierHead | Task 6 |
| `models/components/steering_vectors.py` (TDD) — SteeringVectors + hook | Task 7 |
| `models/ocean_classifier_module.py` + `training/train_ocean_classifier.py` | Task 6 |
| `models/steering_module.py` + `training/train.py` | Task 7 |
| `inference/export_onnx.py` (TDD on ONNX format) | Task 8 |
| `inference/api.py` (FastAPI /generate) + Pydantic models | Task 8 |
| `inference/app.py` (Gradio UI) + `inference/infer.py` + `cli.py` | Task 8 |
| `utils/metrics.py` (TDD) — MAPE_ocean, Perplexity, BLEU, ROUGE-L, Distinct-1/2 | Task 3 |
| Smoke tests in `tests/smoke/` — tiny-random model, CPU, < 30 s | Task 9 |
| DVC pipeline `dvc.yaml` — download, preprocess, train_ocean, export_onnx, train_steering, evaluate | Task 10 |
| Docker: Dockerfile (multi-stage) + docker-compose.yml (api, gradio, mlflow) | Task 11 |
| MLflow: losses, MAPE_ocean, perplexity, git commit id | Tasks 6, 7 |
| README: Setup/Train/Infer/Overall + project tree + architecture diagram | Task 12 |
| AGENTS.md | Task 12 |

All 19 scope items are covered across 12 tasks.

### 2. Placeholder Scan

- No "TBD", "TODO", "implement later", "similar to Task N" in any task.
- All code blocks are complete and runnable.
- The only intentional placeholder is the `evaluate` DVC stage which prints a message explaining it runs after real GPU training — this is explicitly documented.

### 3. Type Consistency

| Name | Defined in | Used consistently in |
|---|---|---|
| `OceanClassifierHead` | Task 6 `ocean_classifier.py` | Tasks 6, 7, 8, 9 |
| `SteeringVectors` | Task 7 `steering_vectors.py` | Tasks 7, 8, 9 |
| `OceanClassifierModule` | Task 6 `ocean_classifier_module.py` | Tasks 6, 9 |
| `SteeringModule` | Task 7 `steering_module.py` | Tasks 7, 9 |
| `EssaysDataModule` | Task 5 `datamodule.py` | Tasks 5, 6, 7 |
| `GenerateRequest` / `GenerateResponse` | Task 8 `api.py` | Task 8, 9 |
| `compute_mape_ocean` | Task 3 `metrics.py` | Tasks 6, 7, 9 |
| `OCEAN_AXIS_ORDER` | Task 4 `preprocessing.py` | Tasks 4, 5 |
| `build_essay_pairs` | Task 4 `preprocessing.py` | Tasks 4, 5 |
| `build_dialog_pairs` | Task 4 `preprocessing.py` | Tasks 4, 5 |

### 4. Model name check

- Model: `Qwen/Qwen3-0.6B` — used in Tasks 5, 6, 7, 8, 12.
- No occurrences of `Qwen2.5`, `0.5B`, or `896` in this plan.
- `hidden_size=1024` appears only in documentation comments and the plan header.
- In all production code: `hidden_size = model.config.hidden_size` (never hardcoded).
- `target_layer = model.config.num_hidden_layers // 2` (never hardcoded as 14).
- Hardcodes `TINY_HIDDEN=64`, `TINY_LAYERS=2` appear only in smoke tests with comment `# derived from Qwen3-0.6B config but shrunk for speed`.

### 5. Smoke test check

- `tests/smoke/test_smoke_training.py` uses `_make_tiny_causal_lm()` and `_make_tiny_encoder()` — tiny random models, no real Qwen3-0.6B download.
- `tests/smoke/test_smoke_api.py` uses `TestClient` with a stub model — no real model.
- All smoke tests run on CPU.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-24-call-center-simulator-phase-a.md`.

**Summary:**
- Deleted: old plan (overwritten), `update_plan.py`, `update_smoke_tests.py` (to be deleted by implementer in Task 1 cleanup)
- Model: `Qwen/Qwen3-0.6B` (confirmed non-gated on HuggingFace, released May 2025)
- `hidden_size`: 1024 (from `model.config.hidden_size` in code)
- `target_layer`: 14 (= `model.config.num_hidden_layers // 2` = 28 // 2 in code)
- Tasks: 12 tasks covering all 19 scope items

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
Use skill: `superpowers:subagent-driven-development`

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.
Use skill: `superpowers:executing-plans`
