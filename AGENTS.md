# AGENTS.md

This file provides guidance to AI agents (Yandex Code Assistant, Cursor, Copilot, etc.) when working with code in this repository.

## Common commands

### Environment setup

Create virtualenv (Python 3.11) and activate:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

Install project with dev dependencies:

```bash
uv pip install -e ".[dev]"
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Bootstrap local env vars:

```bash
cp .env.example .env
# Edit .env: add HF_TOKEN if needed (Qwen3-0.6B is non-gated, token optional)
```

### Data management

Download Essays + PersonaChat datasets (recommended via DVC):

```bash
uv run dvc repro download
```

Or directly via CLI:

```bash
uv run python -m call_center_simulator.cli download-data
```

### Training

Start local MLflow server for experiment tracking:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Train OCEAN classifier head on Essays dataset:

```bash
uv run python -m call_center_simulator.cli train-ocean
```

Export OCEAN classifier to ONNX:

```bash
uv run python -m call_center_simulator.cli export-onnx
```

Train steering vectors (frozen backbone + frozen OCEAN classifier):

```bash
uv run python -m call_center_simulator.cli train-steering
```

Run full DVC pipeline (download → preprocess → train_ocean → export_onnx → train_steering → evaluate):

```bash
uv run dvc repro
```

Override Hydra config at CLI (e.g., learning rate, epochs):

```bash
uv run python -m call_center_simulator.training.train \
  train.learning_rate=5e-4 \
  train.max_epochs=5
```

### Inference

CLI inference (no server required):

```bash
uv run python -m call_center_simulator.cli infer \
  --situation "Client calls about delayed delivery" \
  --neuroticism 0.8 --agreeableness 0.3
```

Start FastAPI server (requires model loaded via env vars):

```bash
uv run python -m call_center_simulator.cli serve-api
# API available at http://localhost:8000
```

Start Gradio UI:

```bash
uv run python -m call_center_simulator.cli serve-ui
# UI available at http://localhost:7860
```

### Testing and quality

Run full test suite (33 unit + 8 smoke, < 5 s on CPU):

```bash
uv run pytest
```

Run unit tests only:

```bash
uv run pytest tests/unit/ -v
```

Run smoke tests only (tiny-random model, CPU, < 30 s):

```bash
uv run pytest tests/smoke/ -v --timeout=30
```

Run with coverage:

```bash
uv run pytest --cov=call_center_simulator --cov-report=term-missing
```

Linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

Run all pre-commit hooks:

```bash
uv run pre-commit run --all-files
```

### Docker deployment

Build and run all services (api:8000, gradio:7860, mlflow:5000):

```bash
docker compose up --build
```

Build and run API only:

```bash
docker build -t call-center-simulator .
docker run -p 8000:8000 \
  -e BACKBONE_NAME=Qwen/Qwen3-0.6B \
  -v $(pwd)/models:/app/models:ro \
  call-center-simulator
```

## High-level architecture

### Pipeline overview

Two-stage ML pipeline for OCEAN-conditioned text generation:

1. **OceanClassifierHead** pre-trained on Essays dataset (BCE loss) → exported to ONNX
2. **SteeringVectors** trained on Essays with frozen backbone + frozen OCEAN classifier

Data flow:
```
Essays CSV
    → EssaysDataModule (user-based split 80/10/10, seed=42)
    → OceanClassifierModule (frozen Qwen3-0.6B + MLP head, BCE)
    → ocean_classifier.onnx
    → SteeringModule (frozen backbone + frozen OCEAN clf + trainable SteeringVectors)
    → steering_best.ckpt
    → FastAPI /generate ← Gradio UI
```

### Model architecture

**Backbone:** `Qwen/Qwen3-0.6B` — frozen in all training stages.
- `hidden_size = model.config.hidden_size` (= 1024 for Qwen3-0.6B; **never hardcode**)
- `num_hidden_layers = model.config.num_hidden_layers` (= 28; **never hardcode**)
- `target_layer = num_hidden_layers // 2` (= 14; **never hardcode**)

**OceanClassifierHead** (`call_center_simulator.models.components.ocean_classifier`):
- `Linear(hidden_size → 256) → ReLU → Dropout(0.1) → Linear(256 → 5) → Sigmoid`
- Input: mean-pooled last hidden state `[B, hidden_size]`
- Output: 5 values in `[0, 1]`, axis order **O, C, E, A, N**

**SteeringVectors** (`call_center_simulator.models.components.steering_vectors`):
- `nn.Parameter([5, hidden_size])`, initialized to zeros
- Injected via `register_forward_hook` on layer `target_layer`
- Delta: `ocean_profile @ vectors` → `[B, hidden_size]`, broadcast over seq_len
- Total trainable params: 5 × 1024 = **5 120**

**Loss function:**
```
L = CE_LM + λ × BCE(OceanClassifier(pooled_last_hidden), target_profile)
```
where `λ = cfg.model.lambda_steering = 0.1`

**OCEAN axis order (canonical throughout all code):** `["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]`

### Configuration system

**Hydra configs** (`configs/`):

- `configs/config.yaml` — root config composing data, model, train configs
- `configs/model/qwen.yaml` — backbone name, target_layer_fraction, lambda_steering
- `configs/model/ocean_classifier.yaml` — hidden_dim, dropout, onnx_path
- `configs/train/default.yaml` — epochs, lr, early stopping, checkpointing
- `configs/train/smoke.yaml` — fast CPU config for CI (1 epoch, 3 steps)
- `configs/data/essays.yaml` — CSV path, OCEAN columns, batch size, splits
- `configs/data/personachat.yaml` — HF dataset name, max_history_turns

**Critical:** Hydra changes working directory to a run-specific folder. Always use `cfg.paths.*` for file paths, never assume `cwd`.

### Data module

`EssaysDataModule` (`call_center_simulator.data.datamodule`):
- Loads Essays CSV, normalizes OCEAN columns to `[0, 1]`
- User-based split (no data leakage between users): 80% train / 10% val / 10% test
- Tokenizes with `AutoTokenizer` from backbone name
- Returns `TensorDataset(input_ids, attention_mask, ocean_labels)`

`PersonaChatDataModule`:
- Loads `bavard/personachat_truecased` from HuggingFace
- Builds `{history, response}` pairs for BLEU/ROUGE-L evaluation

### Serving

**FastAPI** (`call_center_simulator.inference.api`):
- `GET /health` — returns 503 if model not loaded
- `POST /generate` — generates client reply conditioned on OCEAN profile
- Validates `ocean_profile` fields in `[0.0, 1.0]` via Pydantic `Field(ge=0.0, le=1.0)`
- Returns 422 on out-of-range OCEAN values

**Gradio UI** (`call_center_simulator.inference.app`):
- 5 sliders for OCEAN profile (0.0–1.0)
- Textbox for situation + dialog history
- Calls FastAPI `/generate` via `httpx`

**CLI** (`call_center_simulator.cli`):
- Typer-based CLI: `download-data`, `train-ocean`, `train-steering`, `export-onnx`, `infer`, `serve-api`, `serve-ui`
- `train-ocean` and `train-steering` invoke Hydra entry-points via subprocess to avoid `sys.argv` conflicts
- Use `python -m call_center_simulator.cli <command>` for all operations

### DVC pipeline

6 stages in `dvc.yaml`:

| Stage | Command | Inputs | Outputs |
|---|---|---|---|
| download | `cli download-data` | `download.py` | `data/raw/essays.csv`, `data/raw/personachat` |
| preprocess | inline Python | `preprocessing.py`, essays.csv | `data/processed/essays_{train,val,test}.csv` |
| train_ocean_classifier | `training.train_ocean_classifier` | processed CSVs | `models/ocean_classifier_best.ckpt` |
| export_onnx | `inference.export_onnx` | classifier ckpt | `models/ocean_classifier.onnx` |
| train_steering | `training.train` | processed CSVs, classifier ckpt | `models/steering_best.ckpt` |
| evaluate | inline Python | `metrics.py`, steering ckpt | `models/metrics.json` |

Local DVC remote: `.dvc-storage/` in repo root.

### Package structure

```
call_center_simulator/
├── cli.py                          # Typer CLI entrypoint
├── data/
│   ├── download.py                 # Essays (GitHub raw) + PersonaChat (HF)
│   ├── preprocessing.py            # normalize_ocean, user_based_split, build_*_pairs
│   └── datamodule.py               # EssaysDataModule, PersonaChatDataModule
├── models/
│   ├── components/
│   │   ├── ocean_classifier.py     # OceanClassifierHead MLP
│   │   └── steering_vectors.py     # SteeringVectors nn.Parameter + hook
│   ├── ocean_classifier_module.py  # LightningModule: BCE pre-training
│   └── steering_module.py          # LightningModule: CE_LM + lambda*BCE
├── training/
│   ├── train_ocean_classifier.py   # Hydra entry-point
│   └── train.py                    # Hydra entry-point
├── inference/
│   ├── export_onnx.py              # OceanClassifierHead → ONNX
│   ├── api.py                      # FastAPI /generate + /health
│   ├── app.py                      # Gradio UI
│   └── infer.py                    # CLI inference utilities
└── utils/
    └── metrics.py                  # MAPE_ocean, Perplexity, BLEU, ROUGE-L, Distinct-1/2
```

## Key design decisions

1. **`hidden_size` from config**: Always use `model.config.hidden_size`, **never hardcode 1024**.
2. **`target_layer` from config**: Always use `model.config.num_hidden_layers // 2`, **never hardcode 14**.
3. **OCEAN axis order**: `["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]` (O, C, E, A, N) throughout all code.
4. **Smoke tests**: Use tiny-random model (`hidden_size=64`, `num_hidden_layers=2`), **never download real Qwen3-0.6B** in tests.
5. **No HF Trainer**: Training via PyTorch Lightning only.
6. **DVC local remote**: `.dvc-storage/` in repo root — no cloud credentials needed for Phase A.
7. **User-based split**: Essays split by `#AUTHID` to prevent data leakage between users.
8. **Subprocess for Hydra**: `train-ocean` and `train-steering` CLI commands use `subprocess` to invoke Hydra entry-points, avoiding `sys.argv` conflicts with Typer.
9. **Pydantic v2**: All API models use `Field(ge=0.0, le=1.0)` for OCEAN validation.
10. **MLflow tracking**: All training runs log git commit SHA, hyperparameters, and metrics.

## Environment variables

Defined in `.env.example`:

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token (optional, Qwen3-0.6B is non-gated) |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:8080` | MLflow server URI |
| `BACKBONE_NAME` | `Qwen/Qwen3-0.6B` | Backbone model name for API server |
| `STEERING_CKPT` | — | Path to trained steering vectors checkpoint |

## Test structure

```
tests/
├── unit/                           # 33 tests, pure Python/PyTorch, no model download
│   ├── test_metrics.py             # MAPE_ocean, Perplexity, BLEU, ROUGE-L, Distinct
│   ├── test_preprocessing.py       # normalize_ocean, user_based_split, build_*_pairs
│   ├── test_ocean_classifier.py    # OceanClassifierHead shape/range/gradient
│   └── test_steering_vectors.py    # SteeringVectors shape/trainable/apply_delta
└── smoke/                          # 8 tests, tiny-random model, CPU, < 5 s total
    ├── test_smoke_training.py      # OceanClassifierModule + SteeringModule train steps
    └── test_smoke_api.py           # FastAPI /health + /generate + 422 validation
```

All smoke tests use `AutoConfig.for_model("qwen2", hidden_size=64, num_hidden_layers=2)` — no real model download.
