# Call-Center Simulator

Мун Павел Юрьевич

Условная генерация текстовых реплик клиента колл-центра по заданному психологическому профилю (Big-5 / OCEAN) с использованием обучаемых steering vectors, вставляемых в замороженную модель Qwen3-0.6B.

## Постановка задачи

- **Сухой остаток:** условная генерация текстовых реплик клиента колл-центра по заданному психологическому профилю (Big-5 / OCEAN) и описанию ситуации.
- **Решаемая проблема:** обучение операторов колл-центра требует взаимодействия с разнообразными типами клиентов. Реальные тренировки дороги и труднопредсказуемы.
- **Решение:** текстовый бот с профилем клиента (5 числовых параметров OCEAN). Управление стилем — через learnable steering vectors, вставляемые в активации через forward hook.
- **Ценность:** система позволяет операторам тренироваться с виртуальными клиентами разных психотипов без привлечения реальных людей.

## Формат входных и выходных данных

- **Input:** POST-запрос к `/generate`:

  ```json
  {
    "history": [{"role": "operator", "text": "Здравствуйте, чем могу помочь?"}],
    "situation": "Клиент звонит по поводу задержки доставки",
    "ocean_profile": {
      "openness": 0.3,
      "conscientiousness": 0.7,
      "extraversion": 0.2,
      "agreeableness": 0.4,
      "neuroticism": 0.8
    },
    "max_new_tokens": 128
  }
  ```

- **Output:**

  ```json
  { "reply": "Я уже третий раз звоню! Где моя посылка?!" }
  ```

- **Протокол:** HTTP, FastAPI-сервер на порту 8000. Gradio UI на порту 7860.

## Датасет

- **Основной:** Essays (Mairesse/Pennebaker 2007) — 2 467 эссе с бинарными OCEAN-метками. Загружается с публичного GitHub-зеркала.
- **Вспомогательный:** PersonaChat (`bavard/personachat_truecased` через HuggingFace datasets) — диалоговые пары для оценки BLEU/ROUGE-L.
- **Разбивка:** user-based split 80/10/10 (seed=42) — без утечки данных между пользователями.

## Метрики

| Метрика | Целевое значение | Описание |
|---|---|---|
| MAPE_ocean | < 0.25 | Средняя абсолютная процентная ошибка по 5 осям OCEAN |
| Perplexity | < 50 | Перплексия языковой модели |
| BLEU | > 0.10 | Корпусный BLEU-4 на PersonaChat |
| ROUGE-L | > 0.20 | Средний ROUGE-L F1 на PersonaChat |
| Distinct-1 | > 0.5 | Доля уникальных унiграмм (разнообразие) |
| Distinct-2 | > 0.7 | Доля уникальных биграмм (разнообразие) |

## Моделирование

### Архитектура

Двухэтапный pipeline:

1. **Этап 1 — OceanClassifierHead** (предобучение на Essays):
   - Замороженный backbone Qwen3-0.6B (`hidden_size=1024`, 28 слоёв)
   - MLP-голова: `Linear(1024→256) → ReLU → Dropout(0.1) → Linear(256→5) → Sigmoid`
   - Loss: BCE, оптимизатор: Adam
   - Экспорт в ONNX для инференса

2. **Этап 2 — SteeringVectors** (обучение управляющих векторов):
   - Замороженный backbone + замороженный OCEAN-классификатор
   - 5 обучаемых векторов `nn.Parameter([5, 1024])`, инициализация нулями
   - Инъекция через `register_forward_hook` на слой 14 (= `num_hidden_layers // 2`)
   - Loss: `CE_LM + 0.1 × BCE(OCEAN_classifier(pooled_hidden), target_profile)`
   - Обучаемых параметров: 5 × 1024 = **5 120**

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
Essays CSV ──────────> EssaysDataModule (user-based split 80/10/10)
                              |
                              v
                    OceanClassifierModule
                    (frozen Qwen3-0.6B + MLP head)
                    BCE loss ──> ocean_classifier.onnx
                              |
                              v
                        SteeringModule
                    (frozen backbone + frozen OCEAN clf)
                    CE_LM + 0.1*BCE ──> steering_best.ckpt
```

## Setup

### Предварительные требования

- [Python 3.11](https://www.python.org/downloads/)
- Пакетный менеджер [uv](https://github.com/astral-sh/uv)
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/) (для развёртывания)

### Установка

1. Склонировать репозиторий:

```bash
git clone <repo-url>
cd mlops-call-center-simulator
```

2. Создать окружение и установить зависимости:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

3. Настроить pre-commit хуки:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

4. Настроить переменные окружения:

```bash
cp .env.example .env
# Отредактировать .env: добавить HF_TOKEN при необходимости
# (Qwen3-0.6B — публичная модель, токен не обязателен)
```

## Train

### 1. Загрузка данных

```bash
# Через DVC pipeline
uv run dvc repro download

# Или напрямую через CLI
uv run python -m call_center_simulator.cli download-data
```

### 2. Запуск MLflow

```bash
uv run mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

> **Примечание:** При развёртывании через Docker Compose MLflow будет доступен на порту 5000.

### 3. Обучение OCEAN-классификатора

```bash
uv run python -m call_center_simulator.cli train-ocean
```

### 4. Экспорт в ONNX

```bash
uv run python -m call_center_simulator.cli export-onnx
```

### 5. Обучение steering vectors

```bash
uv run python -m call_center_simulator.cli train-steering
```

### 6. Полный DVC pipeline

```bash
uv run dvc repro
```

DVC pipeline: `download → preprocess → train_ocean_classifier → export_onnx → train_steering → evaluate`

### Конфигурация

Гиперпараметры настраиваются в директории `configs/`:

- `configs/config.yaml` — корневой конфиг (seed, пути, MLflow)
- `configs/model/qwen.yaml` — backbone, lambda_steering
- `configs/model/ocean_classifier.yaml` — hidden_dim, dropout
- `configs/train/default.yaml` — epochs, lr, early stopping, checkpointing
- `configs/train/smoke.yaml` — быстрый конфиг для CI (1 эпоха, 3 шага, CPU)
- `configs/data/essays.yaml` — пути к данным, OCEAN-колонки
- `configs/data/personachat.yaml` — HF dataset, max_history

## Production preparation

### Артефакты для поставки

| Артефакт | Путь | Описание |
|---|---|---|
| ONNX-модель | `models/ocean_classifier.onnx` | Экспортированный OCEAN-классификатор |
| Checkpoint | `models/steering_best.ckpt` | Обученные steering vectors |
| API-сервер | `call_center_simulator/inference/api.py` | FastAPI /generate |
| Конфигурация | `configs/` | Hydra-конфиги |

## Infer

### CLI

```bash
uv run python -m call_center_simulator.cli infer \
  --situation "Клиент звонит по поводу задержки доставки" \
  --neuroticism 0.8 --agreeableness 0.3
```

### API-сервер (FastAPI)

```bash
uv run python -m call_center_simulator.cli serve-api
# POST http://localhost:8000/generate
```

Эндпоинты:

- `GET /health` — проверка состояния (503 если модель не загружена)
- `POST /generate` — генерация реплики клиента

Пример запроса:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "history": [{"role": "operator", "text": "Здравствуйте!"}],
    "situation": "Задержка доставки",
    "ocean_profile": {
      "openness": 0.3, "conscientiousness": 0.7,
      "extraversion": 0.2, "agreeableness": 0.4, "neuroticism": 0.8
    },
    "max_new_tokens": 128
  }'
```

### Gradio UI

```bash
uv run python -m call_center_simulator.cli serve-ui
# Открыть http://localhost:7860
```

### Docker Compose

```bash
docker compose up --build
```

Сервисы:

- `api` — FastAPI + модель на порту **8000**
- `gradio` — Gradio UI на порту **7860**
- `mlflow` — сервер MLflow на порту **5000**

## Разработка

### Запуск тестов

```bash
# Все тесты (33 unit + 8 smoke, < 5 с на CPU)
uv run pytest

# Только unit-тесты
uv run pytest tests/unit/ -v

# Только smoke-тесты (tiny-random модель, CPU)
uv run pytest tests/smoke/ -v --timeout=30

# С покрытием
uv run pytest --cov=call_center_simulator --cov-report=term-missing
```

### Качество кода

```bash
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

## Overall

### Структура проекта

```
mlops-call-center-simulator/
├── call_center_simulator/
│   ├── cli.py                    # Typer CLI
│   ├── data/
│   │   ├── download.py           # Essays + PersonaChat
│   │   ├── preprocessing.py      # normalize_ocean, user_based_split
│   │   └── datamodule.py         # EssaysDataModule, PersonaChatDataModule
│   ├── models/
│   │   ├── components/
│   │   │   ├── ocean_classifier.py  # OceanClassifierHead MLP
│   │   │   └── steering_vectors.py  # SteeringVectors + hook
│   │   ├── ocean_classifier_module.py  # LightningModule
│   │   └── steering_module.py          # LightningModule
│   ├── training/
│   │   ├── train_ocean_classifier.py   # Hydra entry-point
│   │   └── train.py                    # Hydra entry-point
│   ├── inference/
│   │   ├── export_onnx.py        # OceanClassifierHead → ONNX
│   │   ├── api.py                # FastAPI /generate
│   │   ├── app.py                # Gradio UI
│   │   └── infer.py              # CLI inference
│   └── utils/
│       └── metrics.py            # MAPE_ocean, Perplexity, BLEU, ROUGE-L, Distinct
├── configs/                      # Hydra configs
├── tests/
│   ├── unit/                     # 33 unit-теста (TDD)
│   └── smoke/                    # 8 smoke-тестов (tiny-random, CPU)
├── data/                         # DVC-managed datasets
├── models/                       # DVC-managed checkpoints + ONNX
├── dvc.yaml                      # DVC pipeline (6 стадий)
├── Dockerfile                    # Multi-stage
└── docker-compose.yml            # api + gradio + mlflow
```

### Phase B (в разработке)

Phase B — реальное GPU-обучение на Google Colab:

- Загрузка полного датасета Essays (2 467 записей)
- Обучение OCEAN-классификатора на GPU (10 эпох)
- Обучение steering vectors на GPU (10 эпох)
- Оценка на PersonaChat (BLEU, ROUGE-L, Distinct)
- Публикация обученных весов через DVC remote

## Лицензия

MIT
