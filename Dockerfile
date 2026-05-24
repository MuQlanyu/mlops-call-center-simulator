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
