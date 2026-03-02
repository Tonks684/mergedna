# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
ENV PYTHONPATH=/workspace

# Tell uv to create/use a venv outside /workspace (so mounts won't overwrite it)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy dependency metadata first (cacheable)
COPY pyproject.toml uv.lock ./

# Runtime deps only
RUN uv sync --no-dev

# --- DEV IMAGE (adds pytest etc.) ---
FROM base AS dev
RUN uv sync --group dev
CMD ["bash"]

# --- FINAL IMAGE ---
FROM base AS final
COPY mergedna/ mergedna/
COPY nanochat/ nanochat/
COPY scripts/ scripts/
COPY tests/ tests/
CMD ["pytest", "-q"]