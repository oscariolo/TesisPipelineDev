# Build context: scripts/. Build with:
#   docker build -t log-analysis .
#   docker build -t log-analysis --build-arg GPU_BACKEND=cuda .
#   docker build -t log-analysis --build-arg GPU_BACKEND=cpu .
#
# The default GPU_BACKEND=rocm matches the current environment (ROCm 7.2).
# See scripts/AGENTS.md for the full CLI reference.

ARG GPU_BACKEND=rocm

# ── Base stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── ROCm backend (default) ────────────────────────────────────────────────
FROM base AS rocm

# Note: Keeping your original spelling of "requierements.txt"
COPY requierements.txt .

# Use BuildKit cache mount for pip and remove --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/rocm7.2

RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^torch==|^torchvision==' requierements.txt > /tmp/deps.txt \
    && pip install -r /tmp/deps.txt

# ── CUDA backend ──────────────────────────────────────────────────────────
FROM base AS cuda

COPY requierements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^torch==|^torchvision==|^triton-rocm==' requierements.txt > /tmp/deps.txt \
    && pip install -r /tmp/deps.txt

# ── CPU backend ───────────────────────────────────────────────────────────
FROM base AS cpu

COPY requierements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^torch==|^torchvision==|^nvidia-|^cuda-|^triton' requierements.txt > /tmp/deps.txt \
    && pip install -r /tmp/deps.txt \
    && pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── Final stage ───────────────────────────────────────────────────────────
FROM ${GPU_BACKEND} AS final

COPY . .

VOLUME /app/logs
VOLUME /app/analysis

ENTRYPOINT ["python", "main.py"]