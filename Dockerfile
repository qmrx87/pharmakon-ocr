# =============================================================================
# VignOCR serving image — CPU inference (ONNX Runtime), NOT the full [ml] extra.
# -----------------------------------------------------------------------------
# Ships the stateless FastAPI service (vignocr.serving.app:app). Installs the
# light *core* deps from pyproject (fastapi/uvicorn/pydantic/pillow/... ) plus
# onnxruntime for CPU inference — deliberately WITHOUT torch / rfdetr / paddle
# (those live in the `.[ml]` extra and are only needed for training/export on
# GPU hosts). The image runs `docker run` today against a fixture/stub model:
# if no weights are mounted, the pipeline falls back to the deterministic stub
# (see vignocr.serving.deps), so /health, /ready and /extract all answer.
#
# Build:  docker build -t vignocr-serving .
# Run:    docker run --rm -p 8000:8000 vignocr-serving
#         # with real weights:
#         docker run --rm -p 8000:8000 \
#           -e VIGNOCR_DETECTOR_PATH=/models/detector.onnx \
#           -e VIGNOCR_RECOGNIZER_PATH=/models/recognizer.onnx \
#           -e VIGNOCR_ALLOW_STUB=0 \
#           -v "$PWD/models:/models:ro" vignocr-serving
#
# For a GPU TRAINING/EXPORT image instead, see the commented variant at the
# bottom of this file (CUDA base + the `.[ml]` extra).
# =============================================================================
FROM python:3.11-slim AS runtime

# --- runtime env -------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 12-factor defaults (override at `docker run` time with -e):
    VIGNOCR_LOG_JSON=1 \
    VIGNOCR_DEFAULT_FLOW=selling \
    VIGNOCR_MAX_UPLOAD_MB=10 \
    # Resolve configs/ from a fixed location regardless of CWD:
    VIGNOCR_REPO_ROOT=/app

WORKDIR /app

# --- OS deps -----------------------------------------------------------------
# libgomp1 is the OpenMP runtime onnxruntime links against. Keep the layer slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps -------------------------------------------------------------
# Copy only what the build needs first so the (slow) dependency layer is cached
# and not invalidated by source/config edits.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the CORE deps + the project (NOT `.[ml]`), then onnxruntime for CPU
# inference (version pinned to match the `[ml]` extra in pyproject).
RUN pip install --upgrade pip \
    && pip install . \
    && pip install "onnxruntime==1.20.1"

# --- App assets --------------------------------------------------------------
# configs/ is the single source of truth (read at runtime via vignocr.common).
# Copied last because it changes most often and must not bust the pip layer.
COPY configs ./configs

# --- non-root user -----------------------------------------------------------
RUN useradd --create-home --uid 10001 vignocr \
    && chown -R vignocr:vignocr /app
USER vignocr

EXPOSE 8000

# Liveness probe hits /health (no model load).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

# Stateless single-worker by default; scale with a process manager / replicas.
CMD ["uvicorn", "vignocr.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# GPU variant (training / ONNX export — the heavy `.[ml]` stack).
# Not used for serving; documented here so the two images stay in sync.
# -----------------------------------------------------------------------------
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS ml
# ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         python3.11 python3-pip libgl1 libglib2.0-0 libgomp1 \
#     && rm -rf /var/lib/apt/lists/*
# WORKDIR /app
# COPY pyproject.toml README.md ./
# COPY src ./src
# # CUDA torch wheels come from the module stack / wheelhouse on Narval; install
# # the full extra (torch/torchvision/rfdetr/onnx/paddleocr/...):
# RUN pip install ".[ml]"
# COPY configs ./configs
# # Training/export is launched via the CLI, e.g.:
# #   docker run --gpus all vignocr-ml vignocr detection train --config configs/detection/rfdetr_medium.yaml
# CMD ["vignocr", "--help"]
# =============================================================================
