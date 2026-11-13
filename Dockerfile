# Kerala Voice Helpdesk API: CPU only, Gemma 4 E2B Q4_0 in-process via llama-cpp-python.
#   docker compose up --build        (see README "Run locally with Docker")
# Base image pinned by digest (python:3.12.16-slim-bookworm, linux/amd64 + arm64 index).
ARG BASE=python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# --- build: compile llama-cpp-python once into a wheel --------------------------------------
FROM ${BASE} AS build
ARG LLAMA_CPP_PYTHON=0.3.36
# Portable x86-64 build (AVX2/FMA: any Intel Mac since 2013, any recent PC), no BLAS, no GPU.
# GGML_NATIVE=ON would tune for the build machine's CPU and may crash on another one.
ARG CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF"
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
RUN CMAKE_ARGS="${CMAKE_ARGS}" pip wheel --no-cache-dir --no-deps -w /wheels "llama-cpp-python==${LLAMA_CPP_PYTHON}"
COPY requirements-api.txt /tmp/
RUN pip wheel --no-cache-dir -w /wheels -r /tmp/requirements-api.txt

# --- runtime ------------------------------------------------------------------------------
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin helpdesk \
    && mkdir -p /models && chown helpdesk:helpdesk /models
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl && rm -rf /wheels

WORKDIR /app
COPY --chown=helpdesk:helpdesk helpdesk/ helpdesk/
COPY --chown=helpdesk:helpdesk data/ data/
USER helpdesk

# The GGUF (~3 GB) is downloaded on first start into /models (a volume), public repo, no token.
ENV HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    N_THREADS=2 \
    OMP_NUM_THREADS=2 \
    HELPDESK_LLM_BACKEND=inproc \
    HELPDESK_CANON=1 \
    HELPDESK_CTX_K=1 \
    HELPDESK_DENSE=0 \
    HELPDESK_WARMUP=kb \
    HELPDESK_CACHE_MB=2048
EXPOSE 8000
# Healthy once warm-up is done (status "ok"); the first start includes the model download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import json,sys,urllib.request as u; sys.exit(json.load(u.urlopen('http://127.0.0.1:8000/health', timeout=4))['status'] != 'ok')"
CMD ["uvicorn", "helpdesk.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
