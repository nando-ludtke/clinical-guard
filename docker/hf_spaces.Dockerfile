# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CLINICAL_GUARD_ENVIRONMENT=production \
    CLINICAL_GUARD_ENABLE_LOCAL_NER=true \
    CLINICAL_GUARD_RATE_LIMIT_ENABLED=true \
    CLINICAL_GUARD_RATE_LIMIT_REQUESTS=20 \
    CLINICAL_GUARD_RATE_LIMIT_WINDOW_SECONDS=60 \
    CLINICAL_GUARD_MAX_CONCURRENT_INFERENCES=4 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "fastapi>=0.115" "httpx>=0.27" "litellm>=1.50" \
    "pydantic>=2.9" "pydantic-settings>=2.5" "pyyaml>=6.0" "uvicorn[standard]>=0.30" \
    "torch>=2.2" "sentence-transformers>=3.0,<6.0" \
    "transformers>=4.45,<5.0" "onnxruntime>=1.19" "optimum[onnxruntime]>=1.23"

COPY src/ ./src/
COPY data/eval/*_benchmark.json ./data/eval/
COPY docker/scripts/download_models.py /app/docker/scripts/download_models.py
COPY docker/scripts/start_hf_spaces.sh /app/docker/scripts/start_hf_spaces.sh

RUN mkdir -p /app/models /app/.cache && \
    chmod +x /app/docker/scripts/start_hf_spaces.sh && \
    chmod -R 777 /app/models /app/.cache

ENV HF_HOME=/app/.cache

EXPOSE 7860

CMD ["/app/docker/scripts/start_hf_spaces.sh"]
