# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "fastapi>=0.115" "httpx>=0.27" "litellm>=1.50" \
    "pydantic>=2.9" "pydantic-settings>=2.5" "pyyaml>=6.0" "uvicorn[standard]>=0.30" \
    "onnxruntime>=1.19" "optimum[onnxruntime]>=1.23" "transformers>=4.45,<5.0"

COPY src/ ./src/

COPY docker/scripts/start_ner_guard.sh /app/docker/scripts/start_ner_guard.sh
RUN chmod +x /app/docker/scripts/start_ner_guard.sh

EXPOSE 8003

CMD ["/app/docker/scripts/start_ner_guard.sh"]
