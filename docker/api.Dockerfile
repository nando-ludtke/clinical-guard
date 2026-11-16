# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "fastapi>=0.115" "httpx>=0.27" "litellm>=1.50" \
    "pydantic>=2.9" "pydantic-settings>=2.5" "pyyaml>=6.0" "uvicorn[standard]>=0.30"

COPY src/ ./src/

COPY data/eval/*_benchmark.json ./data/eval/

EXPOSE 8000

CMD ["uvicorn", "clinical_guard.main:app", "--host", "0.0.0.0", "--port", "8000"]
