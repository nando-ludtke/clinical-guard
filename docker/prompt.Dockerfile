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
    "torch>=2.2" "transformers>=4.45,<5.0"

COPY src/ ./src/

COPY docker/scripts/start_prompt_guard.sh /app/docker/scripts/start_prompt_guard.sh
RUN chmod +x /app/docker/scripts/start_prompt_guard.sh

EXPOSE 8002

CMD ["/app/docker/scripts/start_prompt_guard.sh"]
