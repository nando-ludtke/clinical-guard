#!/usr/bin/env bash
set -euo pipefail

python /app/docker/scripts/download_models.py

exec uvicorn clinical_guard.main:app --host 0.0.0.0 --port "${PORT:-7860}"
