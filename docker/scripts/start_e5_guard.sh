#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/app/models/multilingual-e5-small"
MODEL_ID="intfloat/multilingual-e5-small"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "[e5-guard] Model not found at $MODEL_DIR; downloading $MODEL_ID..."
  mkdir -p "$MODEL_DIR"
  python -c "
import os
from sentence_transformers import SentenceTransformer

token = os.environ.get('HF_TOKEN') or None
model = SentenceTransformer('$MODEL_ID', token=token)
model.save('$MODEL_DIR')
"
  echo "[e5-guard] Model saved to $MODEL_DIR."
else
  echo "[e5-guard] Using cached model at $MODEL_DIR."
fi

exec uvicorn clinical_guard.workers.e5_worker:app --host 0.0.0.0 --port 8001
