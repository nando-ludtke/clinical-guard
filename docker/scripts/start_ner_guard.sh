#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/app/models/multilang-pii-ner-ONNX"
MODEL_ID="onnx-community/multilang-pii-ner-ONNX"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "[ner-pii-guard] Model not found at $MODEL_DIR; downloading $MODEL_ID..."
  mkdir -p "$MODEL_DIR"
  python -c "
import os
from huggingface_hub import snapshot_download

token = os.environ.get('HF_TOKEN') or None
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='$MODEL_DIR',
    token=token,
)
"
  echo "[ner-pii-guard] Model saved to $MODEL_DIR."
else
  echo "[ner-pii-guard] Using cached model at $MODEL_DIR."
fi

exec uvicorn clinical_guard.workers.ner_worker:app --host 0.0.0.0 --port 8003
