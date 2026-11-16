#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/app/models/Llama-Prompt-Guard-2-86M"
MODEL_ID="meta-llama/Llama-Prompt-Guard-2-86M"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "[prompt-guard] Model not found at $MODEL_DIR; downloading $MODEL_ID..."
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
  echo "[prompt-guard] Model saved to $MODEL_DIR."
else
  echo "[prompt-guard] Using cached model at $MODEL_DIR."
fi

exec uvicorn clinical_guard.workers.prompt_worker:app --host 0.0.0.0 --port 8002
