#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import os
import sys
from pathlib import Path

models_root = Path("/app/models")
models_root.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN", "").strip() or None

def is_populated(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())

prompt_dir = models_root / "Llama-Prompt-Guard-2-86M"
if not is_populated(prompt_dir):
    if not token:
        print(
            "[single] ERROR: /app/models/Llama-Prompt-Guard-2-86M is missing and HF_TOKEN is not set.",
            file=sys.stderr,
        )
        sys.exit(1)
    from huggingface_hub import snapshot_download

    print("[single] Downloading meta-llama/Llama-Prompt-Guard-2-86M...")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="meta-llama/Llama-Prompt-Guard-2-86M", local_dir=str(prompt_dir), token=token)
else:
    print("[single] Using cached Llama-Prompt-Guard-2-86M.")

e5_dir = models_root / "multilingual-e5-small"
if not is_populated(e5_dir):
    from sentence_transformers import SentenceTransformer

    print("[single] Downloading intfloat/multilingual-e5-small...")
    e5_dir.mkdir(parents=True, exist_ok=True)
    SentenceTransformer("intfloat/multilingual-e5-small", token=token).save(str(e5_dir))
else:
    print("[single] Using cached multilingual-e5-small.")

ner_dir = models_root / "multilang-pii-ner-ONNX"
if not is_populated(ner_dir):
    from huggingface_hub import snapshot_download

    print("[single] Downloading onnx-community/multilang-pii-ner-ONNX...")
    ner_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="onnx-community/multilang-pii-ner-ONNX", local_dir=str(ner_dir), token=token)
else:
    print("[single] Using cached multilang-pii-ner-ONNX.")
PY

exec uvicorn clinical_guard.main:app --host 0.0.0.0 --port 8000
