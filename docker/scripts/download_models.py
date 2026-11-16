"""Ensure all required local models are present under /app/models before startup."""

import os
import sys
from pathlib import Path

MODELS_ROOT = Path(os.environ.get("CLINICAL_GUARD_MODELS_DIR", "/app/models"))

PROMPT_GUARD_DIR = MODELS_ROOT / "Llama-Prompt-Guard-2-86M"
PROMPT_GUARD_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

E5_DIR = MODELS_ROOT / "multilingual-e5-small"
E5_ID = "intfloat/multilingual-e5-small"

NER_DIR = MODELS_ROOT / "multilang-pii-ner-ONNX"
NER_ID = "onnx-community/multilang-pii-ner-ONNX"


def _is_populated(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def main() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    if not _is_populated(PROMPT_GUARD_DIR):
        if not hf_token:
            print(
                f"[hf-spaces] ERROR: {PROMPT_GUARD_DIR} is missing and HF_TOKEN is not set. "
                f"Add HF_TOKEN in Hugging Face Space Settings -> Variables and secrets "
                f"(after accepting the license for {PROMPT_GUARD_ID}).",
                file=sys.stderr,
            )
            sys.exit(1)
        from huggingface_hub import snapshot_download

        print(f"[hf-spaces] Downloading {PROMPT_GUARD_ID} to {PROMPT_GUARD_DIR}...")
        PROMPT_GUARD_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=PROMPT_GUARD_ID, local_dir=str(PROMPT_GUARD_DIR), token=hf_token)
        print(f"[hf-spaces] Saved {PROMPT_GUARD_ID}.")
    else:
        print(f"[hf-spaces] Using cached {PROMPT_GUARD_DIR}.")

    if not _is_populated(E5_DIR):
        from sentence_transformers import SentenceTransformer

        print(f"[hf-spaces] Downloading {E5_ID} to {E5_DIR}...")
        E5_DIR.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(E5_ID, token=hf_token)
        model.save(str(E5_DIR))
        print(f"[hf-spaces] Saved {E5_ID}.")
    else:
        print(f"[hf-spaces] Using cached {E5_DIR}.")

    if not _is_populated(NER_DIR):
        from huggingface_hub import snapshot_download

        print(f"[hf-spaces] Downloading {NER_ID} to {NER_DIR}...")
        NER_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=NER_ID, local_dir=str(NER_DIR), token=hf_token)
        print(f"[hf-spaces] Saved {NER_ID}.")
    else:
        print(f"[hf-spaces] Using cached {NER_DIR}.")


if __name__ == "__main__":
    main()
