"""Shared process-wide lock for HuggingFace/PyTorch model loading."""

import threading

HF_MODEL_LOAD_LOCK = threading.Lock()
