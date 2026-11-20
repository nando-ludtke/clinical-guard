"""System 1 prompt injection safety guard wrapping Llama-Prompt-Guard-2-86M on CPU."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_guard.safety._load_lock import HF_MODEL_LOAD_LOCK

try:
    import torch
except ImportError:  # pragma: no cover - optional ml extra
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - optional ml extra
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

DEFAULT_PROMPT_GUARD_HF_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
LOCAL_PROMPT_GUARD_MODEL_DIR = Path("models/Llama-Prompt-Guard-2-86M")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENIGN_LABELS = frozenset({"BENIGN", "SAFE", "LABEL_0"})
_DELIMITER_SMUGGLING_PATTERN = re.compile(
    r"(?:<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[/?INST\]|<</?SYS>>|"
    r"</?(?:system|assistant)>|###\s*(?:System|Instruction)\s*:)",
    re.IGNORECASE,
)


def resolve_default_prompt_guard_model() -> str:
    """Return local Prompt Guard model path if present, otherwise the HuggingFace ID."""
    for candidate in (
        LOCAL_PROMPT_GUARD_MODEL_DIR,
        _REPO_ROOT / LOCAL_PROMPT_GUARD_MODEL_DIR,
    ):
        if candidate.is_dir():
            return str(candidate)
    return DEFAULT_PROMPT_GUARD_HF_MODEL


DEFAULT_PROMPT_GUARD_MODEL = resolve_default_prompt_guard_model()


@dataclass(frozen=True, slots=True)
class PromptGuardResult:
    """Outcome of the System 1 prompt-injection classification check."""

    is_injection: bool
    label: str
    score: float
    latency_ms: float


class PromptInjectionGuard:
    """CPU-based prompt-injection classifier using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.model_name = model_name or resolve_default_prompt_guard_model()
        self.threshold = threshold
        self.device = device
        self.max_length = max_length

        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Return True when tokenizer and model weights are initialized."""
        return self._tokenizer is not None and self._model is not None

    def load(self) -> None:
        """Load tokenizer and sequence classification model onto CPU in eval mode."""
        if self.is_loaded:
            return
        with HF_MODEL_LOAD_LOCK:
            if self.is_loaded:
                return
            if (
                torch is None
                or AutoTokenizer is None
                or AutoModelForSequenceClassification is None
            ):
                raise ImportError(
                    "PromptInjectionGuard requires 'torch' and 'transformers'. "
                    "Install with: pip install 'clinical-guard[ml]'"
                )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model

    def evaluate(self, query: str) -> PromptGuardResult:
        """Classify whether the incoming query contains a prompt injection."""
        t0 = time.perf_counter()
        if _DELIMITER_SMUGGLING_PATTERN.search(query):
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return PromptGuardResult(
                is_injection=True,
                label="DELIMITER_SMUGGLING",
                score=1.0,
                latency_ms=latency_ms,
            )

        if not self.is_loaded:
            self.load()

        assert self._tokenizer is not None
        assert self._model is not None

        t0 = time.perf_counter()
        inputs = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        id2label: dict[int, str] = getattr(
            getattr(self._model, "config", None),
            "id2label",
            None,
        ) or {0: "BENIGN", 1: "INJECTION"}

        pred_idx = int(torch.argmax(probs).item())
        pred_label = str(id2label.get(pred_idx, f"LABEL_{pred_idx}"))

        malicious_indices = [
            idx
            for idx, lbl in id2label.items()
            if str(lbl).upper() not in _BENIGN_LABELS
        ]
        if malicious_indices:
            injection_score = float(sum(float(probs[idx].item()) for idx in malicious_indices))
        elif probs.shape[0] > 1:
            injection_score = float(probs[1].item())
        else:
            injection_score = float(probs[0].item())

        is_injection = injection_score >= self.threshold
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        return PromptGuardResult(
            is_injection=is_injection,
            label=pred_label,
            score=round(injection_score, 6),
            latency_ms=latency_ms,
        )
