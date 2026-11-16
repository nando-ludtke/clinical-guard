"""Request-scoped entity anonymization and uncovering for clinical context evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from optimum.onnxruntime import ORTModelForTokenClassification
except ImportError:  # pragma: no cover - optional privacy/ml extra
    ORTModelForTokenClassification = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer, pipeline
except ImportError:  # pragma: no cover - optional privacy/ml extra
    AutoTokenizer = None  # type: ignore[assignment]
    pipeline = None  # type: ignore[assignment]

CustomEntityDetector = Callable[[str], Sequence["EntitySpan"]]

DEFAULT_NER_HF_MODEL = "onnx-community/multilang-pii-ner-ONNX"
LOCAL_NER_MODEL_DIR = Path("models/multilang-pii-ner-ONNX")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_default_ner_model() -> str:
    """Return local ONNX PII NER model path if present, otherwise the HuggingFace ID."""
    for candidate in (
        LOCAL_NER_MODEL_DIR,
        _REPO_ROOT / LOCAL_NER_MODEL_DIR,
    ):
        if candidate.is_dir():
            return str(candidate)
    return DEFAULT_NER_HF_MODEL


DEFAULT_NER_MODEL = resolve_default_ner_model()

_NER_LABEL_MAP: dict[str, str] = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "FIRSTNAME": "PERSON",
    "LASTNAME": "PERSON",
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "NAME": "PERSON",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "CITY": "LOCATION",
    "STREET": "LOCATION",
    "ADDRESS": "LOCATION",
    "ZIPCODE": "LOCATION",
    "POSTALCODE": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    "DATE": "DATE",
    "DOB": "DATE",
    "DATEOFBIRTH": "DATE",
    "TIME": "DATE",
    "EMAIL": "EMAIL",
    "EMAILADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "TEL": "PHONE",
    "ID": "ID",
    "IDCARD": "ID",
    "PASSPORT": "ID",
    "SSN": "ID",
    "DRIVERLICENSE": "ID",
    "TAXNUM": "ID",
    "ACCOUNTNUM": "ID",
    "CREDITCARDNUMBER": "ID",
    "IBAN": "ID",
}

IGNORED_NER_ENTITY_TYPES: frozenset[str] = frozenset({"AGE"})

_BASELINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "ID",
        re.compile(
            r"\b(?:MRN|PATIENT[\s_-]*ID|PID)\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{3,15}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ID",
        re.compile(
            r"\b(?:[A-Za-z]+[- ]?)?id\s*[:#]\s*[A-Za-z0-9-]{4,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)"
        ),
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:"
            r"\d{4}-\d{2}-\d{2}"
            r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
            r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?"
            r"|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PERSON",
        re.compile(
            r"\b(?:Dr|Mr|Mrs|Ms|Prof|Patient)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """Character span of a detected entity within a source string."""

    start: int
    end: int
    entity_type: str


def detect_baseline_entities(text: str) -> list[EntitySpan]:
    """Detect baseline PHI/PII entities using deterministic regex patterns."""
    spans: list[EntitySpan] = []
    for entity_type, pattern in _BASELINE_PATTERNS:
        for match in pattern.finditer(text):
            spans.append(
                EntitySpan(
                    start=match.start(),
                    end=match.end(),
                    entity_type=entity_type,
                )
            )
    return spans


def _resolve_non_overlapping_spans(
    spans: Sequence[EntitySpan],
    text_length: int,
) -> list[EntitySpan]:
    """Filter invalid spans and greedily resolve overlaps (earliest, then longest)."""
    valid = [
        s for s in spans if 0 <= s.start < s.end <= text_length and s.entity_type.strip()
    ]
    ordered = sorted(valid, key=lambda s: (s.start, -(s.end - s.start)))
    resolved: list[EntitySpan] = []
    cursor = 0
    for span in ordered:
        if span.start >= cursor:
            resolved.append(span)
            cursor = span.end
    return resolved


def merge_entity_spans(
    primary_spans: Sequence[EntitySpan],
    secondary_spans: Sequence[EntitySpan],
    text_length: int,
) -> list[EntitySpan]:
    """Merge primary (e.g. regex) and secondary (e.g. NER) spans with primary priority.

    Primary spans are resolved first. Secondary spans are then added in
    (earliest, longest) order only when they do not overlap any already-accepted span.
    """
    accepted = _resolve_non_overlapping_spans(primary_spans, text_length)
    valid_secondary = [
        s for s in secondary_spans if 0 <= s.start < s.end <= text_length and s.entity_type.strip()
    ]
    ordered_secondary = sorted(valid_secondary, key=lambda s: (s.start, -(s.end - s.start)))

    for candidate in ordered_secondary:
        overlaps = any(
            candidate.start < existing.end and candidate.end > existing.start
            for existing in accepted
        )
        if not overlaps:
            accepted.append(candidate)

    accepted.sort(key=lambda s: s.start)
    return accepted


def _merge_adjacent_person_spans(spans: Sequence[EntitySpan], text: str) -> list[EntitySpan]:
    """Merge adjacent PERSON spans separated only by whitespace or hyphens."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[EntitySpan] = [ordered[0]]
    for span in ordered[1:]:
        prev = merged[-1]
        if (
            prev.entity_type == "PERSON"
            and span.entity_type == "PERSON"
            and span.start >= prev.end
        ):
            gap = text[prev.end : span.start]
            if gap == "" or all(ch.isspace() or ch == "-" for ch in gap):
                merged[-1] = EntitySpan(
                    start=prev.start,
                    end=max(prev.end, span.end),
                    entity_type="PERSON",
                )
                continue
        merged.append(span)
    return merged


def _normalize_ner_label(raw_label: str) -> str:
    """Normalize raw NER entity/group labels (e.g. B-PER, LOC) to clean placeholder types."""
    cleaned = raw_label.strip().upper()
    if not cleaned or cleaned == "O":
        return ""
    for prefix in ("B-", "I-", "S-", "E-"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if not cleaned or cleaned == "O":
        return ""
    return _NER_LABEL_MAP.get(cleaned, cleaned)


_ADDRESS_CONTEXT_PATTERN = re.compile(
    r"(?:\b(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|terrace|way|"
    r"place|pl|court|ct|allee|straße|strasse|str|platz|weg|gasse|address|adresse|anschrift)\b"
    r"|straße|strasse|platz|allee)",
    re.IGNORECASE,
)

_CLINICAL_UNIT_AFTER_PATTERN = re.compile(
    r"^\s*(?:%|(?:°[cf]|mg/kg|mg/dl|u/l|mmol|mcg|µg|ug|kg|mg|ml|dl|mmhg|bpm|iu|ie|g|l|"
    r"stunden|std|min|sec|days|tage|weeks|wochen|months|monate|h)\b)",
    re.IGNORECASE,
)


def _has_address_context(text: str, start: int, end: int, window: int = 40) -> bool:
    """Return True when a span is adjacent to street or address context."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return bool(_ADDRESS_CONTEXT_PATTERN.search(text[ctx_start:ctx_end]))


class OnnxPiiDetector:
    """Lazy-loading ONNX token-classification PII detector."""

    def __init__(
        self,
        model_name: str | None = None,
        min_score: float = 0.35,
        device: int | str = -1,
        aggregation_strategy: str = "simple",
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")
        self.model_name = model_name or resolve_default_ner_model()
        self.min_score = min_score
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self._pipeline: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Return True once the ONNX NER pipeline has been initialized."""
        return self._pipeline is not None

    def load(self) -> None:
        """Lazy-load tokenizer, ORT token classification model, and NER pipeline."""
        if self._pipeline is not None:
            return
        if AutoTokenizer is None or pipeline is None or ORTModelForTokenClassification is None:
            raise ImportError(
                "OnnxPiiDetector requires 'transformers' and 'optimum[onnxruntime]'. "
                "Install with: pip install 'clinical-guard[privacy]'"
            )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        load_kwargs: dict[str, Any] = {}
        model_path = Path(self.model_name)
        if model_path.is_dir() and not (model_path / "model.onnx").is_file():
            for subfolder, file_name in (
                ("onnx", "model.onnx"),
                ("onnx", "model_quantized.onnx"),
            ):
                if (model_path / subfolder / file_name).is_file():
                    load_kwargs["subfolder"] = subfolder
                    load_kwargs["file_name"] = file_name
                    break
        model = ORTModelForTokenClassification.from_pretrained(self.model_name, **load_kwargs)
        self._pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            device=self.device,
        )

    def detect(self, text: str) -> list[EntitySpan]:
        """Detect PII entities in text and return normalized EntitySpans."""
        if not text or not text.strip():
            return []

        self.load()
        assert self._pipeline is not None
        raw_entities = self._pipeline(text)

        spans: list[EntitySpan] = []
        text_len = len(text)
        for ent in raw_entities:
            score = float(ent.get("score", 1.0))
            if score < self.min_score:
                continue

            raw_group = str(ent.get("entity_group") or ent.get("entity") or "")
            entity_type = _normalize_ner_label(raw_group)
            if not entity_type or entity_type in IGNORED_NER_ENTITY_TYPES:
                continue

            start = ent.get("start")
            end = ent.get("end")
            if start is None or end is None:
                continue

            start_i = max(0, int(start))
            end_i = min(text_len, int(end))
            while start_i < end_i and text[start_i].isspace():
                start_i += 1
            while end_i > start_i and text[end_i - 1].isspace():
                end_i -= 1
            if start_i >= end_i:
                continue

            if (
                entity_type in ("PHONE", "TELEPHONENUM")
                and start_i > 0
                and text[start_i - 1] == "+"
            ):
                start_i -= 1

            span_text = text[start_i:end_i]
            if len(span_text.strip()) < 2 or not any(ch.isalnum() for ch in span_text):
                continue

            # Reject subword fragments that cut mid-word (e.g. "Met" in "Metformin").
            if (start_i > 0 and text[start_i - 1].isalnum()) or (
                end_i < text_len and text[end_i].isalnum()
            ):
                continue

            if entity_type == "BUILDINGNUM" and not _has_address_context(text, start_i, end_i):
                continue

            is_numeric_span = any(ch.isdigit() for ch in span_text) and all(
                ch.isdigit() or ch in ".,-" for ch in span_text
            )
            if (
                entity_type == "BUILDINGNUM" or is_numeric_span
            ) and _CLINICAL_UNIT_AFTER_PATTERN.match(text[end_i:]):
                continue

            spans.append(EntitySpan(start=start_i, end=end_i, entity_type=entity_type))

        return _merge_adjacent_person_spans(spans, text)

    def __call__(self, text: str) -> list[EntitySpan]:
        """Allow OnnxPiiDetector to be used directly as a CustomEntityDetector."""
        return self.detect(text)


class EntityTable:
    """Request-scoped bidirectional mapping between entities and <TYPE_N> placeholders."""

    def __init__(self, detector: CustomEntityDetector | None = None) -> None:
        self._detector: CustomEntityDetector = detector or detect_baseline_entities
        self._key_to_id: dict[tuple[str, str], str] = {}
        self._id_to_text: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    @property
    def mapping(self) -> dict[str, str]:
        """Return a copy of the placeholder -> original text mapping."""
        return dict(self._id_to_text)

    def __len__(self) -> int:
        return len(self._id_to_text)

    def get_or_create(self, entity_type: str, original_text: str) -> str:
        """Return a consistent placeholder ID for (entity_type, normalized_text)."""
        norm_type = entity_type.strip().upper()
        norm_text = " ".join(original_text.split()).casefold()
        key = (norm_type, norm_text)
        existing = self._key_to_id.get(key)
        if existing is not None:
            return existing

        next_idx = self._counters.get(norm_type, 0) + 1
        self._counters[norm_type] = next_idx
        placeholder = f"<{norm_type}_{next_idx}>"
        self._key_to_id[key] = placeholder
        self._id_to_text[placeholder] = original_text
        return placeholder

    def anonymize(self, text: str) -> str:
        """Replace detected entities in text with consistent <TYPE_N> placeholders."""
        if not text:
            return text

        spans = _resolve_non_overlapping_spans(self._detector(text), len(text))
        if not spans:
            return text

        parts: list[str] = []
        cursor = 0
        for span in spans:
            parts.append(text[cursor : span.start])
            raw_entity = text[span.start : span.end]
            parts.append(self.get_or_create(span.entity_type, raw_entity))
            cursor = span.end
        parts.append(text[cursor:])
        return "".join(parts)

    def deanonymize(self, text: str) -> str:
        """Uncover <TYPE_N> placeholders in text back to their original entity text."""
        if not text or not self._id_to_text:
            return text

        pattern = re.compile(
            "|".join(
                re.escape(placeholder)
                for placeholder in sorted(self._id_to_text, key=len, reverse=True)
            )
        )
        return pattern.sub(lambda match: self._id_to_text[match.group(0)], text)


class NerDetectorProtocol(Protocol):
    """Protocol for local or remote NER span detectors."""

    def detect(self, text: str) -> list[EntitySpan]:
        """Detect PII/PHI entity spans within text."""
        ...


class LocalAnonymizer:
    """Local PHI/PII anonymizer combining baseline regexes, optional ONNX NER, and custom rules."""

    def __init__(
        self,
        custom_detector: CustomEntityDetector | None = None,
        *,
        enable_ner: bool = False,
        ner_detector: NerDetectorProtocol | None = None,
    ) -> None:
        self.custom_detector = custom_detector
        if ner_detector is not None:
            self.ner_detector: NerDetectorProtocol | None = ner_detector
        elif enable_ner:
            self.ner_detector = OnnxPiiDetector()
        else:
            self.ner_detector = None

    def detect(self, text: str) -> list[EntitySpan]:
        """Run baseline regex patterns, optional ONNX NER, and any custom detector.

        Baseline regex spans take deterministic priority over NER/custom spans on
        overlapping or identical character ranges.
        """
        baseline_spans = detect_baseline_entities(text)
        extra_spans: list[EntitySpan] = []
        if self.ner_detector is not None:
            extra_spans.extend(self.ner_detector.detect(text))
        if self.custom_detector is not None:
            extra_spans.extend(self.custom_detector(text))
        return _merge_adjacent_person_spans(
            merge_entity_spans(baseline_spans, extra_spans, len(text)),
            text,
        )

    def create_table(self) -> EntityTable:
        """Create a new request-scoped EntityTable bound to this anonymizer's detector."""
        return EntityTable(detector=self.detect)

    def anonymize(
        self,
        text: str,
        entity_table: EntityTable | None = None,
    ) -> tuple[str, EntityTable]:
        """Anonymize text using the provided or a newly created EntityTable."""
        table = entity_table or self.create_table()
        return table.anonymize(text), table
