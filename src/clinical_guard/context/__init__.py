"""Context evaluation, anonymization, and groundedness checking modules."""

from clinical_guard.context.context_check import (
    ContextGroundednessChecker,
    normalize_context_documents,
    resolve_quote_offsets,
)
from clinical_guard.privacy.anonymizer import EntitySpan, EntityTable, LocalAnonymizer
from clinical_guard.schemas import ClaimCitation, ContextDocument, GroundednessResult

__all__ = [
    "ClaimCitation",
    "ContextDocument",
    "ContextGroundednessChecker",
    "EntitySpan",
    "EntityTable",
    "GroundednessResult",
    "LocalAnonymizer",
    "normalize_context_documents",
    "resolve_quote_offsets",
]
