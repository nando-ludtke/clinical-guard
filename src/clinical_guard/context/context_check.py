"""Provider-agnostic clinical context groundedness checker powered by litellm."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

import litellm
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from clinical_guard.privacy.anonymizer import EntityTable
from clinical_guard.schemas import ClaimCitation, ContextDocument, GroundednessResult

_PROMPTS_PATH = Path(__file__).with_name("prompts.yaml")


class PromptSpec(BaseModel):
    """Versioned system and user prompt specification."""

    model_config = ConfigDict(extra="forbid")

    date: str
    description: str
    system_prompt: str
    user_template: str


class PromptRegistry(BaseModel):
    """Registry of versioned groundedness prompts loaded from prompts.yaml."""

    model_config = ConfigDict(extra="forbid")

    active_version: str
    versions: dict[str, PromptSpec]

    @model_validator(mode="after")
    def _validate_active_version(self) -> PromptRegistry:
        if self.active_version not in self.versions:
            raise ValueError(
                f"active_version {self.active_version!r} is not defined in versions."
            )
        return self

    def get_version(self, version: str | None = None) -> tuple[str, PromptSpec]:
        """Return (version, PromptSpec) for the requested or active version."""
        resolved = version or self.active_version
        if resolved not in self.versions:
            available = ", ".join(sorted(self.versions))
            raise ValueError(
                f"Unknown prompt_version {resolved!r}. Available versions: {available}"
            )
        return resolved, self.versions[resolved]


@lru_cache(maxsize=4)
def load_prompt_registry(path: Path | None = None) -> PromptRegistry:
    """Load and validate the prompt registry from prompts.yaml."""
    target = path or _PROMPTS_PATH
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PromptRegistry.model_validate(raw)


def _strip_code_fences(raw: str) -> str:
    """Remove optional markdown JSON code fences around an LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def normalize_context_documents(
    contexts: Sequence[ContextDocument | str],
) -> list[ContextDocument]:
    """Normalize string or ContextDocument inputs into ContextDocument instances."""
    normalized: list[ContextDocument] = []
    for ctx in contexts:
        if isinstance(ctx, ContextDocument):
            normalized.append(ctx)
        else:
            normalized.append(ContextDocument(text=ctx))
    return normalized


def resolve_quote_offsets(
    quote: str | None,
    doc_text: str,
) -> tuple[int | None, int | None, bool]:
    """Resolve exact character offsets of quote in doc_text (with whitespace fallback)."""
    if not quote or not doc_text:
        return None, None, False

    idx = doc_text.find(quote)
    if idx != -1:
        return idx, idx + len(quote), True

    tokens = quote.split()
    if not tokens:
        return None, None, False

    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens))
    match = pattern.search(doc_text)
    if match is not None:
        return match.start(), match.end(), True

    return None, None, False


def _format_contexts(
    contexts: Sequence[ContextDocument],
    prompt_version: str,
) -> str:
    """Format reference documents for the selected prompt version."""
    if not contexts:
        return "[No reference contexts provided]"
    if prompt_version == "v1.0.0":
        return "\n\n".join(
            f"[Context {idx + 1}]\n{doc.text}" for idx, doc in enumerate(contexts)
        )
    return "\n\n".join(
        f"[Document {doc.doc_id}]\n{doc.text}" for doc in contexts
    )


def _format_conversation_history(
    conversation_history: Sequence[dict[str, str]] | None,
) -> str:
    """Format conversation turns for the user prompt."""
    if not conversation_history:
        return "[No conversation history provided]"
    return "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in conversation_history
    )


class ContextGroundednessChecker:
    """Provider-agnostic LLM-as-a-judge for clinical context groundedness."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        prompt_version: str | None = None,
    ) -> None:
        registry = load_prompt_registry()
        resolved_version, prompt_spec = registry.get_version(prompt_version)
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.timeout = timeout
        self.prompt_version = resolved_version
        self._prompt_spec = prompt_spec

    def _render_user_prompt(
        self,
        query: str,
        response_text: str,
        contexts: Sequence[ContextDocument],
        conversation_history: Sequence[dict[str, str]] | None = None,
    ) -> str:
        """Render the versioned user prompt template using string.Template."""
        template = Template(self._prompt_spec.user_template.strip())
        return template.substitute(
            query=query,
            conversation_history=_format_conversation_history(conversation_history),
            contexts=_format_contexts(contexts, self.prompt_version),
            response_text=response_text,
        )

    async def check(
        self,
        query: str,
        response_text: str,
        contexts: Sequence[ContextDocument | str],
        conversation_history: Sequence[dict[str, str]] | None = None,
        entity_table: EntityTable | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> GroundednessResult:
        """Evaluate whether response_text is grounded in contexts via litellm.acompletion."""
        active_model = model or self.model
        active_api_key = api_key if api_key is not None else self.api_key

        normalized_docs = normalize_context_documents(contexts)
        original_docs_by_id = {doc.doc_id: doc.text for doc in normalized_docs}

        prompt_query = query
        prompt_response = response_text
        prompt_docs = normalized_docs
        prompt_history = list(conversation_history) if conversation_history else None

        if entity_table is not None:
            prompt_query = entity_table.anonymize(query)
            prompt_response = entity_table.anonymize(response_text)
            prompt_docs = [
                ContextDocument(
                    doc_id=doc.doc_id,
                    text=entity_table.anonymize(doc.text),
                )
                for doc in normalized_docs
            ]
            if prompt_history is not None:
                prompt_history = [
                    {
                        **msg,
                        "content": entity_table.anonymize(msg.get("content", "")),
                    }
                    for msg in prompt_history
                ]

        completion_kwargs: dict[str, Any] = {
            "model": active_model,
            "messages": [
                {"role": "system", "content": self._prompt_spec.system_prompt.strip()},
                {
                    "role": "user",
                    "content": self._render_user_prompt(
                        query=prompt_query,
                        response_text=prompt_response,
                        contexts=prompt_docs,
                        conversation_history=prompt_history,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if active_api_key is not None:
            completion_kwargs["api_key"] = active_api_key
        if self.api_base is not None:
            completion_kwargs["api_base"] = self.api_base

        t0 = time.perf_counter()
        completion = await litellm.acompletion(**completion_kwargs)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        raw_content = completion.choices[0].message.content
        if not raw_content:
            raise ValueError("Groundedness judge returned empty response content.")

        parsed = json.loads(_strip_code_fences(raw_content))
        if not isinstance(parsed, dict):
            raise ValueError("Groundedness judge response must be a JSON object.")

        def _uncover(value: str) -> str:
            return entity_table.deanonymize(value) if entity_table is not None else value

        unsupported_claims = [_uncover(str(c)) for c in parsed.get("unsupported_claims", [])]
        contradictions = [_uncover(str(c)) for c in parsed.get("contradictions", [])]
        reasoning = _uncover(str(parsed["reasoning"]))

        claims: list[ClaimCitation] = []
        for raw_claim in parsed.get("claims", []):
            claim_text = _uncover(str(raw_claim["claim"]))
            raw_quote = raw_claim.get("quote")
            uncovered_quote = _uncover(str(raw_quote)) if raw_quote is not None else None
            source_type = raw_claim["source_type"]
            doc_id = raw_claim.get("doc_id")

            start_char: int | None = None
            end_char: int | None = None
            quote_verified = False
            if (
                source_type == "context_doc"
                and doc_id is not None
                and doc_id in original_docs_by_id
                and uncovered_quote is not None
            ):
                start_char, end_char, quote_verified = resolve_quote_offsets(
                    uncovered_quote,
                    original_docs_by_id[doc_id],
                )

            claims.append(
                ClaimCitation(
                    claim=claim_text,
                    status=raw_claim["status"],
                    source_type=source_type,
                    doc_id=doc_id,
                    quote=uncovered_quote,
                    start_char=start_char,
                    end_char=end_char,
                    quote_verified=quote_verified,
                )
            )

        return GroundednessResult.model_validate(
            {
                "grounded": parsed["grounded"],
                "unsupported_claims": unsupported_claims,
                "contradictions": contradictions,
                "claims": claims,
                "reasoning": reasoning,
                "model": active_model,
                "latency_ms": latency_ms,
                "prompt_version": self.prompt_version,
            }
        )
