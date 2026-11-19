"""Strict Pydantic v2 request, response, and domain schemas."""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UserRole = Literal["layman", "medical_professional"]
RiskCategory = Literal["none", "self_harm", "overdose", "toxicology", "prompt_injection"]
GuardDecision = Literal[
    "bypass_emergency",
    "block_security",
    "route_clinical",
    "route_layman",
]
RoutingTarget = Literal[
    "emergency_escalation",
    "security_block",
    "advanced_clinical_reasoning",
    "simplified_layman_guidance",
]


class GuardRequest(BaseModel):
    """Incoming evaluation request payload."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=8000, description="User clinical query.")
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Opaque client session identifier for audit correlation.",
    )


class GuardDiagnostics(BaseModel):
    """Detailed System 1 model diagnostics for UI and observability."""

    model_config = ConfigDict(extra="forbid")

    prompt_guard_score: float | None = None
    prompt_guard_label: str | None = None
    prompt_guard_is_injection: bool | None = None
    prompt_guard_latency_ms: float | None = None
    e5_top_category: str | None = None
    e5_top_score: float | None = None
    e5_benign_score: float | None = None
    e5_margin: float | None = None
    e5_latency_ms: float | None = None


class GuardResponse(BaseModel):
    """Structured safety and routing evaluation response."""

    model_config = ConfigDict(extra="forbid")

    decision: GuardDecision
    risk_category: RiskCategory
    routing_target: RoutingTarget
    response_text: str
    disclaimer: str
    simplify_output: bool = False
    rule_id: str | None = None
    latency_ms: float = Field(..., ge=0.0)
    session_id: str
    user_role: UserRole
    diagnostics: GuardDiagnostics | None = None


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    service: str
    version: str



ClaimStatus = Literal["grounded", "ungrounded", "contradicted"]
ClaimSourceType = Literal["context_doc", "conversation_history", "none"]
ContextRoutingTarget = Literal["context_grounded", "context_ungrounded", "security_block"]


class ContextDocument(BaseModel):
    """Reference document supplied to context groundedness evaluation."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    text: str


class ClaimCitation(BaseModel):
    """Atomic factual claim with source citation and verified character offsets."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    status: ClaimStatus
    source_type: ClaimSourceType
    doc_id: str | None = None
    quote: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    quote_verified: bool = False


class GroundednessResult(BaseModel):
    """Structured result from the context groundedness evaluation."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    claims: list[ClaimCitation] = Field(default_factory=list)
    reasoning: str
    model: str
    latency_ms: float = Field(..., ge=0.0)
    prompt_version: str = "v1.1.0"


class ContextEvaluateRequest(BaseModel):
    """Incoming context-groundedness evaluation request payload."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=8000, description="User clinical query.")
    response_text: str = Field(
        ...,
        min_length=1,
        max_length=16000,
        description="Candidate clinical response to verify against contexts.",
    )
    contexts: list[ContextDocument | str] = Field(default_factory=list)
    conversation_history: list[dict[str, str]] | None = None
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Opaque client session identifier for audit correlation.",
    )


class ContextEvaluateResponse(BaseModel):
    """Structured context-pipeline evaluation response with injection gate and citations."""

    model_config = ConfigDict(extra="forbid")

    security_blocked: bool
    routing_target: ContextRoutingTarget
    flagged_doc_ids: list[str] = Field(default_factory=list)
    groundedness: GroundednessResult | None = None
    latency_ms: float = Field(..., ge=0.0)
    session_id: str


PromptBlockedBy = Literal["none", "prompt_guard", "e5_prompt_injection"]


class PromptCheckResponse(BaseModel):
    """Structured prompt-injection check response (PromptInjectionGuard + E5 defense-in-depth)."""

    model_config = ConfigDict(extra="forbid")

    is_injection: bool
    blocked_by: PromptBlockedBy
    rule_id: str | None = None
    prompt_guard_score: float
    prompt_guard_label: str
    e5_margin: float
    latency_ms: float = Field(..., ge=0.0)
    session_id: str
