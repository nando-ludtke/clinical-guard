"""Role-based table-driven safety and routing policy."""

from dataclasses import dataclass

from clinical_guard.safety.content_safety import E5Action, E5GuardResult
from clinical_guard.safety.prompt_safety import PromptGuardResult
from clinical_guard.schemas import GuardDecision, RiskCategory, RoutingTarget, UserRole

_EMERGENCY_DISCLAIMER = (
    "CRITICAL SAFETY NOTICE: Automated clinical generation was bypassed due to a "
    "high-risk emergency signal. Seek immediate emergency medical attention."
)

_SECURITY_DISCLAIMER = (
    "SECURITY POLICY NOTICE: Request was blocked due to detected prompt-injection "
    "or instruction-override behavior."
)

_CLINICAL_DISCLAIMER = (
    "CLINICAL DECISION SUPPORT ONLY: Output is intended for licensed medical "
    "professionals and does not replace independent clinical judgment or local protocols."
)

_LAYMAN_DISCLAIMER = (
    "INFORMATIONAL GUIDANCE ONLY: This response is simplified for general education, "
    "is not a medical diagnosis, and does not replace consultation with a qualified "
    "healthcare professional."
)

_SELF_HARM_FALLBACK = (
    "Immediate support is available. If you or someone else is in immediate danger or "
    "experiencing thoughts of self-harm or suicide, call your local emergency number "
    "(e.g., 911 in the US, 112 in the EU, 999 in the UK) or go to the nearest emergency "
    "department. In the US, you can also call or text 988 to reach the Suicide & Crisis "
    "Lifeline 24/7."
)

_OVERDOSE_FALLBACK = (
    "Possible acute pharmacological overdose detected. Call emergency medical services "
    "immediately (e.g., 911 / 112 / 999) or contact Poison Control (1-800-222-1222 in the "
    "US). Do not wait for symptoms to worsen, and keep the medication container available "
    "for responders."
)

_TOXICOLOGY_FALLBACK = (
    "Possible acute toxicological emergency detected. Call emergency medical services "
    "(e.g., 911 / 112 / 999) or Poison Control (1-800-222-1222 in the US) immediately. "
    "Do not induce vomiting unless explicitly instructed by a poison specialist or "
    "emergency clinician."
)

_PROMPT_INJECTION_FALLBACK = (
    "Request rejected: untrusted instruction-override or prompt-injection attempt detected. "
    "Please submit a direct clinical question without system or role manipulation."
)

CATEGORY_FALLBACKS: dict[RiskCategory, str] = {
    "self_harm": _SELF_HARM_FALLBACK,
    "overdose": _OVERDOSE_FALLBACK,
    "toxicology": _TOXICOLOGY_FALLBACK,
    "prompt_injection": _PROMPT_INJECTION_FALLBACK,
    "none": _EMERGENCY_DISCLAIMER,
}


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Declarative routing specification for a (E5Action, UserRole) pair."""

    decision: GuardDecision
    routing_target: RoutingTarget
    disclaimer: str
    simplify_output: bool
    use_category_fallback: bool = False
    audit_flag: bool = False


@dataclass(frozen=True, slots=True)
class RoleProfile:
    """Default downstream routing profile for a UserRole."""

    decision: GuardDecision
    routing_target: RoutingTarget
    disclaimer: str
    simplify_output: bool
    emergency_categories: frozenset[RiskCategory]


ROLE_PROFILES: dict[UserRole, RoleProfile] = {
    "medical_professional": RoleProfile(
        decision="route_clinical",
        routing_target="advanced_clinical_reasoning",
        disclaimer=_CLINICAL_DISCLAIMER,
        simplify_output=False,
        emergency_categories=frozenset({"self_harm"}),
    ),
    "layman": RoleProfile(
        decision="route_layman",
        routing_target="simplified_layman_guidance",
        disclaimer=_LAYMAN_DISCLAIMER,
        simplify_output=True,
        emergency_categories=frozenset({"self_harm", "overdose", "toxicology"}),
    ),
}

POLICY_MATRIX: dict[tuple[E5Action, UserRole], PolicySpec] = {
    ("hard_block", "layman"): PolicySpec(
        decision="bypass_emergency",
        routing_target="emergency_escalation",
        disclaimer=_EMERGENCY_DISCLAIMER,
        simplify_output=False,
        use_category_fallback=True,
    ),
    ("hard_block", "medical_professional"): PolicySpec(
        decision="bypass_emergency",
        routing_target="emergency_escalation",
        disclaimer=_EMERGENCY_DISCLAIMER,
        simplify_output=False,
        use_category_fallback=True,
    ),
    ("soft_flag", "layman"): PolicySpec(
        decision="bypass_emergency",
        routing_target="emergency_escalation",
        disclaimer=_EMERGENCY_DISCLAIMER,
        simplify_output=False,
        use_category_fallback=True,
    ),
    ("soft_flag", "medical_professional"): PolicySpec(
        decision=ROLE_PROFILES["medical_professional"].decision,
        routing_target=ROLE_PROFILES["medical_professional"].routing_target,
        disclaimer=ROLE_PROFILES["medical_professional"].disclaimer,
        simplify_output=ROLE_PROFILES["medical_professional"].simplify_output,
        audit_flag=True,
    ),
    ("allow", "layman"): PolicySpec(
        decision=ROLE_PROFILES["layman"].decision,
        routing_target=ROLE_PROFILES["layman"].routing_target,
        disclaimer=ROLE_PROFILES["layman"].disclaimer,
        simplify_output=ROLE_PROFILES["layman"].simplify_output,
    ),
    ("allow", "medical_professional"): PolicySpec(
        decision=ROLE_PROFILES["medical_professional"].decision,
        routing_target=ROLE_PROFILES["medical_professional"].routing_target,
        disclaimer=ROLE_PROFILES["medical_professional"].disclaimer,
        simplify_output=ROLE_PROFILES["medical_professional"].simplify_output,
    ),
}


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Resolved routing outcome produced by the policy engine."""

    decision: GuardDecision
    risk_category: RiskCategory
    routing_target: RoutingTarget
    response_text: str
    disclaimer: str
    simplify_output: bool
    rule_id: str | None = None
    audit_flag: bool = False


def _security_block_decision(rule_id: str) -> RoutingDecision:
    """Construct a security-block RoutingDecision for prompt-injection detections."""
    return RoutingDecision(
        decision="block_security",
        risk_category="prompt_injection",
        routing_target="security_block",
        response_text=CATEGORY_FALLBACKS["prompt_injection"],
        disclaimer=_SECURITY_DISCLAIMER,
        simplify_output=False,
        rule_id=rule_id,
        audit_flag=True,
    )


def resolve_policy(
    user_role: UserRole,
    e5_result: E5GuardResult | None = None,
    prompt_result: PromptGuardResult | None = None,
) -> RoutingDecision:
    """Resolve prompt-injection short-circuit or table-driven E5 content policy."""
    if prompt_result is not None and prompt_result.is_injection:
        return _security_block_decision(f"PROMPT_GUARD_{prompt_result.label.upper()}")

    if e5_result is None:
        raise ValueError("e5_result is required when prompt_result is not an injection.")

    if e5_result.risk_category == "prompt_injection" and e5_result.action in {
        "hard_block",
        "soft_flag",
    }:
        return _security_block_decision(f"E5_{e5_result.action.upper()}_PROMPT_INJECTION")

    role_profile = ROLE_PROFILES[user_role]
    effective_action: E5Action = e5_result.action
    if (
        e5_result.action == "hard_block"
        and e5_result.risk_category not in role_profile.emergency_categories
    ):
        effective_action = "soft_flag"

    spec = POLICY_MATRIX[(effective_action, user_role)]
    if spec.use_category_fallback:
        response_text = CATEGORY_FALLBACKS.get(e5_result.risk_category, _EMERGENCY_DISCLAIMER)
        rule_id: str | None = f"E5_{e5_result.action.upper()}_{e5_result.risk_category.upper()}"
        risk_category: RiskCategory = e5_result.risk_category
    elif spec.audit_flag:
        response_text = (
            "Safety checks passed with clinical audit flag. Query cleared for advanced "
            "clinical reasoning pipeline."
        )
        rule_id = f"E5_{e5_result.action.upper()}_{e5_result.risk_category.upper()}"
        risk_category = e5_result.risk_category
    else:
        response_text = "Safety checks passed. Query cleared for downstream clinical pipeline."
        rule_id = None
        risk_category = "none"

    return RoutingDecision(
        decision=spec.decision,
        risk_category=risk_category,
        routing_target=spec.routing_target,
        response_text=response_text,
        disclaimer=spec.disclaimer,
        simplify_output=spec.simplify_output,
        rule_id=rule_id,
        audit_flag=spec.audit_flag,
    )
