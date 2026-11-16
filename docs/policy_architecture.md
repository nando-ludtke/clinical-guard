# Policy & Decision Routing Architecture

**Scope:** `src/clinical_guard/routing/policy.py` (`PolicySpec`, `RoleProfile`, `POLICY_MATRIX`, `ROLE_PROFILES`, `CATEGORY_FALLBACKS`, `resolve_policy`, `RoutingDecision`), `src/clinical_guard/pipelines/safety_pipeline.py` (`run_safety_pipeline`), and `src/clinical_guard/schemas.py` (`GuardResponse`, `GuardDiagnostics`).

---

## 1. Policy Decision Matrix

The policy engine maps `(E5Action, UserRole)` pairs to a deterministic `(GuardDecision, RoutingTarget, Behavior)` outcome via table-driven dictionary lookup. Prompt-injection detections from `PromptInjectionGuard` short-circuit immediately before `POLICY_MATRIX` lookup.

| Guard Signal | `UserRole` | `GuardDecision` | `RoutingTarget` | Behavior |
| --- | --- | --- | --- | --- |
| `prompt_injection` (`PromptInjectionGuard.is_injection = true`) | Any (`layman` / `medical_professional`) | `block_security` | `security_block` | Short-circuit before `E5SafetyGuard`; return `CATEGORY_FALLBACKS["prompt_injection"]` + `_SECURITY_DISCLAIMER`; `simplify_output = false`; `audit_flag = true`. |
| `prompt_injection` (`E5SafetyGuard.risk_category == "prompt_injection"`, `action in {"hard_block", "soft_flag"}`) | Any (`layman` / `medical_professional`) | `block_security` | `security_block` | Catch roleplay/persona-override jailbreaks via E5 semantic anchors; return `CATEGORY_FALLBACKS["prompt_injection"]` + `_SECURITY_DISCLAIMER`; `rule_id = "E5_<ACTION>_PROMPT_INJECTION"`; `simplify_output = false`; `audit_flag = true`. |
| `hard_block` (`risk_category` in `emergency_categories`) | `layman` (`self_harm`, `overdose`, `toxicology`) | `bypass_emergency` | `emergency_escalation` | Bypass LLM generation immediately; return `CATEGORY_FALLBACKS[risk_category]` + `_EMERGENCY_DISCLAIMER`; `simplify_output = false`. |
| `hard_block` (`risk_category` in `emergency_categories`) | `medical_professional` (`self_harm` only) | `bypass_emergency` | `emergency_escalation` | Bypass LLM generation immediately; return `CATEGORY_FALLBACKS["self_harm"]` + `_EMERGENCY_DISCLAIMER`; `simplify_output = false`. |
| `hard_block` (`risk_category` not in `emergency_categories`) | `medical_professional` (`overdose`, `toxicology`) | `route_clinical` | `advanced_clinical_reasoning` | Downgrade `effective_action` to `soft_flag` before `POLICY_MATRIX` lookup; route clinician query to `advanced_clinical_reasoning` with `audit_flag = true`, preserved `risk_category`, `rule_id = "E5_HARD_BLOCK_<CATEGORY>"`, `_CLINICAL_DISCLAIMER`, and `simplify_output = false`. |
| `soft_flag` | `layman` | `bypass_emergency` | `emergency_escalation` | **Fail closed for non-clinicians:** bypass LLM generation and return `CATEGORY_FALLBACKS[risk_category]` + `_EMERGENCY_DISCLAIMER`; `simplify_output = false`. |
| `soft_flag` | `medical_professional` | `route_clinical` | `advanced_clinical_reasoning` | **Clinical audit pass-through:** allow clinician query into `advanced_clinical_reasoning` with `audit_flag = true`, preserved `risk_category`, `_CLINICAL_DISCLAIMER`, and `simplify_output = false`. |
| `allow` | `layman` | `route_layman` | `simplified_layman_guidance` | Route to `simplified_layman_guidance` with `_LAYMAN_DISCLAIMER` and `simplify_output = true`. |
| `allow` | `medical_professional` | `route_clinical` | `advanced_clinical_reasoning` | Route to `advanced_clinical_reasoning` with `_CLINICAL_DISCLAIMER` and `simplify_output = false`. |

### `soft_flag` vs. `hard_block` & `emergency_categories` Rationale
- **`hard_block` (`margin >= 0.02`):** High-confidence acute crisis (`self_harm`, `overdose`, `toxicology`). `resolve_policy()` checks `RoleProfile.emergency_categories` before `POLICY_MATRIX` lookup:
  - **`layman` (`emergency_categories = frozenset({"self_harm", "overdose", "toxicology"})`):** All three categories trigger immediate `emergency_escalation` with category-specific crisis resources (e.g., 988 Lifeline, Poison Control).
  - **`medical_professional` (`emergency_categories = frozenset({"self_harm"})`):** Only `self_harm` triggers `emergency_escalation`. Clinical `overdose` and `toxicology` queries (e.g., NAC dosing, ethylene glycol management) are routed as `effective_action = "soft_flag"` to `advanced_clinical_reasoning` with `audit_flag = true` so clinicians are not blocked from acute toxicology/overdose decision support.
- **`soft_flag` (`0.0 <= margin < 0.02`):** Borderline semantic band where clinical queries discussing lethality or dosing sit near the benign boundary.
  - **`layman`:** Fails safe to `bypass_emergency` (`emergency_escalation`).
  - **`medical_professional`:** Routed to `advanced_clinical_reasoning` with `audit_flag = true` so legitimate clinicians are not blocked while retaining full auditability.

---

## 2. Table-Driven Dictionary Dispatch

Rather than growing nested `if / elif / else` chains as new `UserRole` values (e.g., `pharmacist`, `triage_nurse`) or `RiskCategory` values (e.g., `pediatric_dosing`, `anaphylaxis`) are introduced, `src/clinical_guard/routing/policy.py` uses **O(1) dictionary dispatch**.

### 2.1 Declarative Structures (`PolicySpec` & `RoleProfile`)
- **`RoleProfile`**: Captures the default downstream routing profile (`decision`, `routing_target`, `disclaimer`, `simplify_output`) and `emergency_categories: frozenset[RiskCategory]` for each `UserRole` in `ROLE_PROFILES`.
- **`PolicySpec`**: Captures the policy outcome for a `(E5Action, UserRole)` key in `POLICY_MATRIX`, including `use_category_fallback` and `audit_flag`.

### 2.2 Lookup Tables
- **`CATEGORY_FALLBACKS: dict[RiskCategory, str]`**: Maps `"self_harm"`, `"overdose"`, `"toxicology"`, `"prompt_injection"`, and `"none"` to deterministic, medically appropriate fallback messages.
- **`ROLE_PROFILES: dict[UserRole, RoleProfile]`**: Defines default routing and `emergency_categories` for `"medical_professional"` (`frozenset({"self_harm"})`) and `"layman"` (`frozenset({"self_harm", "overdose", "toxicology"})`).
- **`POLICY_MATRIX: dict[tuple[E5Action, UserRole], PolicySpec]`**: Maps all `3 x 2 = 6` `(E5Action, UserRole)` combinations declaratively.

### 2.3 Resolution (`resolve_policy`)
`resolve_policy(user_role, e5_result=None, prompt_result=None) -> RoutingDecision`:
1. If `prompt_result is not None and prompt_result.is_injection`, immediately return `block_security` (`security_block`) with `CATEGORY_FALLBACKS["prompt_injection"]`.
2. If `e5_result.risk_category == "prompt_injection"` and `e5_result.action in {"hard_block", "soft_flag"}`, immediately return `block_security` (`security_block`) with `CATEGORY_FALLBACKS["prompt_injection"]` and `rule_id = f"E5_{e5_result.action.upper()}_PROMPT_INJECTION"`.
3. Check `role_profile = ROLE_PROFILES[user_role]`. When `e5_result.action == "hard_block"` and `e5_result.risk_category not in role_profile.emergency_categories`, set `effective_action = "soft_flag"`; otherwise keep `effective_action = e5_result.action`.
4. Look up `spec = POLICY_MATRIX[(effective_action, user_role)]` in O(1) time.
5. Select `CATEGORY_FALLBACKS[e5_result.risk_category]` when `spec.use_category_fallback` is `True`, the audit-flagged clinical clearance message when `spec.audit_flag` is `True` (preserving `rule_id = f"E5_{e5_result.action.upper()}_{e5_result.risk_category.upper()}"`), or the standard cleared downstream message otherwise.

---

## 3. Extensibility & Exhaustiveness

- **Adding a new `RiskCategory`:** Add the literal to `RiskCategory` in `src/clinical_guard/schemas.py` and one entry in `CATEGORY_FALLBACKS` — zero changes to control flow.
- **Adding a new `UserRole`:** Add the literal to `UserRole`, add a `RoleProfile` in `ROLE_PROFILES`, and populate its `(E5Action, UserRole)` entries in `POLICY_MATRIX`.
- **CI Exhaustiveness Guard:** `tests/test_guard.py::test_policy_matrix_exhaustive` asserts `set(POLICY_MATRIX.keys()) == set(itertools.product(get_args(E5Action), get_args(UserRole)))` so any missing matrix row fails tests immediately.
