# Context Groundedness Check — Architecture & Implementation

**Scope:** `src/clinical_guard/context/context_check.py` (`ContextGroundednessChecker`, `PromptRegistry`, `resolve_quote_offsets`), `src/clinical_guard/context/prompts.yaml` (`v1.1.0`), `src/clinical_guard/pipelines/context_pipeline.py` (`run_context_pipeline`, `ContextPipelineResult`), `src/clinical_guard/privacy/anonymizer.py` (`LocalAnonymizer`, `EntityTable`, `OnnxPiiDetector`), `src/clinical_guard/schemas.py` (`ContextDocument`, `ClaimCitation`, `GroundednessResult`, `ContextEvaluateRequest`, `ContextEvaluateResponse`), and `src/clinical_guard/main.py` (`POST /v1/guard/evaluate_context`).

---

## 1. Rationale

Clinical decision support and patient-facing guidance must not hallucinate dosing numbers, contraindications, or recommendations beyond the retrieved clinical evidence. `ContextGroundednessChecker` verifies that a candidate response is strictly faithful to provided reference documents while:
1. **Protecting patient privacy (PHI/PII):** Sensitive identifiers must be anonymized **locally** with consistent placeholders before any text leaves the trust boundary to a third-party LLM provider.
2. **Preventing indirect prompt injection:** Reference documents (`ContextDocument`) are screened by `PromptInjectionGuard` before reaching the judge LLM.
3. **Supporting valid clinical arithmetic:** Real clinical responses frequently compute weight-based doses (e.g., `20 mg/kg/day * 15 kg = 300 mg/day`) or unit conversions from formulas stated in the context; the judge explicitly allows accurate arithmetic derivations while flagging wrong calculations as contradictions.
4. **Remaining provider-agnostic:** Powered by `litellm.acompletion`, supporting OpenAI, Anthropic, Azure, or local OpenAI-compatible endpoints via BYOK (`api_key` parameter) or standard environment variables.

---

## 2. Current Implementation (`context_check.py` & `prompts.yaml` `v1.1.0`)

Implemented in `src/clinical_guard/context/context_check.py` and `src/clinical_guard/schemas.py`:
- **`ClaimCitation`** (Pydantic `BaseModel`, `extra="forbid"`):
  - `claim: str`
  - `status: Literal["grounded", "ungrounded", "contradicted"]`
  - `source_type: Literal["context_doc", "conversation_history", "none"]`
  - `doc_id: str | None = None`
  - `quote: str | None = None`
  - `start_char: int | None = None`
  - `end_char: int | None = None`
  - `quote_verified: bool = False`
- **`GroundednessResult`** (Pydantic `BaseModel`, `extra="forbid"`):
  - `grounded: bool`
  - `unsupported_claims: list[str]`
  - `contradictions: list[str]`
  - `claims: list[ClaimCitation]`
  - `reasoning: str`
  - `model: str`
  - `latency_ms: float`
  - `prompt_version: str = "v1.1.0"`
- **`ContextGroundednessChecker`**:
  - Loads versioned prompts from `src/clinical_guard/context/prompts.yaml` (`active_version: "v1.1.0"`, `PromptRegistry`).
  - Configured with `model` (default `"openai/gpt-4o-mini"`), `api_key`, `api_base`, `temperature=0.0`, `timeout=30.0`, and `prompt_version` (default `None` -> active `v1.1.0`).
  - `async def check(query, response_text, contexts, conversation_history=None, entity_table=None, api_key=None, model=None, prompt_version=None) -> GroundednessResult`:
    - Normalizes `contexts` into `ContextDocument(doc_id, text)` and formats `[Document <doc_id>] ...` blocks alongside `query`, `conversation_history`, and `response_text` (anonymized via `entity_table` when supplied).
    - Enforces structured JSON output (`response_format={"type": "json_object"}`), strips optional markdown code fences, and validates against `GroundednessResult`.
    - Verifies verbatim quotes against the original or anonymized document text via `resolve_quote_offsets()` (exact match + whitespace-normalized fallback), populating `start_char`, `end_char`, and `quote_verified`.
    - Uncovers entity placeholders (`entity_table.deanonymize(...)`) in `unsupported_claims`, `contradictions`, `reasoning`, and each `ClaimCitation` (`claim`, `quote`) before returning.
    - **BYOK & env key handling:** Passes `api_key` to `litellm.acompletion` only when explicitly provided (via `X-OpenAI-Api-Key` header, per-call `api_key`, or instance-level); otherwise omits `api_key` so `litellm` resolves provider keys from environment variables (`OPENAI_API_KEY`, etc.).
    - **Arithmetic / `mg/kg` derivation support:** `prompts.yaml` explicitly permits valid arithmetic derivations when the underlying formula, rate, or unit rule is present in the contexts, and flags incorrect arithmetic in `contradictions`.

---

## 3. End-to-End Context Pipeline & API Integration

### 3.1 `ContextDocument(doc_id, text)` Ingestion & Endpoint (`main.py`)
`POST /v1/guard/evaluate_context` in `src/clinical_guard/main.py` accepts `ContextEvaluateRequest(query, response_text, contexts, conversation_history, session_id)` and an optional `X-OpenAI-Api-Key` header, returning `ContextEvaluateResponse(security_blocked, routing_target, flagged_doc_ids, groundedness, latency_ms, session_id)`.
- `ContextDocument` accepts `doc_id: str` (defaulting to a UUID4 string) and `text: str`.
- Contexts are tagged with `[Document <doc_id>]` in the judge prompt so unsupported claims and contradictions trace back to specific source documents.

### 3.2 Parallel `PromptInjectionGuard` & Local NER Anonymization (`context_pipeline.py`)
Implemented in `run_context_pipeline()` (`src/clinical_guard/pipelines/context_pipeline.py`) using `anyio.create_task_group()`:
1. **`PromptInjectionGuard` Screening (Injection Gate):** Evaluates `query`, `response_text`, `conversation_history` messages, and all `ContextDocument.text` entries concurrently for direct overrides, jailbreaks, delimiter smuggling, and indirect context injections.
2. **Local NER/PII Anonymization (`src/clinical_guard/privacy/anonymizer.py`):** Runs concurrently with `PromptInjectionGuard` via `LocalAnonymizer` and `EntityTable` because local NER/regex extraction is **non-generative** and unaffected by prompt-injection payloads. In the 4-container Docker deployment (`docker-compose.yml`), NER inference runs in the dedicated `ner-pii-guard` container (`OnnxPiiDetector` / `models/multilang-pii-ner-ONNX`), called via `HttpNerDetector`.

**Strict Injection Gate:** If **any** input message or `ContextDocument` is flagged (`is_injection = true`), the external `litellm` Groundedness API call is immediately blocked and the pipeline returns `security_blocked = true`, `routing_target = "security_block"`, and `flagged_doc_ids` (without logging raw text).

### 3.3 Request-Scoped Shared Entity Table (`<PERSON_1>`, `<ORG_1>`, `<DATE_1>`)
To preserve referential integrity across documents and conversation turns without leaking PHI/PII to third-party LLM providers:
- Implemented in `src/clinical_guard/privacy/anonymizer.py` (`LocalAnonymizer`, `EntityTable`, `EntitySpan`, `OnnxPiiDetector`): combines deterministic regexes (emails, MRNs/patient IDs, general IDs, phones, dates, titled names) with `OnnxPiiDetector` (`models/multilang-pii-ner-ONNX`) or a custom detector hook (including `HttpNerDetector` for `ner-pii-guard`) to populate a **request-scoped `EntityTable`**.
- This single **`EntityTable`** is shared across `query`, `conversation_history`, `response_text`, and all `ContextDocument`s for the request, ensuring the same normalized entity receives the **exact same placeholder** everywhere before leaving the local boundary (e.g., `<PERSON_1>` in `doc_1` matches `<PERSON_1>` in `conversation_history`), and is uncovered locally on the returned `GroundednessResult`.
- The `EntityTable` is discarded as soon as the request completes; raw entities are never logged or persisted.

### 3.4 Evaluation Dataset
A 16-row bilingual doctor's-letter evaluation set is maintained at `data/eval/context_check_eval.jsonl` (8 `en`, 8 `de`), covering `grounded_letter`, `grounded_arithmetic`, `grounded_history`, `ungrounded_claim`, `contradicted_claim`, and `indirect_injection`.

---

## Benchmark Results

Evaluated against `data/eval/context_check_eval.jsonl` (`16` rows: `8` EN / `8` DE) and recorded in `data/eval/context_check_benchmark.json` using the full 3-stage pipeline (`models/Llama-Prompt-Guard-2-86M` + `models/multilang-pii-ner-ONNX` + `openai/gpt-4o-mini`, `prompts.yaml` `v1.1.0`):

| Metric | Overall (`16` rows) | EN (`8` rows) | DE (`8` rows) |
| --- | --- | --- | --- |
| **Grounded Accuracy** | **100.0%** | 100.0% | 100.0% |
| **Security Block Accuracy** | **100.0%** | 100.0% | 100.0% |
| **Flagged `doc_id` Exact Match** | **100.0%** | 100.0% | 100.0% |
| **Source `doc_id` Recall** | **100.0%** | 100.0% | 100.0% |
| **Quote Verification (`quote_verified_rate`)** | **95.0%** (`38/40`) | 94.7% (`18/19`) | 95.2% (`20/21`) |

- **End-to-End Pipeline Latency:** `5,058.8 ms` mean / `5,280.9 ms` p50 / `9,700.0 ms` p95 (and max).
- **Case-Type Coverage:** `grounded_letter` (4), `contradicted_claim` (4), `grounded_arithmetic` (2), `grounded_history` (2), `ungrounded_claim` (2), and `indirect_injection` (2) all achieved `100.0%` groundedness / security-block accuracy.
