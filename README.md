# clinical-guard

> **Disclaimer:** Engineering & research demonstration; not a certified medical device / SaMD.

Built at the intersection of clinical medicine and backend engineering, `clinical-guard` addresses common failure modes of generic LLM guardrails in healthcare: blocking legitimate clinician queries (e.g., overdose antidotes), over-redacting clinical parameters like `"72 kg"` as PII, missing multilingual crisis signals (EN/DE), and leaking raw PHI to third-party APIs. It provides a **local-first, CPU-friendly FastAPI microservice** combining role-aware semantic safety routing (`layman` vs. `medical_professional`), domain-preserving local PHI/PII anonymization, and citation-verified context groundedness.

## Capabilities & Endpoints

- **`POST /v1/guard/evaluate`**: Runs `PromptInjectionGuard` (`Llama-Prompt-Guard-2-86M` + delimiter smuggling) and `E5SafetyGuard` (`intfloat/multilingual-e5-small`) concurrently, then dispatches via `POLICY_MATRIX` (`emergency_escalation`, `security_block`, `advanced_clinical_reasoning`, `simplified_layman_guidance`).
- **`POST /v1/guard/prompt`**: Lightweight prompt-injection screening endpoint reusing `PromptInjectionGuard` + `E5SafetyGuard` defense-in-depth without clinical role headers or routing overhead.
- **`POST /v1/guard/evaluate_context`**: Runs `PromptInjectionGuard` and `LocalAnonymizer` (`OnnxPiiDetector` + regexes + shared `EntityTable`) in parallel, blocks injected documents (`security_block`), and checks candidate responses against `ContextDocument`s via `litellm` (`prompts.yaml` `v1.1.0`) with verified `ClaimCitation` offsets.
- **`GET /` & `GET /v1/benchmarks`**: Interactive Showcase UI and normalized evaluation suites (`data/eval/`).
- **Abuse Protection (`rate_limit.py`)**: Per-IP sliding-window rate limiting (`429`) and `asyncio.Semaphore` concurrency bounding (`503`) on all `/v1/guard/*` endpoints.

## Quickstart

### 1. Local Development

```bash
.venv/bin/uvicorn clinical_guard.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` for the Showcase UI, or `http://localhost:8000/docs` for OpenAPI docs.

### 2. Single-Container Docker (Recommended for 4 GB VPS / Showcase)

Runs all models in-process with local NER enabled (`CLINICAL_GUARD_ENABLE_LOCAL_NER=true`) and binds `127.0.0.1:8080` (container port `8000`):

```bash
docker compose -f docker-compose.single.yml up --build
```

### 3. 4-Container Docker Compose (Isolated Workers)

Runs `clinical-guard-api` (`:8080` -> `:8000`), `e5-guard` (`:8001`), `prompt-guard` (`:8002`), and `ner-pii-guard` (`:8003`):

```bash
docker compose up --build
```

Environment variables (`.env` or shell):
- `HF_TOKEN`: Required on first boot (when `models/` is empty) to download the gated `meta-llama/Llama-Prompt-Guard-2-86M` model.
- `OPENAI_API_KEY`: Optional default key for `ContextGroundednessChecker`; visitors can also supply a per-request key via `X-OpenAI-Api-Key` (BYOK).

### 4. Tests & Linting

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests scripts
```

## Documentation

- [Overall System Architecture](docs/overall_architecture.md)
- [Prompt Safety Guard](docs/prompt_safety.md)
- [Content Safety Guard](docs/content_safety.md)
- [Policy & Decision Routing Architecture](docs/policy_architecture.md)
- [Context Groundedness Check](docs/context_check.md)
- [Local NER & PII Anonymization](docs/privacy_anonymizer.md)
- [Docker Deployment & Configuration](docs/deployment.md)

## License

Licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International) — see [LICENSE](LICENSE). Non-commercial research and portfolio use only; contact the author for commercial licensing.
