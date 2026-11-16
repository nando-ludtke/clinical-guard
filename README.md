# clinical-guard

> **Disclaimer:** Engineering & research demonstration; not a certified medical device / SaMD.

Built from the intersection of clinical medicine and backend engineering, `clinical-guard` addresses a recurring failure mode of off-the-shelf LLM guardrails in healthcare: generic safety filters either **block legitimate clinician queries** (e.g., overdose antidotes or toxicology dosing), **over-redact medically critical parameters** like `"72 kg"` or patient age as PII, **miss multilingual or colloquial crisis signals** (such as German/slang self-harm expressions), or **leak raw PHI to third-party APIs** before screening. `clinical-guard` solves this with a **local-first, CPU-friendly FastAPI microservice stack** that pairs role-aware semantic safety routing (`layman` vs. `medical_professional`), domain-preserving local PHI/PII anonymization, and citation-verified context groundedness before any external LLM call.

Deterministic pre-generation safety guard, role-based clinical routing, local PHI/PII anonymization, and citation-aware context groundedness microservice built with FastAPI and Pydantic v2.

## Core Capabilities

- **Prompt Safety (`PromptInjectionGuard`)**: CPU-based prompt-injection and jailbreak screening using `Llama-Prompt-Guard-2-86M` (`models/Llama-Prompt-Guard-2-86M`) plus deterministic delimiter-smuggling detection (`_DELIMITER_SMUGGLING_PATTERN`) and E5 roleplay-jailbreak defense-in-depth.
- **Content Safety (`E5SafetyGuard`)**: Multilingual (EN/DE) semantic crisis classification using `intfloat/multilingual-e5-small` (`models/multilingual-e5-small`) with per-category `hard_block` margins across `self_harm`, `overdose`, `toxicology`, and `prompt_injection`.
- **Role-Based Policy (`POLICY_MATRIX` / `resolve_policy`)**: O(1) table-driven dispatch mapping `(E5Action, UserRole)` (`layman` vs. `medical_professional`) to deterministic routing targets (`emergency_escalation`, `security_block`, `advanced_clinical_reasoning`, `simplified_layman_guidance`) with structured `GuardDiagnostics`.
- **Local NER/PII Anonymization (`LocalAnonymizer` / `EntityTable`)**: Request-scoped entity anonymization combining deterministic regexes with `OnnxPiiDetector` (`models/multilang-pii-ner-ONNX`) so sensitive entities receive consistent placeholders (`<PERSON_1>`, `<ORG_1>`, `<DATE_1>`, etc.) across queries, conversation history, and reference documents before any external LLM call.
- **Context Groundedness (`ContextGroundednessChecker` & `run_context_pipeline`)**: Runs `PromptInjectionGuard` and `LocalAnonymizer` in parallel via `anyio.create_task_group()`, blocks injected documents (`security_block` with `flagged_doc_ids`), and evaluates candidate responses against `ContextDocument`s and conversation history via `litellm.acompletion` (`prompts.yaml` `v1.1.0`) with `ClaimCitation` quote-offset verification and local entity uncovering.
- **Showcase UI & Benchmarks**: Interactive HTML showcase served at `GET /` (`src/clinical_guard/static/index.html`) and normalized evaluation summaries at `GET /v1/benchmarks`.

## Repository Structure

```text
clinical-guard/
├── src/clinical_guard/
│   ├── main.py                  # FastAPI app, lifespan, /v1/guard/*, /v1/benchmarks, /
│   ├── config.py                # Pydantic Settings (CLINICAL_GUARD_* env vars)
│   ├── schemas.py               # Strict Pydantic v2 request/response/citation schemas
│   ├── benchmarks.py            # Benchmark suite loader for /v1/benchmarks
│   ├── logging_config.py        # Structured JSON logging & query metadata redaction
│   ├── safety/                  # PromptInjectionGuard & E5SafetyGuard
│   ├── privacy/                 # LocalAnonymizer, EntityTable, OnnxPiiDetector
│   ├── context/                 # ContextGroundednessChecker & prompts.yaml (v1.1.0)
│   ├── routing/                 # POLICY_MATRIX, ROLE_PROFILES, resolve_policy
│   ├── pipelines/               # run_safety_pipeline & run_context_pipeline
│   ├── workers/                 # Standalone FastAPI workers (e5, prompt, ner)
│   ├── clients/                 # Async httpx clients for worker containers
│   └── static/                  # Showcase UI (index.html)
├── docker/                      # api, e5, prompt, ner Dockerfiles & startup scripts
├── docker-compose.yml           # 4-container local/showcase deployment
├── scripts/
│   └── benchmark_safety.py      # Offline CPU & reference benchmark runner
├── data/eval/                   # Bilingual EN/DE .jsonl eval sets & .json benchmark results
├── docs/                        # Architecture & component documentation
└── tests/                       # Pytest unit, pipeline, and worker/client tests
```

## Quickstart

### 1. Local Development Server

```bash
.venv/bin/uvicorn clinical_guard.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` for the Showcase UI, or `http://localhost:8000/docs` for the OpenAPI specification.

### 2. 4-Container Docker Compose Deployment

Runs `clinical-guard-api` (`:8000`), `e5-guard` (`:8001`), `prompt-guard` (`:8002`), and `ner-pii-guard` (`:8003`):

```bash
docker compose up --build
```

Optional environment variables (`.env` or shell):
- `OPENAI_API_KEY`: Default provider key for `litellm` in `ContextGroundednessChecker` (can also be supplied per request via `X-OpenAI-Api-Key`).
- `HF_TOKEN`: **One-time setup note** — `meta-llama/Llama-Prompt-Guard-2-86M` is a gated Meta repository, so on first startup (when not already cached under `models/`) you must accept the model license on Hugging Face and provide `HF_TOKEN` in `.env`. By contrast, `intfloat/multilingual-e5-small` and `onnx-community/multilang-pii-ner-ONNX` download automatically without a token.

### 3. Tests, Linting & Benchmarks

```bash
# Run unit and integration test suite
.venv/bin/pytest

# Run linter
.venv/bin/ruff check src tests scripts

# Run offline safety & NER benchmarks
.venv/bin/python scripts/benchmark_safety.py
```

## Documentation

- [Overall System Architecture](docs/overall_architecture.md)
- [Prompt Safety Guard](docs/prompt_safety.md)
- [Content Safety Guard](docs/content_safety.md)
- [Policy & Decision Routing Architecture](docs/policy_architecture.md)
- [Context Groundedness Check](docs/context_check.md)
- [Local NER & PII Anonymization](docs/privacy_anonymizer.md)
- [4-Container Docker Deployment & Configuration](docs/deployment.md)

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (**CC BY-NC 4.0**) — see [LICENSE](LICENSE) for details. Non-commercial research and portfolio use only; contact the author for commercial licensing.
