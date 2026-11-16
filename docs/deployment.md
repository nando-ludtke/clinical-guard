# 4-Container Docker Deployment & Configuration

**Scope:** `docker-compose.yml`, `docker/api.Dockerfile`, `docker/e5.Dockerfile`, `docker/prompt.Dockerfile`, `docker/ner.Dockerfile`, `docker/scripts/start_e5_guard.sh`, `docker/scripts/start_prompt_guard.sh`, `docker/scripts/start_ner_guard.sh`, `src/clinical_guard/config.py` (`Settings`), `src/clinical_guard/main.py`, `src/clinical_guard/clients/worker_clients.py`, and `src/clinical_guard/workers/*.py`.

---

## 1. 4-Container Architecture

For production isolation and independent ML dependency packaging (PyTorch CPU vs. ONNX Runtime vs. slim API gateway), `docker-compose.yml` splits `clinical-guard` into four services:

```text
                    ┌────────────────────────────────────────┐
                    │       clinical-guard-api (:8000)       │
                    │  FastAPI gateway, POLICY_MATRIX,       │
                    │  EntityTable, ContextGroundedness      │
                    └───────┬───────────┬───────────┬────────┘
                            │           │           │ HTTP (httpx)
               ┌────────────┘           │           └────────────┐
               ▼                        ▼                        ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│     e5-guard (:8001)    │ │   prompt-guard (:8002)  │ │  ner-pii-guard (:8003)  │
│  E5SafetyGuard          │ │  PromptInjectionGuard   │ │  OnnxPiiDetector        │
│  multilingual-e5-small  │ │  Llama-Prompt-Guard-2   │ │  multilang-pii-ner-ONNX │
└────────────┬────────────┘ └────────────┬────────────┘ └────────────┬────────────┘
             └───────────────────────────┴───────────────────────────┘
                                         │
                             Shared volume: ./models:/app/models
```

| Service | Dockerfile | Port | Model / Role | Healthcheck & Dependencies |
| --- | --- | --- | --- | --- |
| `clinical-guard-api` | `docker/api.Dockerfile` | `8000` | FastAPI API gateway, routing, `EntityTable`, `litellm` judge | `GET /health` (`15s` interval, `5` retries); `depends_on` all 3 workers with `condition: service_healthy` |
| `e5-guard` | `docker/e5.Dockerfile` | `8001` | `E5SafetyGuard` (`intfloat/multilingual-e5-small`) via `e5_worker.py` | `GET /health` (`15s` interval, `10` retries, `60s` `start_period`) |
| `prompt-guard` | `docker/prompt.Dockerfile` | `8002` | `PromptInjectionGuard` (`meta-llama/Llama-Prompt-Guard-2-86M`) via `prompt_worker.py` | `GET /health` (`15s` interval, `10` retries, `60s` `start_period`) |
| `ner-pii-guard` | `docker/ner.Dockerfile` | `8003` | `OnnxPiiDetector` (`onnx-community/multilang-pii-ner-ONNX`) via `ner_worker.py` | `GET /health` (`15s` interval, `10` retries, `60s` `start_period`) |

Start the full stack with:

```bash
docker compose up --build
```

---

## 2. Shared `./models` Volume & Per-Model Startup Scripts

All three worker containers mount `./models:/app/models`. Before launching `uvicorn`, each container runs its startup script in `docker/scripts/`:

1. **`docker/scripts/start_e5_guard.sh`**:
   - Checks `/app/models/multilingual-e5-small`.
   - If missing or empty, downloads `intfloat/multilingual-e5-small` via `SentenceTransformer(..., token=HF_TOKEN)` and saves it to `/app/models/multilingual-e5-small`.
   - Executes `uvicorn clinical_guard.workers.e5_worker:app --host 0.0.0.0 --port 8001`.
2. **`docker/scripts/start_prompt_guard.sh`**:
   - Checks `/app/models/Llama-Prompt-Guard-2-86M`.
   - If missing or empty, calls `huggingface_hub.snapshot_download("meta-llama/Llama-Prompt-Guard-2-86M", local_dir=..., token=HF_TOKEN)`. Because `Llama-Prompt-Guard-2-86M` is a gated Hugging Face repository, `HF_TOKEN` is required on first download when `./models/Llama-Prompt-Guard-2-86M` is not already cached locally.
   - Executes `uvicorn clinical_guard.workers.prompt_worker:app --host 0.0.0.0 --port 8002`.
3. **`docker/scripts/start_ner_guard.sh`**:
   - Checks `/app/models/multilang-pii-ner-ONNX`.
   - If missing or empty, calls `huggingface_hub.snapshot_download("onnx-community/multilang-pii-ner-ONNX", local_dir=..., token=HF_TOKEN)`.
   - Executes `uvicorn clinical_guard.workers.ner_worker:app --host 0.0.0.0 --port 8003`.

Because `./models` is bind-mounted from the host, models downloaded once persist across container rebuilds, and air-gapped deployments can pre-populate `./models/` beforehand.

---

## 3. Configuration & Operating Modes (`src/clinical_guard/config.py`)

All application settings use `pydantic-settings` (`Settings`, `env_prefix="CLINICAL_GUARD_"`, `.env` support):

| Environment Variable | Default | Description |
| --- | --- | --- |
| `CLINICAL_GUARD_ENVIRONMENT` | `"development"` | `"development"`, `"staging"`, or `"production"` |
| `CLINICAL_GUARD_LOG_LEVEL` | `"INFO"` | Structured JSON log level |
| `CLINICAL_GUARD_LOG_REDACT_QUERIES` | `true` | Redacts raw query text in structured logs |
| `CLINICAL_GUARD_E5_GUARD_URL` | `None` | If set (e.g. `http://e5-guard:8001`), uses `HttpE5SafetyGuard` |
| `CLINICAL_GUARD_PROMPT_GUARD_URL` | `None` | If set (e.g. `http://prompt-guard:8002`), uses `HttpPromptInjectionGuard` |
| `CLINICAL_GUARD_NER_GUARD_URL` | `None` | If set (e.g. `http://ner-pii-guard:8003`), uses `HttpNerDetector` |
| `CLINICAL_GUARD_WORKER_TIMEOUT_SECONDS` | `10.0` | HTTP timeout for worker `httpx.AsyncClient` calls |
| `CLINICAL_GUARD_ENABLE_LOCAL_NER` | `false` | Enables in-process `OnnxPiiDetector` when `NER_GUARD_URL` is unset |

### 3.1 In-Process Local Mode vs. Docker HTTP Worker Mode
Implemented in `src/clinical_guard/main.py` (`_build_prompt_guard`, `_build_e5_guard`, `_build_anonymizer`):
- **In-process local mode (default for local `uvicorn`):** When `CLINICAL_GUARD_*_GUARD_URL` env vars are unset (`None`), `clinical-guard-api` instantiates `PromptInjectionGuard`, `E5SafetyGuard`, and `LocalAnonymizer(enable_ner=settings.enable_local_ner)` directly in-process.
- **Docker HTTP worker mode (`docker-compose.yml`):** `clinical-guard-api` sets `CLINICAL_GUARD_E5_GUARD_URL=http://e5-guard:8001`, `CLINICAL_GUARD_PROMPT_GUARD_URL=http://prompt-guard:8002`, and `CLINICAL_GUARD_NER_GUARD_URL=http://ner-pii-guard:8003`, routing inference to `HttpE5SafetyGuard`, `HttpPromptInjectionGuard`, and `HttpNerDetector`.

### 3.2 `OPENAI_API_KEY` vs. `X-OpenAI-Api-Key` (BYOK)
Used by `ContextGroundednessChecker` (`POST /v1/guard/evaluate_context` via `litellm.acompletion`):
- **Server default (`OPENAI_API_KEY`):** Passed into `clinical-guard-api` via environment / `.env`. When no per-request header is sent, `api_key` is omitted from `litellm.acompletion(...)` so `litellm` resolves `OPENAI_API_KEY` (or another configured provider env var) automatically.
- **Per-request BYOK header (`X-OpenAI-Api-Key`):** `POST /v1/guard/evaluate_context` accepts an optional `X-OpenAI-Api-Key` HTTP header (`Header(alias="X-OpenAI-Api-Key")`), forwarding the caller's key directly to `ContextGroundednessChecker.check(..., api_key=x_openai_api_key)` for that request without persisting it server-side.

### 3.3 CPU Threading, Warmup & Layer-Cached Dockerfiles
- **CPU thread-contention defaults (`docker-compose.yml`):** Each ML worker (`e5-guard`, `prompt-guard`, `ner-pii-guard`) sets `OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}`, `MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}`, `OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}`, and `TOKENIZERS_PARALLELISM="false"` to prevent thread oversubscription across concurrent CPU containers.
- **Startup warmup (`src/clinical_guard/workers/*.py`):** Each worker runs a single lightweight warmup inference call (`guard.evaluate("warmup check")` or `detector.detect("warmup check")`) in its FastAPI `lifespan` immediately after `.load()` so lazy PyTorch/ONNX kernels compile before `/health` reports ready.
- **Layer-cached Dockerfiles (`docker/*.Dockerfile`):** All 4 Dockerfiles use `# syntax=docker/dockerfile:1` with BuildKit pip cache mounts (`--mount=type=cache,target=/root/.cache/pip`) to install third-party dependencies before `COPY src/ ./src/`, followed by `pip install --no-deps --no-cache-dir .` so code-only edits rebuild without re-downloading ML wheels.
