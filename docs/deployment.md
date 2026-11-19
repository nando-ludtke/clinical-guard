# Docker Deployment & Configuration

**Scope:** `docker-compose.yml`, `docker-compose.single.yml`, `docker/*.Dockerfile`, `docker/scripts/*.sh`, `src/clinical_guard/config.py` (`Settings`), `src/clinical_guard/rate_limit.py`, `src/clinical_guard/main.py`, `src/clinical_guard/clients/worker_clients.py`, and `src/clinical_guard/workers/*.py`.

---

## 1. Deployment Modes: Single-Container vs. 4-Container

### 1.1 Single-Container Mode (`docker-compose.single.yml`) — Recommended for 4 GB VPS / Showcases
For single-node or 4 GB RAM VPS deployments (e.g., Hetzner `CAX11`), `docker-compose.single.yml` (`docker/single.Dockerfile` + `docker/scripts/start_single.sh`) runs all three models in-process inside one container:
- Eliminates duplicated Python/PyTorch/ONNX runtimes and internal HTTP hops (~2–2.5 GB RAM instead of ~3.5 GB).
- Sets `CLINICAL_GUARD_ENABLE_LOCAL_NER=true` so `OnnxPiiDetector` runs in-process.
- Binds `127.0.0.1:8080:8000` so a reverse proxy (Caddy/Nginx) handles public HTTPS on port `8080` -> container `8000`.
- Downloads any missing models into `./models` on first boot using `HF_TOKEN`.

```bash
docker compose -f docker-compose.single.yml up -d --build
```

### 1.2 4-Container Architecture (`docker-compose.yml`)

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
| `clinical-guard-api` | `docker/api.Dockerfile` | `8080 -> 8000` | FastAPI API gateway, routing, `EntityTable`, `litellm` judge | `GET /health` (`15s` interval, `5` retries); `depends_on` all 3 workers with `condition: service_healthy` |
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
| `CLINICAL_GUARD_RATE_LIMIT_ENABLED` | `true` | Enables per-IP sliding-window rate limiting on `/v1/guard/*` |
| `CLINICAL_GUARD_RATE_LIMIT_REQUESTS` | `20` | Max requests per IP within the sliding window |
| `CLINICAL_GUARD_RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding-window duration in seconds |
| `CLINICAL_GUARD_MAX_CONCURRENT_INFERENCES` | `4` | Max concurrent model-pipeline executions (`2` in `docker-compose.single.yml`) |
| `CLINICAL_GUARD_INFERENCE_QUEUE_TIMEOUT_SECONDS` | `15.0` | Seconds a request waits for an inference slot before returning `503` |

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
- **Layer-cached Dockerfiles (`docker/*.Dockerfile`):** All Dockerfiles use `# syntax=docker/dockerfile:1` with BuildKit pip cache mounts (`--mount=type=cache,target=/root/.cache/pip`) to install third-party dependencies before `COPY src/ ./src/`.

### 3.4 Rate Limiting & Inference Concurrency (`src/clinical_guard/rate_limit.py`)
All three guard endpoints (`POST /v1/guard/evaluate`, `POST /v1/guard/prompt`, `POST /v1/guard/evaluate_context`) are protected by:
- **`SlidingWindowRateLimiter` (`enforce_rate_limit`):** Extracts the client IP (`X-Forwarded-For` → `X-Real-IP` → `request.client.host`), tracks sliding-window timestamps per IP (default `20` requests per `60s`), prunes stale buckets, and returns `HTTP 429` with `Retry-After`. System and benchmark routes (`/`, `/health`, `/v1/config/status`, `/v1/benchmarks`) remain unthrottled.
- **`acquire_inference_slot` (`asyncio.Semaphore`):** Bounds concurrent CPU inference runs (`CLINICAL_GUARD_MAX_CONCURRENT_INFERENCES`) and returns `HTTP 503` (`Retry-After: 5`) if a slot is not acquired within `CLINICAL_GUARD_INFERENCE_QUEUE_TIMEOUT_SECONDS`.
