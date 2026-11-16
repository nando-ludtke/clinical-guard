# Content Safety Guard — Multilingual E5 Semantic Layer

**Scope:** `src/clinical_guard/safety/content_safety.py` (`E5SafetyGuard`, `E5GuardResult`, `E5Action`) and its integration with `src/clinical_guard/main.py`, `src/clinical_guard/pipelines/safety_pipeline.py`, `src/clinical_guard/workers/e5_worker.py`, `src/clinical_guard/clients/worker_clients.py`, `src/clinical_guard/routing/policy.py`, `src/clinical_guard/schemas.py`, and `src/clinical_guard/logging_config.py`.

---

## 1. Rationale

Rule-based keyword or regex filters cannot generalize across:
- **Non-English languages:** Multilingual clinical deployments require consistent recall across languages (e.g., English and German) without maintaining brittle per-language pattern lists.
- **Slang, euphemisms, typos, and obfuscations:** Terms like `"unalive myself"`, `"suizd gedanken"`, or `"pool chlorene fumes"` evade literal string matching.
- **Semantic context & idioms:** Surface keywords like `"killing me"` or `"dying for"` appear in benign idioms, while acute clinical terminology appears in legitimate `medical_professional` queries.

`E5SafetyGuard` runs a compact, CPU-feasible multilingual semantic safety classifier (`intfloat/multilingual-e5-small`, `~26.8 ms` mean latency on CPU) directly on incoming queries after prompt-injection screening.

---

## 2. Benchmark Evidence

Results observed in `data/eval/clinical_safety_benchmark.json` across the 48-case bilingual evaluation set (`data/eval/clinical_safety_eval.jsonl`, 24 `en` / 24 `de`) using `models/multilingual-e5-small` on CPU with per-category `hard_block` margins (`self_harm: 0.02`, `overdose: 0.045`, `toxicology: 0.045`, `model_load_ms: 1921.08`):

| Metric / Breakdown (`metrics_crisis_vs_allow`) | `models/multilingual-e5-small` (CPU) |
| --- | --- |
| **Accuracy** | `89.6%` (`0.8958`, `43/48`) |
| **Precision** | `93.3%` (`0.9333`, `28/30`) |
| **Recall** | `90.3%` (`0.9032`, `28/31`) |
| **F1 Score** | `0.9180` (`0.918`) |
| **German (`de`) Accuracy / Recall** | `91.7%` (`0.9167`) / `93.8%` (`15/16`) |
| **English (`en`) Accuracy / Recall** | `87.5%` (`0.875`) / `86.7%` (`13/15`) |
| **Explicit Crisis Recall** | `100.0%` (`8/8`) |
| **Slang / Typo Recall** | `100.0%` (`8/8`) |
| **Ambiguous Boundary Accuracy** | `100.0%` (`12/12`) |
| **Idiom False Positives (`idiom_false_positive`)** | `0/6` |
| **Mean Latency** | `26.8424 ms` (`26.84 ms`) |
| **p50 / p95 Latency** | `26.3993 ms` / `34.1678 ms` |
| **Max Latency** | `35.7765 ms` |

*(Note: Under `metrics_hard_block_only`, `models/multilingual-e5-small` achieved `75.0%` accuracy, `60.0%` precision, `100.0%` recall, and `0.75` F1.)*

### Candidate Alternatives Kept in Mind
While `intfloat/multilingual-e5-small` is the baseline CPU-ready architecture, the modular guard interface accommodates future drop-in evaluators:
- **`mmbert-safety`**: Dedicated multilingual safety encoder candidate.
- **`Qwen3Guard-Stream-0.6B`**: Streaming-capable guardrail model for token/prefix safety.
- **`MANTA-4B`**: Larger model candidate reserved for **GPU-backed clinical decision triage**, not CPU-bound pre-filtering.

---

## Test Environment

Local CPU benchmark environment used to run `.venv/bin/python`:

| Component | Observed Value |
| --- | --- |
| **Architecture** | `arm64` |
| **CPU** | `Apple M3` (`8` logical cores, `hw.ncpu = 8`) |
| **Memory** | `16 GiB` (`17179869184` bytes, `hw.memsize`) |
| **OS** | `macOS 27.2` (`BuildVersion: 26B123`) |
| **Python** | `Python 3.11.9` (`.venv/bin/python`) |
| **Key Libraries** | `torch==2.15.0`, `transformers==5.23.0`, `sentence-transformers==6.3.0`, `fastapi==0.163.1`, `pydantic==2.14.3`, `litellm==1.109.1` |

---

## 3. Architecture

### 3.1 `E5SafetyGuard` & `E5GuardResult`
Implemented in `src/clinical_guard/safety/content_safety.py`:
- **`resolve_default_e5_model()`**: Prefers a local `models/multilingual-e5-small` directory when present (enabling offline/air-gapped container runs) and falls back to `"intfloat/multilingual-e5-small"`.
- **Startup / lazy load (`load()`)**: Loads `SentenceTransformer` on CPU and pre-encodes L2-normalized anchor passages with the required `"passage: "` prefix across `self_harm`, `overdose`, `toxicology`, `prompt_injection` (bilingual EN/DE roleplay & persona-override jailbreak prototypes), and `benign` (including clinical, idiom, and educational-framing anchors).
- **Non-blocking inference**: `main.py` executes `e5_guard.evaluate(payload.query)` via `anyio.to_thread.run_sync` so CPU inference does not block the FastAPI event loop.

### 3.2 Margin-Based Classification & Per-Category `hard_block` Margins
For each incoming query:
1. Encode `"query: " + query` and L2-normalize.
2. Compute cosine similarity (dot product) against all pre-encoded anchor embeddings and take the max score per category.
3. If `top_category == "prompt_injection"` and `top_score - benign_score >= prompt_injection_margin` (defaults to `hard_block_margin = 0.05`) -> `action = "hard_block"`, `risk_category = "prompt_injection"`, which `resolve_policy()` routes to `decision = "block_security"`, `routing_target = "security_block"` for all roles.
4. Otherwise look up `effective_hard_margin = self.category_hard_block_margins.get(top_category, self.hard_block_margin)` using `DEFAULT_CATEGORY_HARD_BLOCK_MARGINS` (`self_harm: 0.02`, `overdose: 0.045`, `toxicology: 0.045`; fallback `hard_block_margin = 0.05`) and compute `margin = top_crisis_score - benign_score`:
   - `margin >= effective_hard_margin` -> `action = "hard_block"`, `risk_category = top_crisis_cat`.
   - `margin >= soft_flag_margin` (default `0.0`) -> `action = "soft_flag"`, `risk_category = top_crisis_cat`.
   - Otherwise -> `action = "allow"`, `risk_category = "none"`.

The tighter `0.02` margin on `self_harm` preserves high sensitivity for acute self-harm, while the `0.045` margin on `overdose` and `toxicology` reduces over-blocking on borderline clinical or layman medication/chemical queries (routing `0.0 <= margin < 0.045` to `soft_flag`).

### 3.3 Role-Aware Downstream Handling
`E5SafetyGuard` remains role-agnostic and emits `E5GuardResult(action, risk_category, top_category, top_score, benign_score, margin, latency_ms)`. Role-specific escalation (`soft_flag` -> `bypass_emergency` for `layman` vs. `route_clinical` with audit flag for `medical_professional`) is handled cleanly by `POLICY_MATRIX` in `src/clinical_guard/routing/policy.py`.

---

## 4. Containerization & Optimization Plan

1. **Dedicated Container Service (`e5-guard`) [Implemented]**: `E5SafetyGuard` is packaged with `models/multilingual-e5-small` in `docker/e5.Dockerfile` (`src/clinical_guard/workers/e5_worker.py`, port `8001`) as part of the 4-container `docker-compose.yml` deployment, called by `clinical-guard-api` via `HttpE5SafetyGuard` (`src/clinical_guard/clients/worker_clients.py`).
2. **ONNX INT8 Quantization**: Export `intfloat/multilingual-e5-small` to ONNX with dynamic INT8 quantization (`onnxruntime`) to reduce CPU memory footprint and lower p95 latency under concurrent load.
3. **Timeout & Fail-Closed Wrapper**: Enforce a configurable per-guard latency budget (beyond `CLINICAL_GUARD_WORKER_TIMEOUT_SECONDS`) with fail-closed emergency escalation for `layman` queries on timeout/error.
4. **Expand Multilingual & Adversarial Eval Set**: Grow `data/eval/clinical_safety_eval.jsonl` beyond 48 rows to cover additional languages, indirect self-harm paraphrasing, and borderline `medical_professional` queries.

---

## Optional / Future Planning

### Lightweight Linear / Logistic Regression Classifier Head on `multilingual-e5-small` Embeddings
As a future alternative or complement to zero-shot cosine-similarity anchor margins, a lightweight supervised **Linear / Logistic Regression classifier head** can be trained directly on top of the frozen 384-dimensional L2-normalized `"query: "` embeddings produced by `intfloat/multilingual-e5-small`:

- **Zero additional encoder cost**: Reuses the exact same 384-dim CPU embedding already computed per request (`~17–20 ms`), adding `< 0.1 ms` for a single `(5, 384)` matrix-vector product + softmax/sigmoid.
- **Better calibration on ambiguous boundaries**: Supervised weights can separate subtle clinical phrasing (e.g., `ambiguous_clinical` or `clinical_professional` dosage discussions) from acute crisis queries (`self_harm`, `overdose`, `toxicology`, `prompt_injection`) more cleanly than nearest-anchor max-similarity alone.
- **Calibrated probabilities**: Outputs well-calibrated class probabilities `P(category | x)` that map naturally to `hard_block`, `soft_flag`, and `allow` thresholds without manual anchor-phrase curation.
- **Lightweight artifact & governance**: Serialized as small weight/bias arrays (e.g., JSON or `.npz` alongside `models/multilingual-e5-small`) with no extra runtime dependencies beyond `numpy`, and trained strictly on held-out/curated splits to avoid overfitting the evaluation set.

---

## External Reference Baseline: TypeSafe AI Jev

For offline benchmarking and calibration reference, `scripts/benchmark_safety.py --target jev` supports evaluating TypeSafe AI's external `jev-1.14.0` System 1 endpoint (`data/eval/jev_benchmark.json`):

- **Content Safety (`48` rows, `clinical_safety_eval.jsonl`)**: `95.8%` accuracy (`0.9583`, `46/48`, `F1 = 0.963`), `188.8 ms` mean latency (`177.5 ms` p50, `275.9 ms` p95).
- **Prompt Safety (`32` rows, `prompt_safety_eval.jsonl`)**: `100.0%` accuracy (`32/32`, `F1 = 1.0`), `212.8 ms` mean latency (`183.9 ms` p50, `367.4 ms` p95).

**Why Jev is not part of the runtime service:**
1. **Raw PHI/PII egress**: System 1 safety screening runs *before* downstream clinical reasoning; sending unredacted patient queries to an external third-party API would violate local-first PHI/PII containment unless preceded by NER anonymization.
2. **External API dependency & cost**: Runtime availability, rate limits, and per-request API cost would couple every inbound request (including benign traffic) to an external SaaS endpoint.
3. **`NER -> Jev` latency**: Adding local NER anonymization before an external `~188.8–212.8 ms` network round-trip would substantially exceed the local CPU budget (`~26.8 ms` for `E5SafetyGuard`).

`--target jev` therefore remains strictly an **optional offline reference benchmark** in `scripts/benchmark_safety.py`.
