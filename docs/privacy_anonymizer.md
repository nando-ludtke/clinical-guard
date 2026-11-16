# Local NER & PII Anonymization

**Scope:** `src/clinical_guard/privacy/anonymizer.py` (`LocalAnonymizer`, `EntityTable`, `EntitySpan`, `OnnxPiiDetector`, `detect_baseline_entities`, `merge_entity_spans`), `src/clinical_guard/workers/ner_worker.py` (`ner-pii-guard` container), `src/clinical_guard/clients/worker_clients.py` (`HttpNerDetector`), `src/clinical_guard/pipelines/context_pipeline.py` (`run_context_pipeline`), and `src/clinical_guard/context/context_check.py` (`ContextGroundednessChecker`).

---

## 1. Rationale

Before `ContextGroundednessChecker` calls a third-party LLM via `litellm.acompletion`, all untrusted and clinical payloads (`query`, `response_text`, `conversation_history`, and every `ContextDocument`) must be scrubbed of PHI/PII **inside the local trust boundary**:

1. **Zero raw PHI/PII egress:** Patient names, dates of birth, contact details, medical record numbers (MRNs), and street addresses must never leave the local runtime in plaintext.
2. **Why hybrid regex + ONNX NER outperforms regex-only:** Deterministic regexes achieve `100%` precision on structured identifiers (emails, phones, ISO/slash dates, `MRN-`/`id:` codes, titled names like `Dr. House`), but miss untitled names (`Sarah Jenkins`, `Lukas Weber`), street addresses (`742 Evergreen Terrace`, `Berliner Straße 45`), and organizations (`St. Jude Clinic`)—yielding only `46.1%` overall recall (`39.5%` on German) in `data/eval/ner_pii_benchmark.json`. Pairing baseline regexes with `onnx-community/multilang-pii-ner-ONNX` (`OnnxPiiDetector`) raises overall recall to `97.4%` (`100.0%` EN, `94.7%` DE) while preserving `100.0%` of clinical terms, dosages, and medical eponyms (`0` false positives).
3. **Referential integrity across documents:** Replacing entities with consistent, request-scoped `<TYPE_N>` placeholders lets the external groundedness judge verify cross-document claims (e.g., `<PERSON_1>` in `doc_1` matching `<PERSON_1>` in `conversation_history`) and allows local uncovering (`deanonymize()`) before returning `GroundednessResult`.

---

## 2. Architecture

### 2.1 Deterministic Baseline (`detect_baseline_entities()`)
Implemented in `src/clinical_guard/privacy/anonymizer.py` via `_BASELINE_PATTERNS`:
- **`EMAIL`**: Standard email addresses (`user@domain.tld`).
- **`ID`**: Explicit clinical/record identifiers (`MRN`, `PATIENT_ID`, `PID` with optional `:`, `#`, `-`) and general `<label>id:` / `id#` patterns (e.g., `Case-ID: A1234`, `id: 99812`).
- **`PHONE`**: International and North American/European phone formats (`+49 ...`, `(415) 555-0192`, etc.).
- **`DATE`**: ISO (`YYYY-MM-DD`), slash/hyphen (`DD/MM/YYYY`), and English month-name dates (`May 14, 1978`, `14 May 1978`).
- **`PERSON`**: Titled names prefixed with `Dr`, `Mr`, `Mrs`, `Ms`, `Prof`, or `Patient` followed by 1–3 capitalized name tokens.

### 2.2 `OnnxPiiDetector` (`models/multilang-pii-ner-ONNX`)
Implemented in `src/clinical_guard/privacy/anonymizer.py`:
- **`resolve_default_ner_model()`**: Prefers a local `models/multilang-pii-ner-ONNX` directory (supporting both root `model.onnx` and `onnx/model.onnx` / `onnx/model_quantized.onnx` layouts) and falls back to `"onnx-community/multilang-pii-ner-ONNX"`.
- **Lazy initialization (`load()`)**: Loads `AutoTokenizer` and `ORTModelForTokenClassification` on CPU (`device=-1`) wrapped in a `transformers.pipeline("ner", aggregation_strategy="simple")`.
- **Post-processing & precision guards (`detect(text)`)**:
  - **Score threshold & ignored labels**: Drops raw spans below `min_score=0.35`, normalizes BIO/entity labels via `_NER_LABEL_MAP` (`PER` -> `PERSON`, `LOC`/`STREET`/`ADDRESS`/`ZIPCODE` -> `LOCATION`, `ORG` -> `ORG`, `DOB` -> `DATE`, etc.), and skips `IGNORED_NER_ENTITY_TYPES` (`AGE`, intentionally excluded from masking due to its clinical relevance).
  - **Leading `+` phone expansion**: Expands `PHONE` / `TELEPHONENUM` spans one character left when immediately preceded by `+` so country codes are not left dangling.
  - **Word-boundary guard**: Rejects subword fragments that start or end mid-word (adjacent to `.isalnum()`, e.g., `"Met"` inside `"Metformin"`), as well as spans shorter than 2 characters or without alphanumeric characters.
  - **`BUILDINGNUM` address-context guard**: Keeps `BUILDINGNUM` spans only when `_has_address_context()` matches street/address keywords (e.g., `street`, `avenue`, `straße`, `str`, `platz`, `weg`, `adresse`) within a `40`-character window—preventing false positives on clinical dosages or lab values (`500 mg`, `120/80`).
  - **`_merge_adjacent_person_spans()`**: Merges adjacent `PERSON` spans separated only by whitespace or `-` (e.g., first + last name or hyphenated surnames) into a single span.

### 2.3 `merge_entity_spans()` & Request-Scoped `EntityTable`
- **`LocalAnonymizer.detect(text)`** runs `detect_baseline_entities(text)` plus optional `OnnxPiiDetector` (in-process via `enable_ner=True`, or remote `HttpNerDetector` pointing to `ner-pii-guard` on port `8003`) and any `custom_detector`.
- **`merge_entity_spans(primary_spans, secondary_spans, text_length)`**: Resolves baseline regex spans first (earliest, then longest) and accepts NER/custom spans only when they do not overlap an already-accepted regex span—giving deterministic regexes strict priority on overlaps.
- **Request-scoped `EntityTable`**:
  - Normalizes `(entity_type, " ".join(original_text.split()).casefold())` to assign deterministic `<TYPE_N>` placeholders (`<PERSON_1>`, `<DATE_1>`, `<ORG_1>`, `<LOCATION_1>`, etc.).
  - Shared across `query`, `conversation_history`, `response_text`, and all `ContextDocument`s for a single `run_context_pipeline()` call, then discarded at request completion (never logged or persisted).
  - **`deanonymize(text)`**: Uncovers `<TYPE_N>` placeholders back to original entity strings (longest placeholder matched first) across `GroundednessResult` fields (`unsupported_claims`, `contradictions`, `reasoning`, and each `ClaimCitation.claim` / `ClaimCitation.quote`).

### 2.4 Dedicated `ner-pii-guard` Worker (`ner_worker.py`)
In the 4-container `docker-compose.yml` deployment, `OnnxPiiDetector` runs in the `ner-pii-guard` container (`src/clinical_guard/workers/ner_worker.py`, port `8003`), exposing `GET /health` and `POST /predict`. `clinical-guard-api` delegates NER span detection over HTTP via `HttpNerDetector` (`CLINICAL_GUARD_NER_GUARD_URL`), while keeping `detect_baseline_entities()`, `merge_entity_spans()`, and `EntityTable` in the API process.

---

## 3. Benchmark Results

Results observed in `data/eval/ner_pii_benchmark.json` across the 24-row bilingual evaluation set (`data/eval/ner_pii_eval.jsonl`: `12` `en` / `12` `de`, `76` `must_mask` entities, `83` `must_preserve` clinical terms):

| Metric / Breakdown | `regex_only` | `regex_plus_onnx_ner` (`models/multilang-pii-ner-ONNX`) |
| --- | --- | --- |
| **Overall Mask Recall** | `46.1%` (`0.4605`, `35/76`) | `97.4%` (`0.9737`, `74/76`) |
| **German (`de`) Mask Recall** | `39.5%` (`0.3947`, `15/38`) | `94.7%` (`0.9474`, `36/38`) |
| **English (`en`) Mask Recall** | `52.6%` (`0.5263`, `20/38`) | `100.0%` (`1.0`, `38/38`) |
| **Clinical Preservation Rate** | `100.0%` (`83/83`) | `100.0%` (`83/83`) |
| **False Positives (`must_preserve`)** | `0` (`0.0%`) | `0` (`0.0%`) |
| **Model Load Time** | `0.0 ms` | `896.66 ms` |
| **Mean Latency** | `0.05 ms` (`0.0507 ms`) | `26.35 ms` (`26.3484 ms`) |
| **p50 / p95 / Max Latency** | `0.0474` / `0.0629` / `0.0888 ms` | `24.9248` / `36.124` / `42.0171 ms` |

### Breakdown by Case Type (`mask_recall` / `preservation_rate`)

| Case Type | `regex_only` | `regex_plus_onnx_ner` |
| --- | --- | --- |
| `standard_pii` (`16` mask) | `62.5%` (`10/16`) / `100%` | `100.0%` (`16/16`) / `100%` |
| `multilingual_name_address` (`16` mask) | `37.5%` (`6/16`) / `100%` | `100.0%` (`16/16`) / `100%` |
| `clinical_eponym_trap` (`14` mask) | `57.1%` (`8/14`) / `100%` | `92.9%` (`13/14`) / `100%` |
| `dosage_lab_trap` (`14` mask) | `50.0%` (`7/14`) / `100%` | `100.0%` (`14/14`) / `100%` |
| `mixed_edge_case` (`16` mask) | `25.0%` (`4/16`) / `100%` | `93.8%` (`15/16`) / `100%` |
