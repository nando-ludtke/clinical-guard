"""Benchmark JSON loader and normalizer for the /v1/benchmarks showcase endpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clinical_guard.logging_config import get_logger

logger = get_logger(__name__)

_REPO_EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"


def _resolve_eval_dir() -> Path:
    cwd_eval = Path("data/eval")
    if cwd_eval.is_dir():
        return cwd_eval
    return _REPO_EVAL_DIR


def _normalize_clinical_safety(raw: dict[str, Any]) -> dict[str, Any]:
    bench = raw.get("benchmark", {})
    predictions = []
    for row in bench.get("predictions", []):
        exp = str(row.get("expected_action", ""))
        pred = str(row.get("predicted_action", ""))
        predictions.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "query": row.get("query", ""),
                "expected": f"{exp} ({row.get('expected_category', '')})",
                "predicted": f"{pred} ({row.get('predicted_category', '')})",
                "correct": (exp != "allow") == (pred != "allow"),
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "user_role": row.get("user_role"),
                    "case_type": row.get("case_type"),
                    "top_category": row.get("top_category"),
                    "top_score": row.get("top_score"),
                    "benign_score": row.get("benign_score"),
                    "margin": row.get("crisis_minus_benign_margin"),
                },
            }
        )
    return {
        "id": "clinical_safety",
        "title": "Content Safety (multilingual-e5-small)",
        "model_name": bench.get("name", "models/multilingual-e5-small"),
        "status": "active",
        "category": "Content Safety",
        "rationale": (
            "Zero-shot multilingual (EN/DE) cosine-margin guard on CPU (~27ms mean). "
            "Runs 100% locally with zero PHI egress and separates soft_flag from hard_block."
        ),
        "total_rows": raw.get("total_rows", len(predictions)),
        "latency_ms": bench.get("latency_ms", {}),
        "metrics": bench.get("metrics_crisis_vs_allow", {}),
        "secondary_metrics": {
            "hard_block_only": bench.get("metrics_hard_block_only", {}),
        },
        "predictions": predictions,
    }


def _normalize_prompt_hybrid(raw: dict[str, Any]) -> dict[str, Any]:
    bench = raw.get("benchmark", {})
    predictions = []
    for row in bench.get("predictions", []):
        exp = bool(row.get("expected_is_injection"))
        pred = bool(row.get("predicted_is_injection"))
        predictions.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "query": row.get("query", ""),
                "expected": "injection" if exp else "benign",
                "predicted": (
                    f"{'injection' if pred else 'benign'} ({row.get('predicted_source', '')})"
                ),
                "correct": exp == pred,
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "attack_type": row.get("attack_type"),
                    "user_role": row.get("user_role"),
                    "prompt_guard_score": row.get("prompt_guard_score"),
                    "prompt_guard_label": row.get("prompt_guard_label"),
                    "e5_top_category": row.get("e5_top_category"),
                    "e5_margin": row.get("e5_margin"),
                },
            }
        )
    return {
        "id": "prompt_hybrid",
        "title": "Prompt Safety Hybrid (Prompt-Guard-2-86M + Regex + E5)",
        "model_name": bench.get(
            "name", "models/Llama-Prompt-Guard-2-86M + models/multilingual-e5-small"
        ),
        "status": "active",
        "category": "Prompt Safety",
        "rationale": (
            "Production hybrid gate combining Llama-Prompt-Guard-2-86M, deterministic delimiter "
            "smuggling regex (<|im_start|>, [INST], <<SYS>>), and E5 prompt-injection anchors."
        ),
        "total_rows": raw.get("total_rows", len(predictions)),
        "latency_ms": bench.get("latency_ms", {}),
        "metrics": bench.get("metrics", {}),
        "secondary_metrics": {
            "prompt_guard_latency_ms": bench.get("prompt_guard_latency_ms", {}),
            "e5_latency_ms": bench.get("e5_latency_ms", {}),
        },
        "predictions": predictions,
    }


def _normalize_prompt_safety(raw: dict[str, Any]) -> dict[str, Any]:
    bench = raw.get("benchmark", {})
    predictions = []
    for row in bench.get("predictions", []):
        exp = bool(row.get("expected_is_injection"))
        pred = bool(row.get("predicted_is_injection"))
        predictions.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "query": row.get("query", ""),
                "expected": "injection" if exp else "benign",
                "predicted": (
                    f"{'injection' if pred else 'benign'} ({row.get('predicted_label', '')})"
                ),
                "correct": exp == pred,
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "attack_type": row.get("attack_type"),
                    "user_role": row.get("user_role"),
                    "injection_score": row.get("injection_score"),
                },
            }
        )
    return {
        "id": "prompt_safety",
        "title": "Prompt Safety Standalone (Llama-Prompt-Guard-2-86M)",
        "model_name": bench.get("name", "models/Llama-Prompt-Guard-2-86M"),
        "status": "active",
        "category": "Prompt Safety",
        "rationale": (
            "86M-parameter DeBERTa-class prompt-injection classifier running locally on CPU "
            "(~81ms mean, 1.00 precision) as the core of the hybrid prompt-injection gate."
        ),
        "total_rows": raw.get("total_rows", len(predictions)),
        "latency_ms": bench.get("latency_ms", {}),
        "metrics": bench.get("metrics", {}),
        "secondary_metrics": {},
        "predictions": predictions,
    }


def _normalize_qwen3guard(raw: dict[str, Any]) -> dict[str, Any]:
    bench = raw.get("benchmark", {})
    predictions = []
    for row in bench.get("predictions", []):
        exp = bool(row.get("expected_is_injection"))
        pred = bool(row.get("predicted_is_flagged"))
        predictions.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "query": row.get("query", ""),
                "expected": "injection" if exp else "benign",
                "predicted": (
                    f"{row.get('predicted_risk_level', '')} ({row.get('predicted_category', '')})"
                ),
                "correct": exp == pred,
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "attack_type": row.get("attack_type"),
                    "risk_prob": row.get("risk_prob"),
                    "category_prob": row.get("category_prob"),
                    "predicted_is_unsafe": row.get("predicted_is_unsafe"),
                },
            }
        )
    return {
        "id": "qwen3guard",
        "title": "Qwen3Guard-Stream-0.6B (Evaluated & Excluded)",
        "model_name": bench.get("name", "Qwen/Qwen3Guard-Stream-0.6B"),
        "status": "excluded",
        "category": "Evaluated Baseline",
        "rationale": (
            "Excluded from runtime: ~232ms mean CPU latency (nearly 3x Prompt-Guard-2-86M), "
            "~1.2GB memory footprint, and higher false-positive rate on benign clinical queries."
        ),
        "total_rows": raw.get("total_rows", len(predictions)),
        "latency_ms": bench.get("latency_ms", {}),
        "metrics": bench.get("metrics", {}),
        "secondary_metrics": {
            "unsafe_only": bench.get("metrics_unsafe_only", {}),
        },
        "predictions": predictions,
    }


def _normalize_ner_pii(raw: dict[str, Any]) -> dict[str, Any]:
    benchmarks = raw.get("benchmarks", {})
    onnx_bench = benchmarks.get("regex_plus_onnx_ner", {})
    regex_bench = benchmarks.get("regex_only", {})
    predictions = []
    for row in onnx_bench.get("predictions", []):
        missed = row.get("missed_masks", [])
        fps = row.get("false_positives", [])
        predictions.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "query": row.get("text", ""),
                "expected": f"mask {len(row.get('must_mask', []))} PII items",
                "predicted": (
                    f"masked {len(row.get('masked_items', []))}, "
                    f"missed {len(missed)}, fp {len(fps)}"
                ),
                "correct": len(missed) == 0 and len(fps) == 0,
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "case_type": row.get("case_type"),
                    "anonymized_text": row.get("anonymized_text"),
                    "missed_masks": missed,
                    "false_positives": fps,
                    "mask_recall": row.get("mask_recall"),
                    "preservation_rate": row.get("preservation_rate"),
                },
            }
        )
    return {
        "id": "ner_pii",
        "title": "NER / PII Anonymization (Regex + multilang-pii-ner-ONNX)",
        "model_name": onnx_bench.get("model_name", "models/multilang-pii-ner-ONNX"),
        "status": "active",
        "category": "Privacy / NER",
        "rationale": (
            "Boosts PII mask recall from 46.1% (regex-only) to 97.4% at ~26ms mean CPU latency "
            "while preserving 100% of clinical terms (medications, labs, vitals) before LLM egress."
        ),
        "total_rows": raw.get("total_rows", len(predictions)),
        "latency_ms": onnx_bench.get("latency_ms", {}),
        "metrics": onnx_bench.get("metrics", {}),
        "secondary_metrics": {
            "regex_only": regex_bench.get("metrics", {}),
        },
        "predictions": predictions,
    }


def _normalize_jev(raw: dict[str, Any]) -> dict[str, Any]:
    bench = raw.get("benchmark", {})
    clin = bench.get("clinical_safety", {})
    prompt = bench.get("prompt_safety", {})
    predictions = []
    for row in clin.get("predictions", []):
        exp = str(row.get("expected_action", ""))
        pred = str(row.get("predicted_action", ""))
        predictions.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "query": row.get("query", ""),
                "expected": f"[clinical] {exp} ({row.get('expected_category', '')})",
                "predicted": f"[clinical] {pred} ({row.get('predicted_category', '')})",
                "correct": (exp != "allow") == (pred != "allow"),
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "suite_part": "clinical_safety",
                    "case_type": row.get("case_type"),
                    "user_role": row.get("user_role"),
                },
            }
        )
    for row in prompt.get("predictions", []):
        exp_inj = bool(row.get("expected_is_injection"))
        pred_inj = bool(row.get("predicted_is_injection"))
        predictions.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "query": row.get("query", ""),
                "expected": f"[prompt] {'injection' if exp_inj else 'benign'}",
                "predicted": f"[prompt] {'injection' if pred_inj else 'benign'}",
                "correct": exp_inj == pred_inj,
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "suite_part": "prompt_safety",
                    "attack_type": row.get("attack_type"),
                    "noul_score": row.get("noul_score"),
                },
            }
        )
    return {
        "id": "jev",
        "title": "TypeSafe AI Jev (jev-1.14.0) (Evaluated & Excluded)",
        "model_name": bench.get("name", "TypeSafe AI Jev (jev-1.14.0)"),
        "status": "excluded",
        "category": "External Baseline",
        "rationale": (
            "Excluded from runtime despite strong accuracy (95.8% clinical, 100% prompt): "
            "requires external cloud API egress (violates zero-PHI-egress pre-filtering) and "
            "adds ~198ms network latency vs ~27ms local CPU."
        ),
        "total_rows": len(predictions),
        "latency_ms": bench.get("overall_latency_ms", {}),
        "metrics": clin.get("metrics_crisis_vs_allow", {}),
        "secondary_metrics": {
            "prompt_safety": prompt.get("metrics", {}),
            "clinical_hard_block_only": clin.get("metrics_hard_block_only", {}),
        },
        "predictions": predictions,
    }


def _load_context_eval_rows(eval_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load context_check_eval.jsonl rows keyed by id for query and letter previews."""
    base_dir = eval_dir or _resolve_eval_dir()
    eval_path = base_dir / "context_check_eval.jsonl"
    if not eval_path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    try:
        for line in eval_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item.get("id")
            if row_id:
                rows[str(row_id)] = item
    except (OSError, ValueError) as exc:
        logger.warning(
            "context_eval_load_failed",
            extra={"file": str(eval_path), "error": str(exc)},
        )
    return rows


def _normalize_context_check(
    raw: dict[str, Any], eval_dir: Path | None = None
) -> dict[str, Any]:
    bench = raw.get("benchmark", {})
    eval_rows = _load_context_eval_rows(eval_dir)
    predictions = []
    for row in bench.get("predictions", []):
        row_id = str(row.get("id", ""))
        eval_item = eval_rows.get(row_id, {})
        exp_grounded = row.get("expected_grounded")
        pred_grounded = row.get("pred_grounded")
        exp_block = bool(row.get("expected_security_block"))
        pred_block = bool(row.get("pred_security_blocked"))
        exp_flagged = row.get("expected_flagged_doc_ids", [])
        pred_flagged = row.get("pred_flagged_doc_ids", [])

        if exp_block or pred_block:
            correct = exp_block == pred_block and set(exp_flagged) == set(pred_flagged)
        else:
            correct = exp_block == pred_block and exp_grounded == pred_grounded

        cit_total = int(row.get("context_doc_citation_count", 0))
        cit_verified = int(row.get("verified_context_doc_citation_count", 0))
        quote_verified_summary = f"{cit_verified}/{cit_total}" if cit_total else "n/a"

        contexts = eval_item.get("contexts") or []
        first_doc_text = str(contexts[0].get("text", "")) if contexts else ""
        compact_letter = " ".join(first_doc_text.split())
        letter_preview = (
            f"{compact_letter[:180]}..." if len(compact_letter) > 180 else compact_letter
        )

        predictions.append(
            {
                "id": row_id,
                "language": row.get("language", ""),
                "query": eval_item.get("query") or str(row.get("case_type", "")),
                "expected": f"grounded={exp_grounded}, security_block={exp_block}",
                "predicted": f"grounded={pred_grounded}, security_block={pred_block}",
                "correct": correct,
                "latency_ms": row.get("latency_ms"),
                "details": {
                    "case_type": row.get("case_type"),
                    "expected_grounded": exp_grounded,
                    "predicted_grounded": pred_grounded,
                    "expected_security_block": exp_block,
                    "predicted_security_block": pred_block,
                    "expected_source_doc_ids": row.get("expected_source_doc_ids", []),
                    "pred_source_doc_ids": row.get("pred_source_doc_ids", []),
                    "quote_verified": quote_verified_summary,
                    "context_doc_citation_count": cit_total,
                    "verified_context_doc_citation_count": cit_verified,
                    "letter_preview": letter_preview,
                },
            }
        )
    return {
        "id": "context_check",
        "title": "Context Groundedness (Prompt-Guard + ONNX NER + gpt-4o-mini)",
        "model_name": bench.get(
            "name",
            "ContextPipeline (models/Llama-Prompt-Guard-2-86M + "
            "multilang-pii-ner-ONNX + openai/gpt-4o-mini)",
        ),
        "status": "active",
        "category": "Context Groundedness",
        "rationale": (
            "3-stage doctor's-letter groundedness pipeline: parallel Llama-Prompt-Guard-2-86M "
            "indirect-injection gate and local multilang-pii-ner-ONNX shared EntityTable "
            "anonymization before openai/gpt-4o-mini verbatim quote-verified judging."
        ),
        "total_rows": raw.get("total_rows", len(predictions)),
        "latency_ms": bench.get("latency_ms", {}),
        "metrics": bench.get("metrics", {}),
        "secondary_metrics": {
            "by_language": bench.get("breakdowns", {}).get("by_language", {}),
            "by_case_type": bench.get("breakdowns", {}).get("by_case_type", {}),
        },
        "predictions": predictions,
    }


_BENCHMARK_LOADERS: tuple[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
    ("clinical_safety_benchmark.json", _normalize_clinical_safety),
    ("prompt_hybrid_benchmark.json", _normalize_prompt_hybrid),
    ("prompt_safety_benchmark.json", _normalize_prompt_safety),
    ("qwen3guard_benchmark.json", _normalize_qwen3guard),
    ("ner_pii_benchmark.json", _normalize_ner_pii),
    ("jev_benchmark.json", _normalize_jev),
    ("context_check_benchmark.json", _normalize_context_check),
)


def load_benchmark_suites(eval_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load and normalize available benchmark JSON suites from data/eval/."""
    base_dir = eval_dir or _resolve_eval_dir()
    suites: list[dict[str, Any]] = []
    for filename, normalizer in _BENCHMARK_LOADERS:
        path = base_dir / filename
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if normalizer is _normalize_context_check:
                suites.append(_normalize_context_check(raw, eval_dir=base_dir))
            else:
                suites.append(normalizer(raw))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("benchmark_load_failed", extra={"file": str(path), "error": str(exc)})
    return suites
