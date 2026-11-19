"""Benchmark clinical-guard prompt-injection pipeline against deepset/prompt-injections."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from clinical_guard.safety.content_safety import E5SafetyGuard
from clinical_guard.safety.prompt_safety import PromptInjectionGuard

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "eval" / "deepset_prompt_injections.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "eval" / "deepset_prompt_injections_benchmark.json"

_WORKER_PROMPT_GUARD: PromptInjectionGuard | None = None
_WORKER_E5_GUARD: E5SafetyGuard | None = None


def download_deepset_dataset(output_path: Path) -> list[dict[str, Any]]:
    """Fetch train and test splits of deepset/prompt-injections from HF Datasets Server."""
    rows: list[dict[str, Any]] = []
    base_url = "https://datasets-server.huggingface.co/rows"
    with httpx.Client(timeout=30.0) as client:
        for split in ("test", "train"):
            offset = 0
            length = 100
            while True:
                resp = client.get(
                    base_url,
                    params={
                        "dataset": "deepset/prompt-injections",
                        "config": "default",
                        "split": split,
                        "offset": offset,
                        "length": length,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                batch = payload.get("rows", [])
                if not batch:
                    break
                for item in batch:
                    r = item["row"]
                    rows.append(
                        {
                            "id": f"{split}-{item['row_idx']}",
                            "split": split,
                            "text": r["text"],
                            "label": int(r["label"]),
                            "expected_is_injection": bool(int(r["label"]) == 1),
                        }
                    )
                offset += len(batch)
                if offset >= payload.get("num_rows_total", 0):
                    break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return download_deepset_dataset(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _init_worker() -> None:
    global _WORKER_PROMPT_GUARD, _WORKER_E5_GUARD
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    _WORKER_PROMPT_GUARD = PromptInjectionGuard(device="cpu")
    _WORKER_E5_GUARD = E5SafetyGuard(device="cpu")
    _WORKER_PROMPT_GUARD.load()
    _WORKER_E5_GUARD.load()
    # Warmup
    _WORKER_PROMPT_GUARD.evaluate("Warmup sentence.")
    _WORKER_E5_GUARD.evaluate("Warmup sentence.")


def _eval_row(row: dict[str, Any]) -> dict[str, Any]:
    assert _WORKER_PROMPT_GUARD is not None
    assert _WORKER_E5_GUARD is not None

    text = row["text"]
    t0 = time.perf_counter()
    pg_res = _WORKER_PROMPT_GUARD.evaluate(text)
    e5_res = _WORKER_E5_GUARD.evaluate(text)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    e5_caught = (
        e5_res.risk_category == "prompt_injection"
        and e5_res.action in {"hard_block", "soft_flag"}
    )
    hybrid_caught = pg_res.is_injection or e5_caught
    if pg_res.is_injection:
        blocked_by = f"prompt_guard:{pg_res.label}"
    elif e5_caught:
        blocked_by = "e5:prompt_injection"
    else:
        blocked_by = "allow"

    return {
        **row,
        "pg_is_injection": pg_res.is_injection,
        "pg_label": pg_res.label,
        "pg_score": round(pg_res.score, 6),
        "pg_latency_ms": round(pg_res.latency_ms, 3),
        "e5_is_injection": e5_caught,
        "e5_action": e5_res.action,
        "e5_top_category": e5_res.top_category,
        "e5_margin": round(e5_res.margin, 6),
        "e5_latency_ms": round(e5_res.latency_ms, 3),
        "hybrid_is_injection": hybrid_caught,
        "blocked_by": blocked_by,
        "total_latency_ms": total_ms,
    }


def _compute_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for r in rows:
        gt = bool(r["expected_is_injection"])
        pred = bool(r[pred_key])
        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif not gt and not pred:
            tn += 1
        else:
            fn += 1
    total = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "mean": round(statistics.mean(ordered), 2),
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[min(n - 1, int(0.95 * n))], 2),
        "p99": round(ordered[min(n - 1, int(0.99 * n))], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--split", choices=("test", "train", "all"), default="all")
    args = parser.parse_args()

    all_rows = load_dataset(args.data)
    rows = [r for r in all_rows if r["split"] == args.split] if args.split != "all" else all_rows

    print(
        f"Evaluating {len(rows)} rows (split={args.split}) with {args.workers} worker processes..."
    )
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        results = list(pool.map(_eval_row, rows, chunksize=8))
    wall_seconds = round(time.perf_counter() - t0, 2)

    test_rows = [r for r in results if r["split"] == "test"]
    train_rows = [r for r in results if r["split"] == "train"]

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(subset),
            "prompt_guard_only": _compute_metrics(subset, "pg_is_injection"),
            "e5_only": _compute_metrics(subset, "e5_is_injection"),
            "hybrid_prompt_endpoint": _compute_metrics(subset, "hybrid_is_injection"),
            "latency_ms": {
                "prompt_guard": _latency_summary([r["pg_latency_ms"] for r in subset]),
                "e5_guard": _latency_summary([r["e5_latency_ms"] for r in subset]),
                "combined_sequential_equiv": _latency_summary(
                    [r["total_latency_ms"] for r in subset]
                ),
            },
        }

    false_positives = [
        {
            "id": r["id"],
            "split": r["split"],
            "blocked_by": r["blocked_by"],
            "pg_score": r["pg_score"],
            "e5_top_category": r["e5_top_category"],
            "e5_margin": r["e5_margin"],
            "text": r["text"],
        }
        for r in results
        if not r["expected_is_injection"] and r["hybrid_is_injection"]
    ]
    false_negatives = [
        {
            "id": r["id"],
            "split": r["split"],
            "pg_score": r["pg_score"],
            "e5_top_category": r["e5_top_category"],
            "e5_margin": r["e5_margin"],
            "text": r["text"],
        }
        for r in results
        if r["expected_is_injection"] and not r["hybrid_is_injection"]
    ]

    report = {
        "dataset": "deepset/prompt-injections",
        "workers": args.workers,
        "wall_seconds": wall_seconds,
        "summary": {
            "all": summarize(results),
            **({"test": summarize(test_rows)} if test_rows else {}),
            **({"train": summarize(train_rows)} if train_rows else {}),
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predictions": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done in {wall_seconds}s. Saved results to {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
