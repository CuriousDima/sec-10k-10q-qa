#!/usr/bin/env python3
"""SIA-compatible evaluator for the SIA-FinCheck task.

SIA calls this script as:

    python evaluate.py --gen-dir runs/run_X/gen_Y

The script finds a JSONL prediction file in the generation directory, evaluates it
against private labels, and writes gen_dir/results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASK_DIR = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = TASK_DIR / "data" / "public"
PRIVATE_DIR = TASK_DIR / "data" / "private"

PREDICTION_CANDIDATES = (
    "submission.jsonl",
    "predictions.jsonl",
    "answers.jsonl",
    "output.jsonl",
    "results/submission.jsonl",
    "results/predictions.jsonl",
    "results/answers.jsonl",
)


def load_jsonl_with_validity(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    valid = 0
    total = 0
    if not path.exists():
        return rows, valid, total
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                    valid += 1
            except json.JSONDecodeError:
                pass
    return rows, valid, total


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows, _, _ = load_jsonl_with_validity(path)
    return rows


def parse_number(value: Any, unit: str | None = None) -> float | None:
    """Parse numbers with commas, $, (), %, and million/billion scale words."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        neg = False
        if re.fullmatch(r"\(.*\)", text):
            neg = True
            text = text[1:-1]
        text = text.replace("$", "").replace(",", "").replace("%", "")
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        num = float(match.group(0))
        if neg and num > 0:
            num = -num

    scale_text = f"{value} {unit or ''}".lower()
    if re.search(r"\b(billion|billions|bn)\b", scale_text):
        num *= 1_000_000_000
    elif re.search(r"\b(million|millions|mm)\b", scale_text):
        num *= 1_000_000
    elif re.search(r"\b(thousand|thousands)\b", scale_text):
        num *= 1_000
    return num if math.isfinite(num) else None


def normalize_unit(unit: Any) -> str:
    if unit is None:
        return ""
    normalized = str(unit).strip().lower().replace(" ", "_").replace("$", "usd")
    aliases = {
        "dollars": "usd",
        "dollar": "usd",
        "us_dollars": "usd",
        "usd_million": "usd_millions",
        "usd_mm": "usd_millions",
        "millions_usd": "usd_millions",
        "usd_billion": "usd_billions",
        "billions_usd": "usd_billions",
        "%": "percent",
        "percentage": "percent",
        "percentage_points": "percent",
        "usd/share": "usd_per_share",
        "usd_per_shares": "usd_per_share",
        "dollars_per_share": "usd_per_share",
    }
    return aliases.get(normalized, normalized)


def unit_correct(pred_unit: Any, canonical_unit: str) -> bool:
    pred = normalize_unit(pred_unit)
    canon = normalize_unit(canonical_unit)
    if pred == canon:
        return True
    if canon == "usd" and pred in {"usd_millions", "usd_billions", "usd_thousands"}:
        return True
    if canon == "percent" and pred in {"percentage_point", "percentage_points"}:
        return True
    return False


def numeric_correct(pred: float | None, label: dict[str, Any]) -> bool:
    if pred is None:
        return False
    truth = float(label["canonical_answer"])
    abs_tol = float(label.get("tolerance_abs", 0.0))
    rel_tol = float(label.get("tolerance_rel", 0.0))
    tolerance = max(abs_tol, abs(truth) * rel_tol)
    return abs(pred - truth) <= tolerance


def derived_sanity(pred: float | None, label: dict[str, Any]) -> bool:
    if label.get("formula") is None:
        return True
    if pred is None:
        return False
    answer_type = label.get("answer_type")
    if answer_type == "percent":
        return -10000 <= pred <= 10000
    if answer_type == "ratio":
        return -100 <= pred <= 100
    return math.isfinite(pred)


def find_predictions(gen_dir: Path) -> Path | None:
    for rel in PREDICTION_CANDIDATES:
        candidate = gen_dir / rel
        if candidate.is_file():
            return candidate
    jsonl_files = sorted(gen_dir.glob("*.jsonl"))
    if jsonl_files:
        return max(jsonl_files, key=lambda p: p.stat().st_mtime)
    results_dir = gen_dir / "results"
    if results_dir.is_dir():
        jsonl_files = sorted(results_dir.glob("*.jsonl"))
        if jsonl_files:
            return max(jsonl_files, key=lambda p: p.stat().st_mtime)
    return None


def score_predictions(predictions_path: Path, labels_path: Path, public_path: Path | None = None) -> dict[str, Any]:
    pred_rows, valid_json, total_lines = load_jsonl_with_validity(predictions_path)
    labels = {row["id"]: row for row in load_jsonl(labels_path)}
    public = {row["id"]: row for row in load_jsonl(public_path)} if public_path else {}
    preds_by_id = {row.get("id"): row for row in pred_rows if "id" in row}

    details = []
    missing_required = 0
    for example_id, label in labels.items():
        pred = preds_by_id.get(example_id)
        valid_format = pred is not None and all(key in pred for key in ("id", "answer", "unit"))
        if not valid_format:
            missing_required += 1
        pred_num = parse_number(pred.get("answer"), pred.get("unit")) if pred else None
        n_ok = numeric_correct(pred_num, label)
        u_ok = unit_correct(pred.get("unit") if pred else None, label.get("canonical_unit", ""))
        s_ok = derived_sanity(pred_num, label)
        item_score = 0.80 * float(n_ok) + 0.10 * float(u_ok) + 0.05 * float(valid_format) + 0.05 * float(s_ok)
        meta = public.get(example_id, {})
        details.append(
            {
                "id": example_id,
                "score": item_score,
                "is_correct": n_ok,
                "numeric_correct": n_ok,
                "unit_correct": u_ok,
                "valid_format": valid_format,
                "derived_sanity": s_ok,
                "predicted_number": pred_num,
                "predicted_unit": pred.get("unit") if pred else None,
                "answer_type": label.get("answer_type", meta.get("answer_type", "unknown")),
                "form": meta.get("form", "unknown"),
                "domain": meta.get("answer_type", label.get("answer_type", "unknown")),
                "ticker": meta.get("ticker"),
            }
        )

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_score: dict[str, defaultdict[str, list[float]]] = {
        "score_by_question_type": defaultdict(list),
        "score_by_form_type": defaultdict(list),
    }
    for item in details:
        by_score["score_by_question_type"][item["answer_type"]].append(item["score"])
        by_score["score_by_form_type"][item["form"]].append(item["score"])

    overall_score = avg([item["score"] for item in details])
    numeric_accuracy = avg([float(item["numeric_correct"]) for item in details])
    unit_accuracy = avg([float(item["unit_correct"]) for item in details])

    return {
        "status": "success",
        "predictions_path": str(predictions_path),
        "overall_score": overall_score,
        "accuracy": overall_score,
        "accuracy_percent": overall_score * 100,
        "numeric_accuracy": numeric_accuracy,
        "unit_accuracy": unit_accuracy,
        "valid_json_rate": (valid_json / total_lines) if total_lines else 0.0,
        "valid_output_format_rate": avg([float(item["valid_format"]) for item in details]),
        "derived_sanity_rate": avg([float(item["derived_sanity"]) for item in details]),
        "total": len(labels),
        "correct": sum(1 for item in details if item["numeric_correct"]),
        "n_labels": len(labels),
        "n_prediction_lines": total_lines,
        "n_valid_json_lines": valid_json,
        "n_matched_predictions": sum(1 for example_id in labels if example_id in preds_by_id),
        "n_missing_required_fields": missing_required,
        "score_by_question_type": {key: avg(vals) for key, vals in by_score["score_by_question_type"].items()},
        "score_by_form_type": {key: avg(vals) for key, vals in by_score["score_by_form_type"].items()},
        "counts_by_question_type": dict(Counter(item["answer_type"] for item in details)),
        "counts_by_form_type": dict(Counter(item["form"] for item in details)),
        "details": details,
    }


def missing_predictions_result(gen_dir: Path, labels_path: Path) -> dict[str, Any]:
    labels = load_jsonl(labels_path)
    return {
        "status": "error",
        "reason": "No prediction JSONL file found. Expected submission.jsonl or predictions.jsonl in generation directory.",
        "gen_dir": str(gen_dir),
        "overall_score": 0.0,
        "accuracy": 0.0,
        "accuracy_percent": 0.0,
        "numeric_accuracy": 0.0,
        "unit_accuracy": 0.0,
        "valid_json_rate": 0.0,
        "valid_output_format_rate": 0.0,
        "derived_sanity_rate": 0.0,
        "total": len(labels),
        "correct": 0,
        "n_labels": len(labels),
        "n_prediction_lines": 0,
        "n_valid_json_lines": 0,
        "n_matched_predictions": 0,
        "n_missing_required_fields": len(labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SIA-FinCheck JSONL predictions")
    parser.add_argument("--gen-dir", type=Path, help="SIA generation directory containing submission.jsonl")
    parser.add_argument("--predictions", type=Path, help="Direct path to prediction JSONL")
    parser.add_argument("--labels", type=Path, help="Direct path to private labels JSONL")
    parser.add_argument("--public", type=Path, help="Direct path to matching public JSONL")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", type=Path, help="Output JSON path; defaults to gen_dir/results.json")
    args = parser.parse_args()

    labels_path = args.labels or (PRIVATE_DIR / f"{args.split}_labels.jsonl")
    public_path = args.public or (PUBLIC_DIR / f"{args.split}.jsonl")

    if args.predictions:
        predictions_path = args.predictions
        result = score_predictions(predictions_path, labels_path, public_path)
        output_path = args.output
    elif args.gen_dir:
        predictions_path = find_predictions(args.gen_dir)
        if predictions_path is None:
            result = missing_predictions_result(args.gen_dir, labels_path)
        else:
            result = score_predictions(predictions_path, labels_path, public_path)
        output_path = args.output or (args.gen_dir / "results.json")
    else:
        parser.error("Provide either --gen-dir or --predictions")

    text = json.dumps(result, indent=2, sort_keys=True)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved results to: {output_path}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
