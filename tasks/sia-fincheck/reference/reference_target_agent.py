#!/usr/bin/env python3
"""Reference target agent for SIA-FinCheck.

This is a deliberately simple SIA-compatible seed. It reads `test.jsonl` from the
read-only dataset directory, asks an OpenAI-compatible model for each independent
example, writes `submission.jsonl` in the working directory, and records one
trajectory per example under `agent_execution/`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - lets the file be inspected without deps
    OpenAI = None  # type: ignore[assignment]

MODEL = os.getenv("SIA_TARGET_MODEL", "fugu-mini")
BASE_URL = os.getenv("SAKANA_BASE_URL", "https://api.sakana.ai/v1")
API_KEY_ENV = "SAKANA_API_KEY"
MAX_CONTEXT_CHARS = int(os.getenv("SIA_FINCHECK_MAX_CONTEXT_CHARS", "24000"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def question_terms(question: str) -> set[str]:
    stop = {
        "what", "were", "was", "the", "company", "companies", "for", "and", "of", "as", "at",
        "to", "in", "on", "ended", "year", "quarter", "fiscal", "total", "end", "did", "from",
    }
    return {tok for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower()) if tok not in stop}


def context_excerpt(context: str, question: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Return a compact excerpt biased toward lines relevant to the question."""
    if len(context) <= max_chars:
        return context

    terms = question_terms(question)
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        low = line.lower()
        score = sum(3 for term in terms if term in low)
        score += min(len(re.findall(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?", line)), 5)
        if score:
            scored.append((score, idx, line))

    selected_indices = set()
    for _score, idx, _line in sorted(scored, reverse=True)[:120]:
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected_indices.add(j)

    pieces = []
    # Always include beginning of the filing excerpt for metadata/table context.
    prefix = "\n".join(lines[:80])
    pieces.append(prefix[:4000])
    if selected_indices:
        pieces.append("\n--- relevant excerpt lines ---")
        for idx in sorted(selected_indices):
            pieces.append(lines[idx])

    excerpt = "\n".join(pieces)
    return excerpt[:max_chars]


def make_prompt(example: dict[str, Any]) -> str:
    excerpt = context_excerpt(example.get("context", ""), example.get("question", ""))
    return f"""You are answering one SIA-FinCheck numerical QA example from an SEC filing excerpt.

Return JSON only with keys: answer, unit, confidence, reasoning.

Normalization rules:
- Currency answers should be raw USD numbers. If the filing says values are in millions, multiply by 1,000,000.
- Percent answers are percentage points, e.g. 12.5 for 12.5%.
- Per-share answers use unit USD/share.
- Ratios use unit ratio.
- If uncertain, make the best evidence-based estimate from the supplied context.

Metadata:
ID: {example.get('id')}
Company: {example.get('company_name')} ({example.get('ticker')})
Form: {example.get('form')}
Report date: {example.get('report_date')}
Expected unit: {example.get('expected_unit')}
Answer type: {example.get('answer_type')}
Question: {example.get('question')}

Filing context excerpt:
{excerpt}
"""


def parse_model_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"answer": None, "unit": "", "confidence": 0.0, "reasoning": text[:500]}


def fallback_prediction(example: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": example.get("id"),
        "answer": 0,
        "unit": example.get("expected_unit", ""),
        "confidence": 0.0,
        "reasoning": f"Fallback placeholder because model call failed or was unavailable: {reason}",
    }


def model_client() -> Any | None:
    api_key = os.getenv(API_KEY_ENV) or os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def answer_one(example: dict[str, Any], client: Any | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = make_prompt(example)
    messages = [{"role": "user", "content": prompt}]
    trajectory: list[dict[str, Any]] = [{"role": "user", "content": prompt[:8000]}]

    if client is None:
        pred = fallback_prediction(example, f"missing {API_KEY_ENV} or openai package")
        trajectory.append({"role": "assistant", "content": json.dumps(pred)})
        return pred, trajectory

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=700,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_model_json(content)
        pred = {
            "id": example.get("id"),
            "answer": parsed.get("answer"),
            "unit": parsed.get("unit") or example.get("expected_unit", ""),
            "confidence": parsed.get("confidence", 0.5),
            "reasoning": parsed.get("reasoning", ""),
        }
        trajectory.append({"role": "assistant", "content": content})
        return pred, trajectory
    except Exception as exc:
        pred = fallback_prediction(example, str(exc))
        trajectory.append({"role": "assistant", "content": json.dumps(pred)})
        return pred, trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Read-only public dataset directory")
    parser.add_argument("--working_dir", required=True, help="Writable SIA generation directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    exec_dir = working_dir / "agent_execution"
    exec_dir.mkdir(exist_ok=True)

    examples = load_jsonl(dataset_dir / "test.jsonl")
    client = model_client()
    predictions = []

    print(f"Loaded {len(examples)} SIA-FinCheck test examples")
    print(f"Model: {MODEL}; base_url: {BASE_URL}; client_available={client is not None}")

    for idx, example in enumerate(examples):
        start = time.time()
        pred, trajectory = answer_one(example, client)
        predictions.append(pred)
        (exec_dir / f"execution_q{idx}.json").write_text(json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{idx + 1}/{len(examples)}] {example.get('id')} -> {pred.get('answer')} {pred.get('unit')} ({time.time() - start:.1f}s)")

    write_jsonl(working_dir / "submission.jsonl", predictions)
    summary = {
        "model": MODEL,
        "n_predictions": len(predictions),
        "submission_path": str(working_dir / "submission.jsonl"),
    }
    (working_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {working_dir / 'submission.jsonl'}")


if __name__ == "__main__":
    main()
