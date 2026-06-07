#!/usr/bin/env python3
"""SIA-FinCheck target agent backed by Sakana AI's `fugu-mini` model.

For every record in ``test.jsonl`` this agent:

1. Builds a focused excerpt of the filing context that emphasises the section
   most likely to contain the requested number.
2. Asks ``fugu-mini`` (via the OpenAI-compatible Sakana API) to extract or
   derive a single numeric answer with the correct unit, returning strict
   JSON.
3. Normalises the model's answer so the SIA evaluator (which tolerates
   commas/$/parentheses/% but prefers clean numerics) can score it.
4. Writes ``submission.jsonl`` to ``--working_dir`` and saves one trajectory
   per example under ``<working_dir>/agent_execution/execution_q<i>.json``.

The script accepts ``--dataset_dir`` (read-only) and ``--working_dir``
(read/write) on the command line. Both paths are also passed verbatim into
the prompt so the model is fully aware of the sandbox boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import guard
    OpenAI = None  # type: ignore[assignment]


MODEL = os.getenv("SIA_TARGET_MODEL", "fugu-mini")
BASE_URL = os.getenv("SAKANA_BASE_URL", "https://api.sakana.ai/v1")
API_KEY_ENV = "SAKANA_API_KEY"

MAX_CONTEXT_CHARS = int(os.getenv("SIA_FINCHECK_MAX_CONTEXT_CHARS", "16000"))
MAX_TOKENS = int(os.getenv("SIA_FINCHECK_MAX_TOKENS", "700"))
MAX_WORKERS = int(os.getenv("SIA_FINCHECK_WORKERS", "6"))
MAX_RETRIES = int(os.getenv("SIA_FINCHECK_RETRIES", "3"))


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Context selection
# --------------------------------------------------------------------------- #
_NUMERIC_RE = re.compile(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_/-]{2,}")

_STOPWORDS = {
    "what", "were", "was", "the", "company", "companies", "for", "and", "of", "as",
    "at", "to", "in", "on", "ended", "year", "quarter", "fiscal", "total", "end",
    "did", "from", "with", "this", "that", "are", "have", "has", "its", "their",
    "report", "reported", "value", "period", "by",
}

# Bigger weight for words that strongly map to financial tables.
_KEYWORD_BOOSTS = {
    "revenue": 6, "revenues": 6, "sales": 6, "net": 3,
    "assets": 6, "liabilities": 6, "equity": 5, "stockholders": 4,
    "cash": 5, "operating": 4, "income": 5, "loss": 4, "earnings": 6,
    "diluted": 6, "basic": 5, "eps": 8, "per": 3, "share": 5, "shares": 4,
    "gross": 5, "profit": 5, "margin": 5,
    "expenses": 3, "cost": 3, "tax": 3, "interest": 3, "depreciation": 3,
    "investing": 5, "financing": 5, "free": 4, "flow": 4, "cashflow": 6,
    "dividends": 5, "dividend": 5,
    "research": 4, "development": 4, "r&d": 6,
    "current": 3, "noncurrent": 3, "long-term": 3, "longterm": 3, "short-term": 3,
    "goodwill": 4, "inventory": 4, "inventories": 4, "receivable": 4, "payable": 4,
    "ratio": 4,
}


def _question_terms(question: str) -> set[str]:
    tokens = {tok.lower() for tok in _TOKEN_RE.findall(question)}
    return {tok for tok in tokens if tok not in _STOPWORDS}


def _scale_hint(context: str) -> str:
    """Try to surface the units hint near the top of a filing (e.g. 'in millions')."""
    snippet = context[:6000].lower()
    hints: list[str] = []
    for phrase in (
        "amounts in millions",
        "(in millions",
        "in millions of dollars",
        "(in thousands",
        "in thousands",
        "(in billions",
        "in billions",
        "amounts in thousands",
        "amounts in billions",
        "except per share",
    ):
        if phrase in snippet:
            hints.append(phrase)
    return ", ".join(hints)


def context_excerpt(context: str, question: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Compact a long filing excerpt down to lines relevant to the question."""
    context = context or ""
    if len(context) <= max_chars:
        return context

    terms = _question_terms(question)
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        low = line.lower()
        score = 0
        for term in terms:
            if term in low:
                score += 4 + _KEYWORD_BOOSTS.get(term, 0)
        score += 2 * min(len(_NUMERIC_RE.findall(line)), 6)
        # Reward lines beginning with "Total" because they usually appear in
        # balance sheets / income statements.
        if low.startswith("total "):
            score += 5
        if score:
            scored.append((score, idx, line))

    selected_indices: set[int] = set()
    # Keep up to ~160 most relevant lines and one neighbouring line each side
    # so we preserve table context.
    for _score, idx, _line in sorted(scored, reverse=True)[:160]:
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected_indices.add(j)

    pieces: list[str] = []
    prefix = "\n".join(lines[:80])
    pieces.append(prefix[:3500])
    if selected_indices:
        pieces.append("\n--- relevant excerpt lines ---")
        for idx in sorted(selected_indices):
            pieces.append(lines[idx])

    excerpt = "\n".join(pieces)
    return excerpt[:max_chars]


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a careful financial analyst. You read excerpts from SEC 10-K and 10-Q "
    "filings and answer numerical questions with the correct unit. You always reply "
    "with strict JSON containing keys: answer, unit, confidence, reasoning."
)


def _unit_guidance(answer_type: str, expected_unit: str) -> str:
    answer_type = (answer_type or "").strip().lower()
    expected_unit = (expected_unit or "").strip()
    if answer_type == "currency":
        return (
            "Return a raw USD numeric value. If the filing tables say values are 'in "
            "millions' multiply by 1,000,000; if 'in thousands' multiply by 1,000; if "
            "'in billions' multiply by 1,000,000,000. Use 'USD' as the unit. Do not "
            "include commas, currency symbols, or scale words in `answer`."
        )
    if answer_type == "percent":
        return (
            "Return the value in percentage points (e.g. 12.5 for 12.5%). Use 'percent' "
            "as the unit. Do not include the % sign in `answer`."
        )
    if answer_type in {"usd_per_share", "usd/share", "per_share"}:
        return (
            "Return the per-share dollar amount (e.g. 6.11 for $6.11/share). Use "
            "'USD_per_share' as the unit. Do not multiply by share counts."
        )
    if answer_type == "ratio":
        return (
            "Return a plain decimal ratio (e.g. 0.53 for a 53% ratio). Use 'ratio' as "
            "the unit. Do not multiply by 100. Compute derived ratios from the most "
            "recent balance sheet shown in the excerpt."
        )
    # Fallback to whatever expected_unit indicates.
    return (
        f"Use unit '{expected_unit or 'USD'}'. Output a clean numeric value with no "
        "commas, currency symbols, or scale words."
    )


def build_messages(
    example: dict[str, Any],
    dataset_dir: Path,
    working_dir: Path,
) -> list[dict[str, str]]:
    excerpt = context_excerpt(example.get("context", ""), example.get("question", ""))
    scale_hint = _scale_hint(example.get("context", ""))
    guidance = _unit_guidance(
        example.get("answer_type", ""),
        example.get("expected_unit", ""),
    )

    sandbox_note = (
        "Sandbox paths (informational only — you do not access the filesystem yourself):\n"
        f"- READ-ONLY dataset directory: {dataset_dir}\n"
        f"- READ/WRITE working directory: {working_dir}\n"
        "The orchestrating script already loaded `test.jsonl` from the dataset directory "
        "and will write `submission.jsonl` to the working directory after collecting your "
        "answer. You must NOT attempt any other file access."
    )

    user_prompt = f"""{sandbox_note}

You are answering ONE SIA-FinCheck question for an SEC filing excerpt.

Question metadata:
- ID: {example.get('id')}
- Company: {example.get('company_name')} ({example.get('ticker')})
- Form: {example.get('form')}
- Filing date: {example.get('filing_date')}
- Report date: {example.get('report_date')}
- Fiscal year: {example.get('fiscal_year')}; fiscal period: {example.get('fiscal_period')}
- Answer type: {example.get('answer_type')}
- Expected unit: {example.get('expected_unit')}
- Difficulty: {example.get('difficulty')}
- Scale hint detected near front of context: {scale_hint or 'none'}

Unit / normalisation rule for this question:
{guidance}

General rules:
- Use ONLY the filing context below as evidence. Do not invent numbers.
- For 10-Q questions about a specific quarter ended on a given date, pick the column
  whose period header matches that quarter exactly, NOT the year-to-date column.
- For balance-sheet questions about the end of a period, take the most recent
  reported balance (usually the leftmost numeric column under the most recent date).
- For derived questions (e.g., "liabilities-to-assets ratio") compute the ratio from
  the SAME period's totals you see in the balance sheet.
- Confidence is a float in [0, 1] reflecting how sure you are.
- Reasoning should be a single short sentence describing the source line(s) and any
  unit conversion you applied.

Respond with strict JSON only, exactly in this shape:
{{
  "answer": <number>,
  "unit": "<unit string>",
  "confidence": <float 0-1>,
  "reasoning": "<short evidence string>"
}}

Question: {example.get('question')}

Filing context excerpt:
\"\"\"
{excerpt}
\"\"\"
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------------------------------------------------------- #
# Model output parsing & normalisation
# --------------------------------------------------------------------------- #
def _safe_json_loads(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    # Strip ``` fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to grabbing the largest {...} substring.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


_SCALE_WORD_RE = re.compile(
    r"\b(billion|billions|bn|million|millions|mm|thousand|thousands)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _coerce_number(value: Any, unit_hint: str = "") -> float | None:
    """Best-effort conversion of arbitrary model output to a plain float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None

    negative = False
    if re.fullmatch(r"\(.*\)", text):
        negative = True
        text = text[1:-1]

    cleaned = text.replace("$", "").replace(",", "").replace("%", "")
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None
    num = float(match.group(0))
    if negative and num > 0:
        num = -num

    combined = f"{value} {unit_hint or ''}".lower()
    if re.search(r"\bbillion(s)?\b|\bbn\b", combined):
        num *= 1_000_000_000
    elif re.search(r"\bmillion(s)?\b|\bmm\b", combined):
        num *= 1_000_000
    elif re.search(r"\bthousand(s)?\b", combined):
        num *= 1_000
    return num


def _normalise_unit(answer_type: str, expected_unit: str, model_unit: Any) -> str:
    answer_type = (answer_type or "").strip().lower()
    if answer_type == "currency":
        return "USD"
    if answer_type == "percent":
        return "percent"
    if answer_type in {"usd_per_share", "usd/share", "per_share"}:
        return "USD_per_share"
    if answer_type == "ratio":
        return "ratio"
    # Fall back to model-supplied unit then to expected unit.
    if isinstance(model_unit, str) and model_unit.strip():
        return model_unit.strip()
    return expected_unit or ""


def _post_process_answer(
    answer_type: str,
    expected_unit: str,
    raw_answer: Any,
    raw_unit: Any,
) -> float | int | None:
    answer_type = (answer_type or "").strip().lower()
    raw_unit_str = str(raw_unit or "")
    number = _coerce_number(raw_answer, raw_unit_str)
    if number is None:
        return None

    if answer_type == "currency":
        # Some models forget to expand "in millions"; if they returned a small
        # number while declaring a scaled unit, expand here.
        if abs(number) < 1_000_000:
            lowered = raw_unit_str.lower()
            if "billion" in lowered or "bn" in lowered:
                number *= 1_000_000_000
            elif "million" in lowered or "mm" in lowered:
                number *= 1_000_000
            elif "thousand" in lowered:
                number *= 1_000
        return number

    if answer_type == "ratio":
        # If the model gave a percent-style value (e.g. 53 instead of 0.53),
        # rescale only when the unit hints at a percentage.
        if abs(number) > 5 and "%" in raw_unit_str:
            number /= 100.0
        return number

    if answer_type == "percent":
        # Drop stray /100 if model output a fraction with the % unit.
        if abs(number) <= 1 and "%" not in raw_unit_str and "percent" not in raw_unit_str.lower():
            # Probably already in percent if expected unit is percent.
            return number
        return number

    return number


# --------------------------------------------------------------------------- #
# OpenAI client + per-example workflow
# --------------------------------------------------------------------------- #
def make_client() -> Any | None:
    api_key = os.getenv(API_KEY_ENV) or os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def _call_model(
    client: Any,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, int], str]:
    """Call fugu-mini with retries. Returns (content, usage, finish_reason)."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", "") or ""
            usage_obj = getattr(response, "usage", None)
            usage: dict[str, int] = {}
            if usage_obj is not None:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = getattr(usage_obj, key, None)
                    if val is not None:
                        usage[key] = int(val)
            return content, usage, finish_reason
        except Exception as exc:  # noqa: BLE001 - we want to retry on any client error
            last_exc = exc
            err_text = str(exc).lower()
            # If the server rejects json_object response_format, retry without it.
            if "response_format" in err_text and attempt == 1:
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=MAX_TOKENS,
                    )
                    choice = response.choices[0]
                    content = choice.message.content or ""
                    finish_reason = getattr(choice, "finish_reason", "") or ""
                    usage_obj = getattr(response, "usage", None)
                    usage = {}
                    if usage_obj is not None:
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            val = getattr(usage_obj, key, None)
                            if val is not None:
                                usage[key] = int(val)
                    return content, usage, finish_reason
                except Exception as exc2:  # noqa: BLE001
                    last_exc = exc2
            sleep_for = min(2 ** attempt + random.random(), 12)
            time.sleep(sleep_for)
    raise RuntimeError(f"fugu-mini call failed after {MAX_RETRIES} attempts: {last_exc}")


def _fallback_prediction(example: dict[str, Any], reason: str) -> dict[str, Any]:
    answer_type = (example.get("answer_type") or "").lower()
    if answer_type == "currency":
        unit = "USD"
    elif answer_type == "percent":
        unit = "percent"
    elif answer_type in {"usd_per_share", "usd/share", "per_share"}:
        unit = "USD_per_share"
    elif answer_type == "ratio":
        unit = "ratio"
    else:
        unit = example.get("expected_unit") or ""
    return {
        "id": example.get("id"),
        "answer": 0,
        "unit": unit,
        "confidence": 0.0,
        "reasoning": f"Fallback placeholder: {reason}",
    }


def process_example(
    example: dict[str, Any],
    client: Any | None,
    dataset_dir: Path,
    working_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    messages = build_messages(example, dataset_dir, working_dir)
    trajectory: list[dict[str, Any]] = []
    for msg in messages:
        trajectory.append(
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            }
        )

    meta: dict[str, Any] = {
        "id": example.get("id"),
        "model": MODEL,
        "usage": {},
        "cost": 0,
        "finish_reason": "",
        "ok": False,
    }

    if client is None:
        reason = f"missing {API_KEY_ENV} or `openai` package; using zero placeholder"
        pred = _fallback_prediction(example, reason)
        trajectory.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(pred)}],
            }
        )
        meta["error"] = reason
        return pred, trajectory, meta

    try:
        content, usage, finish_reason = _call_model(client, messages)
        meta["usage"] = usage
        meta["finish_reason"] = finish_reason
        parsed = _safe_json_loads(content) or {}

        raw_answer = parsed.get("answer")
        raw_unit = parsed.get("unit") or example.get("expected_unit", "")
        normalised_number = _post_process_answer(
            example.get("answer_type", ""),
            example.get("expected_unit", ""),
            raw_answer,
            raw_unit,
        )
        unit = _normalise_unit(
            example.get("answer_type", ""),
            example.get("expected_unit", ""),
            raw_unit,
        )

        final_answer: Any
        if normalised_number is None:
            final_answer = raw_answer if raw_answer is not None else 0
        else:
            # Keep ints when the value is integral (cleaner for currency).
            if abs(normalised_number - round(normalised_number)) < 1e-6 and abs(normalised_number) >= 1:
                final_answer = int(round(normalised_number))
            else:
                final_answer = float(normalised_number)

        confidence_raw = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        reasoning = parsed.get("reasoning") or ""
        if not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning)

        pred = {
            "id": example.get("id"),
            "answer": final_answer,
            "unit": unit,
            "confidence": confidence,
            "reasoning": reasoning[:600],
        }
        trajectory.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
            }
        )
        meta["ok"] = True
        return pred, trajectory, meta
    except Exception as exc:  # noqa: BLE001
        reason = f"model call failed: {exc}"
        pred = _fallback_prediction(example, reason)
        trajectory.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(pred)}],
            }
        )
        meta["error"] = reason
        return pred, trajectory, meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="SIA-FinCheck target agent (fugu-mini)")
    parser.add_argument("--dataset_dir", required=True, help="Read-only dataset directory")
    parser.add_argument("--working_dir", required=True, help="Writable output directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    working_dir = Path(args.working_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset_dir does not exist: {dataset_dir}", file=sys.stderr)
        return 2
    working_dir.mkdir(parents=True, exist_ok=True)
    exec_dir = working_dir / "agent_execution"
    exec_dir.mkdir(parents=True, exist_ok=True)

    test_path = dataset_dir / "test.jsonl"
    if not test_path.is_file():
        print(f"ERROR: missing test.jsonl in {dataset_dir}", file=sys.stderr)
        return 2

    examples = load_jsonl(test_path)
    print(f"Loaded {len(examples)} test examples from {test_path}")
    print(f"Model: {MODEL}; base_url: {BASE_URL}")

    client = make_client()
    if client is None:
        print(
            f"WARNING: no Sakana credentials available (set ${API_KEY_ENV}); "
            "writing zero fallbacks.",
            file=sys.stderr,
        )

    predictions: list[dict[str, Any] | None] = [None] * len(examples)
    trajectories: list[list[dict[str, Any]] | None] = [None] * len(examples)
    metas: list[dict[str, Any] | None] = [None] * len(examples)

    workers = max(1, MAX_WORKERS) if client is not None else 1
    print_lock = threading.Lock()
    start_time = time.time()

    def worker(idx: int) -> int:
        example = examples[idx]
        item_start = time.time()
        pred, trajectory, meta = process_example(example, client, dataset_dir, working_dir)
        predictions[idx] = pred
        trajectories[idx] = trajectory
        metas[idx] = meta
        # Persist trajectory immediately so we keep partial logs on crash.
        try:
            (exec_dir / f"execution_q{idx}.json").write_text(
                json.dumps(trajectory, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as write_exc:  # noqa: BLE001
            with print_lock:
                print(f"  ! could not write trajectory {idx}: {write_exc}", file=sys.stderr)
        with print_lock:
            elapsed = time.time() - item_start
            print(
                f"[{idx + 1:3d}/{len(examples)}] {example.get('id')} -> "
                f"{pred.get('answer')} {pred.get('unit')} "
                f"(conf={pred.get('confidence'):.2f}, {elapsed:.1f}s)"
            )
        return idx

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, i) for i in range(len(examples))]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! worker failed: {exc}", file=sys.stderr)
    else:
        for i in range(len(examples)):
            try:
                worker(i)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! worker {i} failed: {exc}", file=sys.stderr)

    # Replace any leftover None predictions with fallbacks (paranoia).
    final_predictions: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        if pred is None:
            pred = _fallback_prediction(examples[idx], "no prediction recorded")
            predictions[idx] = pred
            try:
                (exec_dir / f"execution_q{idx}.json").write_text(
                    json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": json.dumps(pred)}],
                            }
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
        final_predictions.append(pred)

    submission_path = working_dir / "submission.jsonl"
    write_jsonl(submission_path, final_predictions)

    # Aggregate metadata summary.
    summary = {
        "model": MODEL,
        "base_url": BASE_URL,
        "dataset_dir": str(dataset_dir),
        "working_dir": str(working_dir),
        "submission_path": str(submission_path),
        "n_examples": len(examples),
        "n_predictions": len(final_predictions),
        "client_available": client is not None,
        "runtime_seconds": round(time.time() - start_time, 2),
        "ok_predictions": sum(1 for m in metas if m and m.get("ok")),
        "failed_predictions": sum(1 for m in metas if m and not m.get("ok")),
        "total_prompt_tokens": sum(
            (m.get("usage", {}) or {}).get("prompt_tokens", 0) for m in metas if m
        ),
        "total_completion_tokens": sum(
            (m.get("usage", {}) or {}).get("completion_tokens", 0) for m in metas if m
        ),
        "total_tokens": sum(
            (m.get("usage", {}) or {}).get("total_tokens", 0) for m in metas if m
        ),
        "cost": 0,
    }
    (working_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"Wrote {submission_path} with {len(final_predictions)} predictions in "
        f"{summary['runtime_seconds']}s "
        f"(ok={summary['ok_predictions']}, failed={summary['failed_predictions']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
