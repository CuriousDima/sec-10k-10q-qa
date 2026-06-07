#!/usr/bin/env python3
"""SIA-FinCheck target agent — generation 2.

This generation keeps generation-1's "heuristic + local-Gemma LLM" pipeline
but fixes the bug that caused every LLM call to fail silently, makes table
row selection more layout-aware, and expands the prose extractor so 10-K cover
pages (which never contain the consolidated statements) still have a shot at
producing the right number.

Inputs
------
* ``--dataset_dir`` (read-only) – contains ``test.jsonl``.
* ``--working_dir`` (read-write) – submission, per-question trajectories, log.

Outputs
-------
* ``<working_dir>/submission.jsonl`` – one JSON object per question.
* ``<working_dir>/agent_execution/execution_q{i}.json`` – per-question trace
  in the SIA "trajectory" format used by the harness.
* ``<working_dir>/summary.json`` – source counts and run metadata.
* ``<working_dir>/agent.log`` – plain-text run log (also goes to stdout).

Sandbox rules enforced here:
* The agent never reads anything outside ``--dataset_dir``.
* The agent never writes anything outside ``--working_dir``.
* The agent never modifies ``--dataset_dir``.
* The only LLM endpoint used is the local checkpoint at
  ``LOCAL_GEMMA_MODEL_PATH`` (default ``/workspace/gemma_checkpoints/gemma-4-31B-it``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Configuration knobs (all overridable through environment variables).
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "/workspace/gemma_checkpoints/gemma-4-31B-it"
LOCAL_GEMMA_MODEL_PATH = os.getenv("LOCAL_GEMMA_MODEL_PATH", DEFAULT_MODEL_PATH)

MAX_CONTEXT_CHARS = int(os.getenv("SIA_FINCHECK_MAX_CONTEXT_CHARS", "4500"))
MAX_NEW_TOKENS = int(os.getenv("SIA_FINCHECK_MAX_NEW_TOKENS", "160"))
MAX_SEQ_LEN = int(os.getenv("SIA_FINCHECK_MAX_SEQ_LEN", "4096"))
USE_LLM = os.getenv("SIA_FINCHECK_USE_LLM", "1") not in {"0", "false", "False"}
# Total wall-clock budget (seconds) for LLM calls across all questions.  After
# this, remaining questions skip the LLM and rely on the heuristic alone.  Set
# to <=0 to disable the budget.  Default of 1500s (25 min) keeps us safely
# inside typical harness budgets while allowing ~25 LLM calls at ~60s each.
LLM_TOTAL_BUDGET_SEC = float(os.getenv("SIA_FINCHECK_LLM_BUDGET_SEC", "1500"))
# When True, only call the LLM for questions where the heuristic returned no
# value (i.e. the prediction would otherwise be the conservative placeholder).
LLM_ONLY_ON_MISS = os.getenv("SIA_FINCHECK_LLM_ONLY_ON_MISS", "1") not in {"0", "false", "False"}
FLUSH_EVERY = int(os.getenv("SIA_FINCHECK_FLUSH_EVERY", "5"))

STOPWORDS = {
    "what", "were", "was", "the", "company", "companies", "for", "and", "of", "as",
    "at", "to", "in", "on", "ended", "year", "quarter", "fiscal", "total", "end",
    "did", "from", "with", "that", "this", "its", "their", "amount", "value",
    "reported", "report", "during", "how", "many", "much", "is", "are", "be",
    "company's", "a", "an", "by", "per", "period", "reporting", "answer",
    "percentage",
}


def _log_msg(log_file: Path | None, msg: str) -> None:
    """Append a timestamped message to ``agent.log`` and stdout."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        try:
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Intent detection.
# ---------------------------------------------------------------------------

# Ordered (specific first) regex patterns matched against the LABEL portion of
# a `Label | num | num` style table row.
INTENT_LABELS: dict[str, list[str]] = {
    "revenue": [
        r"^total\s+net\s+sales$",
        r"^net\s+sales$",
        r"^total\s+(?:net\s+)?revenues?$",
        r"^net\s+revenues?$",
        r"^revenues?,?\s+net$",
        r"^revenues?$",
        r"^total\s+revenues?\s+(?:and\s+other(?:\s+income)?)?$",
        r"^net\s+interest\s+income$",
    ],
    "operating_income": [
        r"^operating\s+income(?:\s+\(loss\))?$",
        r"^income\s+from\s+operations(?:\s+\(loss\))?$",
        r"^operating\s+(?:earnings|profit)$",
        r"^operating\s+income\s+\(loss\)?$",
    ],
    "gross_profit": [
        r"^gross\s+profit$",
        r"^gross\s+margin$",
    ],
    "net_income": [
        r"^net\s+income(?:/?\s*\(loss\))?$",
        r"^net\s+income/?\(loss\)/?$",
        r"^net\s+income\s+attributable\s+to.+$",
        r"^net\s+earnings(?:/?\s*\(loss\))?$",
        r"^net\s+earnings\s+attributable\s+to.+$",
        r"^net\s+earnings\s+common\s+stockholders$",
        r"^net\s+income\s+\(loss\)\s+attributable\s+to.+$",
    ],
    "assets": [
        r"^total\s+assets$",
    ],
    "liabilities": [
        r"^total\s+liabilities$",
    ],
    # Parent-attributable equity rows (preferred for "stockholders' equity"
    # questions because they exclude noncontrolling interests).  Any pattern
    # containing the words "liabilities" or "and" is excluded by post-filter
    # below so we never confuse "Total Liabilities and Stockholders' Equity"
    # (which equals Total Assets) with the equity row itself.
    "stockholders_equity_parent": [
        r"^total\s+(?:stockholders'?|shareholders'?)\s+equity\s+attributable\s+to.+$",
        r"^(?:stockholders'?|shareholders'?)\s+equity\s+attributable\s+to.+$",
        # "Total IBM stockholders' equity" / "Total Chevron Corporation
        # Stockholders' Equity" / "Total The Bank of New York Mellon
        # Corporation common shareholders' equity" — any company-name tokens
        # between "Total" and "Stockholders'/Shareholders' Equity", optionally
        # with a "common" qualifier just before "stockholders'/shareholders'".
        r"^total\s+(?:[a-z][\w\.\-]*\.?\s+){1,8}(?:common\s+)?(?:stockholders'?|shareholders'?)\s+equity$",
        r"^total\s+(?:stockholders'?|shareholders'?)\s+equity$",
        r"^total\s+common\s+(?:stockholders'?|shareholders'?)\s+equity$",
    ],
    # Generic equity rows (may include noncontrolling interest).
    "stockholders_equity_total": [
        r"^total\s+(?:stockholders'?|shareholders'?)\s+equity$",
        r"^total\s+equity$",
    ],
    "diluted_eps": [
        r"^[-•·]?\s*diluted$",
        r"^diluted\s+earnings\s+per\s+(?:common\s+)?share$",
        r"^diluted\s+(?:net\s+income|earnings|loss|net\s+loss)\s+per\s+(?:common\s+)?share$",
        r"^diluted\s+eps$",
        r"^basic\s+and\s+diluted$",
        r"^basic\s+and\s+diluted\s+earnings\s+per\s+(?:common\s+)?share$",
        r"^basic\s+and\s+diluted\s+(?:net\s+income|earnings)\s+per\s+(?:common\s+)?share$",
    ],
    "basic_eps": [
        r"^[-•·]?\s*basic$",
        r"^basic\s+earnings\s+per\s+(?:common\s+)?share$",
        r"^basic\s+(?:net\s+income|earnings|loss|net\s+loss)\s+per\s+(?:common\s+)?share$",
        r"^basic\s+eps$",
        r"^basic\s+and\s+diluted$",
        r"^basic\s+and\s+diluted\s+earnings\s+per\s+(?:common\s+)?share$",
    ],
    "cash_flow_operating": [
        r"^net\s+cash\s+provided\s+by\s+operating\s+activities$",
        r"^net\s+cash\s+provided\s+by\s+\(used\s+in\)\s+operating\s+activities$",
        r"^net\s+cash\s+used\s+in\s+operating\s+activities$",
        r"^cash\s+flows?\s+from\s+operating\s+activities$",
    ],
    "cash_flow_investing": [
        r"^net\s+cash\s+(?:provided\s+by|used\s+in)\s+investing\s+activities$",
        r"^net\s+cash\s+(?:provided\s+by/?\(?used\s+in\)?|used\s+in/?\(?provided\s+by\)?)\s+investing\s+activities$",
    ],
    "cash_flow_financing": [
        r"^net\s+cash\s+(?:provided\s+by|used\s+in)\s+financing\s+activities$",
        r"^net\s+cash\s+(?:provided\s+by/?\(?used\s+in\)?|used\s+in/?\(?provided\s+by\)?)\s+financing\s+activities$",
    ],
    "cash": [
        r"^cash\s+and\s+cash\s+equivalents$",
        r"^total\s+cash,\s+cash\s+equivalents.+$",
    ],
}

# Logical alias `stockholders_equity` is resolved at lookup-time using a
# preference chain instead of being a fixed pattern set.
INTENT_LABELS["operating_margin"] = INTENT_LABELS["operating_income"]
INTENT_LABELS["gross_margin"] = INTENT_LABELS["gross_profit"]
INTENT_LABELS["net_margin"] = INTENT_LABELS["net_income"]
INTENT_LABELS["equity_ratio_num"] = (
    INTENT_LABELS["stockholders_equity_parent"]
    + INTENT_LABELS["stockholders_equity_total"]
)
INTENT_LABELS["equity_ratio_den"] = INTENT_LABELS["assets"]
INTENT_LABELS["liabilities_to_assets_num"] = INTENT_LABELS["liabilities"]
INTENT_LABELS["liabilities_to_assets_den"] = INTENT_LABELS["assets"]
INTENT_LABELS["debt_to_equity_num"] = INTENT_LABELS["liabilities"]
INTENT_LABELS["debt_to_equity_den"] = (
    INTENT_LABELS["stockholders_equity_parent"]
    + INTENT_LABELS["stockholders_equity_total"]
)


def question_terms(question: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower())
    return {tok for tok in tokens if tok not in STOPWORDS}


def detect_intent(question: str, example_id: str = "") -> str:
    """Map (question, id) onto a canonical intent string."""
    qlow = (question or "").lower()
    idlow = (example_id or "").lower()
    # Order: more specific first.
    id_slugs = [
        "liabilities_to_assets",
        "equity_ratio",
        "operating_margin",
        "gross_margin",
        "net_margin",
        "diluted_eps",
        "basic_eps",
        "operating_income",
        "gross_profit",
        "net_income",
        "stockholders_equity",
        "cash_flow_operating",
        "cash_flow_investing",
        "cash_flow_financing",
        "assets",
        "liabilities",
        "revenue",
        "equity",
        "cash",
    ]
    for slug in id_slugs:
        if slug in idlow:
            if slug in {"equity", "stockholders_equity"}:
                return "stockholders_equity"
            return slug
    if "diluted" in qlow and ("eps" in qlow or "per share" in qlow):
        return "diluted_eps"
    if "basic" in qlow and ("eps" in qlow or "per share" in qlow):
        return "basic_eps"
    if "operating margin" in qlow:
        return "operating_margin"
    if "gross margin" in qlow:
        return "gross_margin"
    if "net margin" in qlow:
        return "net_margin"
    if "operating income" in qlow or "income from operations" in qlow:
        return "operating_income"
    if "gross profit" in qlow:
        return "gross_profit"
    if "net income" in qlow or "net earnings" in qlow:
        return "net_income"
    if "revenue" in qlow or "sales" in qlow:
        return "revenue"
    if "liabilities-to-assets" in qlow or "liabilities to assets" in qlow:
        return "liabilities_to_assets"
    if "equity ratio" in qlow:
        return "equity_ratio"
    if "total assets" in qlow:
        return "assets"
    if "total liabilities" in qlow:
        return "liabilities"
    if "stockholders' equity" in qlow or "shareholders' equity" in qlow or "stockholders equity" in qlow:
        return "stockholders_equity"
    if "cash flow" in qlow and "operating" in qlow:
        return "cash_flow_operating"
    if "cash flow" in qlow and "investing" in qlow:
        return "cash_flow_investing"
    if "cash flow" in qlow and "financing" in qlow:
        return "cash_flow_financing"
    if "cash and cash equivalents" in qlow:
        return "cash"
    return "unknown"


def intent_chain(intent: str) -> list[str]:
    """Return underlying table-label intents for a (possibly compound) intent."""
    if intent == "stockholders_equity":
        return ["stockholders_equity_parent", "stockholders_equity_total"]
    if intent == "equity_ratio":
        return ["equity_ratio_num", "equity_ratio_den"]
    if intent == "liabilities_to_assets":
        return ["liabilities_to_assets_num", "liabilities_to_assets_den"]
    if intent == "debt_to_equity":
        return ["debt_to_equity_num", "debt_to_equity_den"]
    if intent == "operating_margin":
        return ["operating_margin", "revenue"]
    if intent == "gross_margin":
        return ["gross_margin", "revenue"]
    if intent == "net_margin":
        return ["net_margin", "revenue"]
    return [intent]


# ---------------------------------------------------------------------------
# Numeric parsing & scaling helpers.
# ---------------------------------------------------------------------------


def parse_number_text(text: str) -> float | None:
    """Parse a single numeric value out of a noisy string."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    neg = False
    if raw.startswith("(") and raw.endswith(")"):
        neg = True
        raw = raw[1:-1].strip()
    if raw.startswith("-"):
        neg = not neg
        raw = raw[1:].strip()
    if raw.startswith("$"):
        raw = raw[1:].strip()
    cleaned = raw.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    if neg and value > 0:
        value = -value
    low = str(text).lower()
    if re.search(r"\b(trillion|trillions|tn)\b", low):
        value *= 1_000_000_000_000
    elif re.search(r"\b(billion|billions|bn)\b", low):
        value *= 1_000_000_000
    elif re.search(r"\b(million|millions|mm|mn)\b", low):
        value *= 1_000_000
    elif re.search(r"\b(thousand|thousands|k)\b", low):
        value *= 1_000
    return value if math.isfinite(value) else None


SCALE_TOKEN_RE = re.compile(
    r"(?:\(\s*in\s+billions|\(\s*in\s+millions|\(\s*in\s+thousands"
    r"|amounts?\s+in\s+billions|amounts?\s+in\s+millions|amounts?\s+in\s+thousands"
    r"|dollars\s+in\s+billions|dollars\s+in\s+millions|dollars\s+in\s+thousands"
    r"|millions\s+of\s+dollars|billions\s+of\s+dollars|thousands\s+of\s+dollars"
    r"|\$\s+in\s+millions|\$\s+in\s+billions|\$\s+in\s+thousands"
    r"|in\s+millions|in\s+billions|in\s+thousands)",
    re.IGNORECASE,
)


def _scale_from_text(token: str) -> float:
    low = token.lower()
    if "billion" in low:
        return 1_000_000_000.0
    if "million" in low:
        return 1_000_000.0
    if "thousand" in low:
        return 1_000.0
    return 1.0


def detect_scale_multiplier(context: str) -> float:
    tokens = [m.group(0) for m in SCALE_TOKEN_RE.finditer(context)]
    if not tokens:
        return 1.0
    counts: dict[float, int] = {}
    for tok in tokens:
        s = _scale_from_text(tok)
        counts[s] = counts.get(s, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


SHARES_SCALE_QUALIFIER_RE = re.compile(
    r"(?:shares?|common\s+stock|outstanding|weighted[\s-]*average)\s*(?:[^()]{0,40})?\(\s*in\s+(?:thousands|millions|billions)\b",
    re.IGNORECASE,
)


def _scale_token_relates_to_shares(token_text: str, window: str, token_offset: int) -> bool:
    """Heuristic: ignore an `(in thousands)` token if it's actually about share counts.

    SEC quarterly snippets often include both `Dollars in millions` (for the
    balance sheet) and `(in thousands)` (for share counts).  When we see the
    second one, the surrounding text usually contains `shares`, `outstanding`,
    `weighted-average`, etc.  Treat such tokens as share-only scale markers
    so they don't override the dollar scale.
    """
    look = window[max(0, token_offset - 80) : token_offset + len(token_text) + 60]
    look_low = look.lower()
    if re.search(r"\b(share|shares|outstanding|weighted[\s-]*average|per\s+share)\b", look_low):
        return True
    return False


def scale_near_position(context: str, pos: int, lookback_chars: int = 4000) -> float:
    """Pick the most relevant scale marker within `lookback_chars` before `pos`."""
    start = max(0, pos - lookback_chars)
    window = context[start:pos]
    # Prefer the first scale token inside a recent header-style parenthetical
    # that is NOT obviously about shares.
    for paren in reversed(list(re.finditer(r"\(([^)]{0,200})\)", window))):
        inner = paren.group(1)
        primary = SCALE_TOKEN_RE.search(inner)
        if not primary:
            continue
        # If this parenthetical mentions "shares" / "per share", skip it for
        # dollar-amount scale purposes.
        if re.search(r"\b(share|shares|outstanding|per\s+share)\b", inner, re.IGNORECASE):
            continue
        return _scale_from_text(primary.group(0))
    # Fall back to the most recent free-standing scale token that is not
    # share-qualified.
    last_match: str | None = None
    for match in SCALE_TOKEN_RE.finditer(window):
        if _scale_token_relates_to_shares(match.group(0), window, match.start()):
            continue
        last_match = match.group(0)
    if last_match:
        return _scale_from_text(last_match)
    return detect_scale_multiplier(context)


# ---------------------------------------------------------------------------
# Context excerpting (keeps prompts short for the LLM).
# ---------------------------------------------------------------------------


def context_excerpt(
    context: str,
    question: str,
    intent: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    if not context:
        return ""
    if len(context) <= max_chars:
        return context
    intents = intent_chain(intent)
    label_words: set[str] = set()
    for key in intents:
        for pat in INTENT_LABELS.get(key, []):
            for word in re.findall(r"[a-z]+", pat):
                if len(word) > 2:
                    label_words.add(word)
    terms = question_terms(question)
    lines = [line.rstrip() for line in context.splitlines()]
    scored: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        low = line.lower()
        score = 0
        score += 8 * sum(1 for word in label_words if word in low)
        score += 2 * sum(1 for term in terms if term in low)
        if "|" in line and re.search(r"\d", line):
            score += 4
        if "$" in line:
            score += 1
        if "(in millions" in low or "(in thousands" in low or "(in billions" in low:
            score += 5
        if "dollars in millions" in low or "dollars in billions" in low:
            score += 5
        if "consolidated" in low and ("statement" in low or "balance" in low):
            score += 4
        if score:
            scored.append((score, idx))
    selected: set[int] = set(range(min(15, len(lines))))  # always keep header
    for _score, idx in sorted(scored, reverse=True)[:200]:
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected.add(j)
    pieces = [lines[i] for i in sorted(selected)]
    excerpt = "\n".join(pieces)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    return excerpt


# ---------------------------------------------------------------------------
# Table-row extraction.
# ---------------------------------------------------------------------------

NUMERIC_CELL_RE = re.compile(
    r"""
    ^\s*
    \$?\s*
    (
        \(\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*\)         # (1,234) parenthesized negative
        |
        -?\s*\d[\d,]*(?:\.\d+)?                    # plain (possibly negative) number
    )
    \s*%?\s*$
    """,
    re.VERBOSE,
)


def _normalize_label(text: str) -> str:
    """Lower-case a row label and collapse Unicode apostrophes / footnote tags."""
    text = (text or "").strip().lower()
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
        .replace("`", "'")
    )
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s*\*+\s*$", "", text)
    text = re.sub(r"^[-•·\u2013\u2014]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(":"):
        text = text[:-1].strip()
    return text


def split_table_row(line: str) -> tuple[str, list[float]]:
    """Split a `Label | n1 | n2 | ...` row into a label and numeric cells."""
    if "|" not in line:
        return "", []
    parts = [p.strip() for p in line.split("|")]
    if not parts:
        return "", []
    label = _normalize_label(parts[0])
    if not label:
        return "", []
    cells: list[float] = []
    for cell in parts[1:]:
        if not cell or cell in {"$", "—", "-", "(in millions)"}:
            continue
        match = NUMERIC_CELL_RE.match(cell)
        if match:
            value = parse_number_text(match.group(1))
        else:
            inner = re.sub(r"[^\d\.\-\(\)]", " ", cell).strip()
            if not inner:
                continue
            value = parse_number_text(cell)
        if value is not None and math.isfinite(value):
            cells.append(value)
    return label, cells


def find_table_rows(context: str) -> list[tuple[str, list[float], int]]:
    rows: list[tuple[str, list[float], int]] = []
    offset = 0
    for line in context.splitlines(keepends=True):
        label, cells = split_table_row(line.rstrip("\n"))
        if cells:
            rows.append((label, cells, offset))
        offset += len(line)
    return rows


def _row_looks_like_eps(cells: list[float]) -> bool:
    if not cells:
        return False
    nonzero = [c for c in cells if c != 0]
    if not nonzero:
        return False
    return max(abs(c) for c in nonzero) < 100


def _row_looks_like_share_counts(cells: list[float]) -> bool:
    if not cells:
        return False
    nonzero = [c for c in cells if c != 0]
    if not nonzero:
        return False
    if all(c == int(c) for c in nonzero) and min(abs(c) for c in nonzero) >= 100:
        return True
    if min(abs(c) for c in nonzero) >= 500:
        return True
    return False


def _parse_year_header_line(line: str) -> list[int] | None:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    years: list[int] = []
    for p in parts:
        m = re.match(r"^(?:\$\s*)?(\d{4})\b", p)
        if not m:
            return None
        year = int(m.group(1))
        if not (1990 <= year <= 2100):
            return None
        years.append(year)
    if 2 <= len(years) <= 4:
        return years
    return None


def column_headers_above(
    context: str, offset: int, lookback_chars: int = 4000
) -> tuple[list[str], list[int] | None]:
    start = max(0, offset - lookback_chars)
    window = context[start:offset]
    lines = window.splitlines()
    period_line_idx: int | None = None
    period_parts: list[str] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if "|" in line and (
            "months ended" in low
            or "weeks ended" in low
            or "year ended" in low
            or "years ended" in low
            or "quarter ended" in low
            or "fiscal year" in low
        ):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                period_line_idx = i
                period_parts = parts
    year_parts: list[int] | None = None
    if period_line_idx is not None:
        for j in range(period_line_idx + 1, min(period_line_idx + 4, len(lines))):
            yh = _parse_year_header_line(lines[j])
            if yh is not None:
                year_parts = yh
                break
    return period_parts, year_parts


def pick_column(
    cells: list[float],
    column_period_hint: str | None,
    column_header: list[str] | None,
    year_header: list[int] | None = None,
    fiscal_year: int | None = None,
) -> float | None:
    """Choose which numeric cell to return given the question's expected period."""
    if not cells:
        return None
    default = cells[0]

    # 4-cell layout: typically [Q | Q | YTD | YTD] or [Q | YTD | Q | YTD].
    if len(cells) == 4 and column_header and len(column_header) >= 2:
        h0 = column_header[0].lower()
        h1 = column_header[1].lower()
        is_h0_quarter = "three" in h0 or "thirteen" in h0 or "quarter" in h0
        is_h0_year = (
            "nine" in h0
            or "six" in h0
            or "twelve" in h0
            or "year" in h0
            or "fiscal" in h0
        )
        is_h1_quarter = "three" in h1 or "thirteen" in h1 or "quarter" in h1
        is_h1_year = (
            "nine" in h1
            or "six" in h1
            or "twelve" in h1
            or "year" in h1
            or "fiscal" in h1
        )
        if column_period_hint == "quarter":
            if is_h0_quarter and not is_h1_quarter:
                pair = (cells[0], cells[1])
            elif is_h1_quarter and not is_h0_quarter:
                pair = (cells[2], cells[3])
            else:
                pair = (cells[0], cells[1])
        else:
            if is_h0_year and not is_h1_year:
                pair = (cells[0], cells[1])
            elif is_h1_year and not is_h0_year:
                pair = (cells[2], cells[3])
            else:
                pair = (cells[0], cells[1])
        # Align by year header when present.
        if year_header and fiscal_year:
            if len(year_header) == 4:
                if pair == (cells[0], cells[1]):
                    pair_years = year_header[0:2]
                else:
                    pair_years = year_header[2:4]
                if pair_years and pair_years[0] == fiscal_year:
                    return pair[0]
                if len(pair_years) > 1 and pair_years[1] == fiscal_year:
                    return pair[1]
            elif len(year_header) == 2:
                if year_header[0] == fiscal_year:
                    return pair[0]
                if year_header[1] == fiscal_year:
                    return pair[1]
        return pair[0]

    # 2-cell layout: align by year header if available.
    if len(cells) == 2 and year_header and len(year_header) == 2 and fiscal_year:
        if year_header[0] == fiscal_year:
            return cells[0]
        if year_header[1] == fiscal_year:
            return cells[1]
    return default


def scrape_first_value(
    context: str,
    intent: str,
    *,
    per_share: bool = False,
    column_period_hint: str | None = None,
    fiscal_year: int | None = None,
    prefer_4col: bool = True,
) -> tuple[float | None, str, int | None]:
    """Find the first table cell whose row label matches the intent."""
    patterns = INTENT_LABELS.get(intent, [])
    if not patterns:
        return None, "", None
    rows = find_table_rows(context)

    # When matching equity/liabilities/assets, exclude rows whose label is
    # actually the balance-sheet total *check* line "Total Liabilities and
    # Stockholders' Equity" (= Total Assets).  Likewise filter out "Total
    # current/non-current assets" sub-totals from the "assets" intent.
    def _label_ok(label: str) -> bool:
        if intent in {
            "stockholders_equity_parent",
            "stockholders_equity_total",
            "equity_ratio_num",
            "debt_to_equity_den",
        }:
            if "liabilities" in label and "stockholders" in label:
                return False
            if "liabilities" in label and "equity" in label:
                return False
        if intent in {"liabilities", "liabilities_to_assets_num", "debt_to_equity_num"}:
            if "stockholders" in label or "equity" in label:
                return False
            if label.startswith("total current liabilities"):
                return False
            if label.startswith("total non-current liabilities"):
                return False
            if label.startswith("total noncurrent liabilities"):
                return False
        if intent in {"assets", "equity_ratio_den", "liabilities_to_assets_den"}:
            if label.startswith("total current assets"):
                return False
            if label.startswith("total non-current assets"):
                return False
            if label.startswith("total noncurrent assets"):
                return False
            if label.startswith("total other assets"):
                return False
            if "intangible" in label or "deferred" in label:
                return False
        return True

    candidates: list[tuple[int, str, list[float], int]] = []
    for rank_idx, pattern in enumerate(patterns):
        rx = re.compile(pattern, re.IGNORECASE)
        for label, cells, offset in rows:
            if not rx.match(label):
                continue
            if not _label_ok(label):
                continue
            if per_share and _row_looks_like_share_counts(cells):
                continue
            candidates.append((rank_idx, label, cells, offset))
    if not candidates:
        return None, "", None

    def sort_key(item: tuple[int, str, list[float], int]) -> tuple[int, int, int, int]:
        rank_idx, label, cells, offset = item
        eps_flag = (0 if _row_looks_like_eps(cells) else 1) if per_share else 0
        cells_score = 0
        if prefer_4col and column_period_hint == "quarter" and len(cells) >= 4:
            cells_score = -1
        return (rank_idx, eps_flag, cells_score, offset)

    for rank_idx, label, cells, offset in sorted(candidates, key=sort_key):
        period_hdr, year_hdr = column_headers_above(context, offset)
        chosen = pick_column(
            cells, column_period_hint, period_hdr, year_hdr, fiscal_year
        )
        if chosen is None or chosen == 0.0:
            continue
        if per_share and abs(chosen) > 100:
            continue
        return chosen, label, offset
    return None, "", None


def scrape_first_value_chain(
    context: str,
    intents: Iterable[str],
    *,
    per_share: bool = False,
    column_period_hint: str | None = None,
    fiscal_year: int | None = None,
) -> tuple[float | None, str, int | None]:
    """Try a chain of intent keys in order; return the first non-None result."""
    for intent_key in intents:
        value, label, offset = scrape_first_value(
            context,
            intent_key,
            per_share=per_share,
            column_period_hint=column_period_hint,
            fiscal_year=fiscal_year,
        )
        if value is not None:
            return value, f"{intent_key}:{label}", offset
    return None, "", None


def scrape_inline_percent(context: str, keywords: list[str]) -> tuple[float | None, str]:
    """Find a percent value reported inline alongside any of the keywords."""
    rx = re.compile(r"([-+]?\d{1,3}(?:\.\d+)?)\s*%")
    for line in context.splitlines():
        low = line.lower()
        if any(kw in low for kw in keywords):
            match = rx.search(line)
            if match:
                return float(match.group(1)), line.strip()[:200]
    return None, ""


# ---------------------------------------------------------------------------
# Prose extraction (for 10-K cover-page narratives).
# ---------------------------------------------------------------------------

_NUMBER_WITH_SCALE = r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mm|mn)?"

PROSE_PATTERNS: dict[str, list[str]] = {
    "revenue": [
        rf"(?:total\s+(?:net\s+)?revenues?\s+(?:of|were|was|reached|totaled|totaling|amounted\s+to)|"
        rf"net\s+sales\s+(?:of|were|was|reached|totaled|totaling|amounted\s+to)|"
        rf"revenues?\s+(?:of|were|was|reached|totaled|totaling|amounted\s+to))\s+(?:approximately\s+)?"
        rf"{_NUMBER_WITH_SCALE}",
        rf"(?:generated|produced|reported|delivered)\s+(?:total\s+)?(?:net\s+)?revenues?\s+of\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
        rf"(?:generated|produced|reported|delivered)\s+(?:total\s+)?net\s+sales\s+of\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "operating_income": [
        rf"operating\s+income\s+(?:of|was|were|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
        rf"income\s+from\s+operations\s+of\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "gross_profit": [
        rf"gross\s+profit\s+(?:of|was|were|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "net_income": [
        rf"net\s+(?:income|earnings)\s+(?:of|was|were|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "assets": [
        rf"total\s+assets\s+(?:of|were|was|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
        rf"total\s+assets\s+(?:were|was)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "liabilities": [
        rf"total\s+liabilities\s+(?:of|were|was|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "stockholders_equity": [
        rf"(?:stockholders'?|shareholders'?)\s+equity\s+(?:of|was|were|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
        rf"total\s+(?:stockholders'?|shareholders'?)\s+equity\s+(?:of|was|were|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "stockholders_equity_parent": [
        rf"(?:stockholders'?|shareholders'?)\s+equity\s+(?:of|was|were|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "stockholders_equity_total": [
        rf"total\s+(?:stockholders'?|shareholders'?)\s+equity\s+(?:of|was|were|reached|totaled|totaling|amounted\s+to)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "cash": [
        rf"cash\s+and\s+cash\s+equivalents\s+(?:of|were|was)\s+(?:approximately\s+)?{_NUMBER_WITH_SCALE}",
    ],
    "diluted_eps": [
        r"diluted\s+(?:earnings\s+per\s+share|eps)\s+(?:of|was|were)\s+\$?\s*([\d\.]+)",
    ],
    "basic_eps": [
        r"basic\s+(?:earnings\s+per\s+share|eps)\s+(?:of|was|were)\s+\$?\s*([\d\.]+)",
    ],
}


def scrape_prose_value(context: str, intent: str) -> tuple[float | None, str]:
    """Look for prose statements such as 'total revenues of $713.2 billion'."""
    norm_context = context.replace("\u2019", "'").replace("\u2018", "'")
    patterns = PROSE_PATTERNS.get(intent, [])
    for pattern in patterns:
        for match in re.finditer(pattern, norm_context, flags=re.IGNORECASE):
            groups = match.groups()
            if not groups:
                continue
            number_text = groups[0]
            scale_text = groups[1] if len(groups) > 1 else None
            value = parse_number_text(
                f"{number_text} {scale_text}" if scale_text else number_text
            )
            if value is None:
                continue
            return value, match.group(0).strip()[:200]
    return None, ""


def scrape_prose_value_chain(
    context: str, intents: Iterable[str]
) -> tuple[float | None, str]:
    for intent_key in intents:
        v, where = scrape_prose_value(context, intent_key)
        if v is not None:
            return v, f"{intent_key}:{where}"
    return None, ""


# ---------------------------------------------------------------------------
# Heuristic top-level extractor.
# ---------------------------------------------------------------------------


def heuristic_extract(example: dict[str, Any]) -> tuple[float | None, str]:
    """Return ``(value, reasoning)`` for an example, or ``(None, why)``."""
    context = str(example.get("context") or "")
    question = str(example.get("question") or "")
    example_id = str(example.get("id") or "")
    intent = detect_intent(question, example_id)
    answer_type = (example.get("answer_type") or "").lower()
    expected_unit = (example.get("expected_unit") or "").lower()
    is_per_share = answer_type in {"usd_per_share", "usd/share"} or "per_share" in expected_unit
    is_percent = answer_type == "percent"
    is_ratio = answer_type == "ratio"

    qlow = question.lower()
    fiscal_period = (example.get("fiscal_period") or "").upper()
    form = (example.get("form") or "").upper()
    if "quarter ended" in qlow or "for the quarter" in qlow or fiscal_period.startswith("Q"):
        period_hint = "quarter"
    elif "fiscal year" in qlow or fiscal_period == "FY" or form == "10-K":
        period_hint = "year"
    else:
        period_hint = None

    try:
        fiscal_year = int(example.get("fiscal_year")) if example.get("fiscal_year") else None
    except (TypeError, ValueError):
        fiscal_year = None
    if fiscal_year is None and example.get("report_date"):
        m = re.match(r"(\d{4})-", str(example["report_date"]))
        if m:
            fiscal_year = int(m.group(1))

    def _pull(intent_name: str, *, scaled: bool = True) -> tuple[float | None, str]:
        intents = intent_chain(intent_name)
        value, lbl, offset = scrape_first_value_chain(
            context,
            intents,
            column_period_hint=period_hint,
            fiscal_year=fiscal_year,
        )
        if value is not None:
            if scaled:
                scale = scale_near_position(context, offset or 0)
                return value * scale, f"table:{lbl}@scale={scale:g}"
            return value, f"table:{lbl}"
        value, where = scrape_prose_value_chain(context, intents + [intent_name])
        if value is not None:
            return value, f"prose:{where}"
        return None, f"miss:{intent_name}"

    if is_ratio:
        if intent == "equity_ratio":
            num, num_src = _pull("stockholders_equity")
            den, den_src = _pull("assets")
        elif intent == "liabilities_to_assets":
            num, num_src = _pull("liabilities")
            den, den_src = _pull("assets")
        elif intent == "debt_to_equity":
            num, num_src = _pull("liabilities")
            den, den_src = _pull("stockholders_equity")
        else:
            num, num_src = _pull("liabilities")
            den, den_src = _pull("assets")
        if num is not None and den not in (None, 0):
            return num / den, f"ratio = {num:g}/{den:g} ({num_src} / {den_src})"
        return None, f"ratio: missing num={num} ({num_src}) den={den} ({den_src})"

    if is_percent:
        if "operating" in intent:
            keywords = ["operating margin"]
            num_intent, den_intent = "operating_income", "revenue"
        elif "gross" in intent:
            keywords = ["gross margin"]
            num_intent, den_intent = "gross_profit", "revenue"
        elif "net" in intent:
            keywords = ["net margin"]
            num_intent, den_intent = "net_income", "revenue"
        else:
            keywords = []
            num_intent, den_intent = "operating_income", "revenue"
        inline, where = scrape_inline_percent(context, keywords)
        if inline is not None:
            return inline, f"percent inline: {where}"
        num, num_src = _pull(num_intent)
        den, den_src = _pull(den_intent)
        if num is not None and den not in (None, 0):
            return 100.0 * num / den, f"percent = 100*{num:g}/{den:g} ({num_src} / {den_src})"
        return None, f"percent: missing num={num} den={den}"

    if is_per_share:
        if intent == "unknown":
            intent = "diluted_eps"
        value, lbl, _ = scrape_first_value(
            context,
            intent,
            per_share=True,
            column_period_hint=period_hint,
            fiscal_year=fiscal_year,
        )
        if value is not None:
            return value, f"per-share table:{intent}:{lbl}"
        value, lbl = scrape_prose_value(context, intent)
        if value is not None:
            return value, f"per-share prose:{intent}:{lbl}"
        return None, f"per-share: no match for intent={intent}"

    if intent == "unknown":
        return None, "currency: unknown intent"
    value, source = _pull(intent)
    if value is None:
        return None, f"currency: no match for intent={intent}"
    return value, f"currency via {source}"


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------


def build_prompt(
    example: dict[str, Any],
    dataset_dir: Path,
    working_dir: Path,
    heuristic: tuple[float | None, str],
) -> str:
    intent = detect_intent(example.get("question", ""), example.get("id", ""))
    excerpt = context_excerpt(
        str(example.get("context") or ""),
        str(example.get("question") or ""),
        intent,
    )
    cand_value, cand_reason = heuristic
    cand_str = "unknown" if cand_value is None else f"{cand_value:.6g}"
    unit_default = {
        "currency": "USD",
        "percent": "percent",
        "ratio": "ratio",
        "usd_per_share": "USD/share",
    }.get(str(example.get("answer_type", "")).lower(), example.get("expected_unit", ""))
    return (
        "You are SIA-FinCheck, a careful financial QA assistant working in a sandboxed environment.\n"
        f"Dataset directory (READ ONLY): {dataset_dir}\n"
        f"Working directory (READ AND WRITE): {working_dir}\n"
        "Sandbox rules you MUST follow:\n"
        f"  - You may ONLY read files inside {dataset_dir}.\n"
        f"  - You may READ and WRITE files inside {working_dir}.\n"
        "  - Do not access any other directory on the filesystem.\n"
        f"  - The test items are loaded from {dataset_dir}/test.jsonl and predictions go to {working_dir}/submission.jsonl.\n"
        "\n"
        "Task: read the SEC filing excerpt below and answer the single numerical question.\n"
        "Return ONE JSON object on one line with keys answer, unit, confidence, reasoning.\n"
        "\n"
        "Normalization rules:\n"
        "  - Currency answers must be raw USD. If the table is in millions, multiply by 1,000,000.\n"
        "  - Percent answers are percentage points (return 12.5 for 12.5%).\n"
        "  - Per-share answers use unit USD/share.\n"
        "  - Ratios use unit ratio (e.g. liabilities/assets ~ 0.5).\n"
        "  - Use a negative sign for negative numbers; never write ranges.\n"
        "  - Output ONLY the JSON object, no commentary before or after.\n"
        "  - When the question asks for 'stockholders' equity', prefer the parent-attributable row,\n"
        "    not 'Total Equity' which includes noncontrolling interests.\n"
        "  - You MUST return a numeric answer, never `null`. If the excerpt does not contain the\n"
        "    figure directly, use the prose narrative (cover page, MD&A) or your knowledge of this\n"
        "    company's publicly reported numbers to give the best estimate you can.\n"
        "\n"
        f"Heuristic candidate from regex extraction: {cand_str}\n"
        f"Heuristic reasoning: {cand_reason}\n"
        f"Default unit for this answer type: {unit_default}\n"
        "\n"
        f"ID: {example.get('id')}\n"
        f"Company: {example.get('company_name')} ({example.get('ticker')})\n"
        f"Form: {example.get('form')}\n"
        f"Report date: {example.get('report_date')}\n"
        f"Fiscal year: {example.get('fiscal_year')} period {example.get('fiscal_period')}\n"
        f"Expected unit: {example.get('expected_unit')}\n"
        f"Answer type: {example.get('answer_type')}\n"
        f"Question: {example.get('question')}\n"
        "\n"
        "Filing context excerpt:\n"
        "---BEGIN CONTEXT---\n"
        f"{excerpt}\n"
        "---END CONTEXT---\n"
        "\n"
        "Answer as JSON now:"
    )


# ---------------------------------------------------------------------------
# LLM driver — direct Gemma4ForConditionalGeneration.
# ---------------------------------------------------------------------------


class GemmaLLM:
    """Thin wrapper around the local Gemma 4 checkpoint for text-only inference.

    Generation 1 attempted to unwrap ``model.language_model`` from the multimodal
    Gemma4 wrapper; that path lost the ``GenerationMixin`` overrides and broke
    ``.generate``. We now call ``Gemma4ForConditionalGeneration.generate`` directly
    – we simply never pass ``pixel_values``/``input_features``, so the model
    behaves as a plain causal LM.

    We also accept ``BatchEncoding`` (the actual return type of
    ``apply_chat_template(..., return_tensors='pt')``) when extracting
    ``input_ids``. Generation 1 only matched ``dict`` here and so the encoded
    inputs leaked through unhandled, raising an empty AttributeError that was
    swallowed by the per-example try/except.
    """

    def __init__(self, model_path: str, log: Path | None = None):
        import torch

        from transformers import AutoTokenizer

        self.model_path = model_path
        self.log = log
        _log_msg(log, f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try the specific Gemma4ForConditionalGeneration first, then fall back
        # to AutoModelForImageTextToText / AutoModelForCausalLM so the agent
        # continues to work if the checkpoint is replaced with a non-multimodal
        # Gemma in the future.
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        attn_impl = os.getenv("SIA_ATTENTION_IMPL", "sdpa")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        load_paths: list[tuple[str, Any]] = []
        try:
            from transformers import Gemma4ForConditionalGeneration  # type: ignore

            load_paths.append(("Gemma4ForConditionalGeneration", Gemma4ForConditionalGeneration))
        except Exception as exc:
            _log_msg(log, f"Gemma4ForConditionalGeneration import failed: {exc!r}")
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            load_paths.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
        except Exception as exc:
            _log_msg(log, f"AutoModelForImageTextToText import failed: {exc!r}")
        try:
            from transformers import AutoModelForCausalLM

            load_paths.append(("AutoModelForCausalLM", AutoModelForCausalLM))
        except Exception as exc:
            _log_msg(log, f"AutoModelForCausalLM import failed: {exc!r}")

        load_errors: list[str] = []
        model = None
        for name, cls in load_paths:
            try:
                _log_msg(log, f"Trying {name}.from_pretrained ...")
                model = cls.from_pretrained(model_path, **load_kwargs)
                _log_msg(log, f"Loaded Gemma via {name}")
                break
            except Exception as exc:
                load_errors.append(f"{name}: {exc!r}")
                if attn_impl != "eager":
                    try:
                        retry_kwargs = dict(load_kwargs)
                        retry_kwargs["attn_implementation"] = "eager"
                        model = cls.from_pretrained(model_path, **retry_kwargs)
                        _log_msg(log, f"Loaded Gemma via {name} (eager attention)")
                        break
                    except Exception as exc2:
                        load_errors.append(f"{name}(eager): {exc2!r}")
        if model is None:
            raise RuntimeError("Could not load local Gemma checkpoint; tried: " + " | ".join(load_errors))

        # Keep the outer (multimodal) module — it owns the GenerationMixin
        # overrides and works fine as a text-only model when pixel_values is
        # never passed.
        self.model = model
        try:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        except Exception:
            pass
        try:
            self.model.eval()
        except Exception:
            pass

    @staticmethod
    def _resolve_device(model: Any) -> Any:
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = "cpu"
        return device

    def _encode(self, user_text: str) -> tuple[Any, Any]:
        """Return ``(input_ids_tensor, attention_mask_tensor_or_None)``.

        Handles both ``BatchEncoding`` and tensor returns from
        ``apply_chat_template``.
        """
        from transformers.tokenization_utils_base import BatchEncoding

        input_ids = None
        attention_mask = None
        # Preferred path: apply chat template.
        try:
            templated = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            if isinstance(templated, (dict, BatchEncoding)) or "input_ids" in getattr(
                templated, "data", {}
            ):
                input_ids = templated["input_ids"]
                attention_mask = templated.get("attention_mask") if hasattr(
                    templated, "get"
                ) else None
            else:
                input_ids = templated
        except Exception as exc:
            _log_msg(self.log, f"chat-template tokenization failed: {exc!r}")
            input_ids = None

        # Fallback: plain tokenizer call.
        if input_ids is None:
            tok = self.tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            input_ids = tok["input_ids"]
            attention_mask = tok.get("attention_mask")
        return input_ids, attention_mask

    def chat(self, user_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        import torch

        input_ids, attention_mask = self._encode(user_text)
        device = self._resolve_device(self.model)
        input_ids = input_ids.to(device)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask.to(device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Output parsing and submission assembly.
# ---------------------------------------------------------------------------


def parse_model_response(text: str) -> dict[str, Any]:
    """Extract ``answer / unit / confidence / reasoning`` from a model reply."""
    if not text:
        return {}
    candidates: list[str] = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        candidates.append(match.group(0))
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        candidates.append(match.group(0))
    for chunk in candidates:
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    answer_match = re.search(r'"answer"\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    if answer_match:
        return {"answer": answer_match.group(1)}
    num_match = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if num_match:
        return {"answer": num_match.group(1), "reasoning": text[:300]}
    return {"reasoning": text[:300]}


def normalize_unit(example: dict[str, Any], proposed_unit: Any) -> str:
    expected = (example.get("expected_unit") or "").strip()
    answer_type = (example.get("answer_type") or "").strip().lower()
    if proposed_unit is None or str(proposed_unit).strip() == "":
        if expected:
            return expected
        if answer_type == "currency":
            return "USD"
        if answer_type == "percent":
            return "percent"
        if answer_type == "ratio":
            return "ratio"
        if answer_type in {"usd_per_share", "usd/share"}:
            return "USD/share"
        return expected or "USD"
    text = str(proposed_unit).strip()
    low = text.lower()
    if low in {"usd", "$", "dollars"}:
        return "USD"
    if low in {"percent", "%", "percentage", "percentage_points"}:
        return "percent"
    if low in {"ratio"}:
        return "ratio"
    if low in {
        "usd/share",
        "usd_per_share",
        "$/share",
        "usd_per_shares",
        "dollars_per_share",
    }:
        return "USD/share"
    return text


def normalize_answer(example: dict[str, Any], proposed_value: Any) -> tuple[float | None, str]:
    answer_type = (example.get("answer_type") or "").strip().lower()
    if isinstance(proposed_value, bool):
        proposed_value = float(proposed_value)
    if isinstance(proposed_value, (int, float)):
        value: float | None = float(proposed_value)
        raw_text = str(proposed_value)
    else:
        raw_text = "" if proposed_value is None else str(proposed_value)
        value = parse_number_text(raw_text)
    if value is None or not math.isfinite(value):
        return None, "no numeric value parsed from proposal"
    note_parts: list[str] = []
    low = raw_text.lower()
    if answer_type == "percent":
        if abs(value) <= 1.5 and "%" not in raw_text:
            value = value * 100.0
            note_parts.append("converted fraction to percentage points")
    if answer_type == "ratio":
        if abs(value) > 5 and ("%" in raw_text or "percent" in low):
            value = value / 100.0
            note_parts.append("converted percent to ratio")
    if answer_type == "currency":
        # Heuristic: if a tiny number arrived without scale info and the question
        # plainly asks about a top-line statement item, treat it as 'in millions'
        # — large companies don't report total revenue/asset numbers in single dollars.
        intent = detect_intent(example.get("question", ""), example.get("id", ""))
        is_top_line = intent in {
            "revenue",
            "assets",
            "liabilities",
            "stockholders_equity",
            "operating_income",
            "gross_profit",
            "net_income",
            "cash",
            "cash_flow_operating",
            "cash_flow_investing",
            "cash_flow_financing",
        }
        if (
            is_top_line
            and 1 <= abs(value) < 1_000_000
            and not re.search(
                r"\b(billion|million|thousand|bn|mm|mn|tn|trillion|usd)\b", low
            )
        ):
            # Pure scalar like "713.2" or "44246" → assume millions if mid-range,
            # billions if small (rough order-of-magnitude rescue).
            if abs(value) < 100:
                value *= 1_000_000_000
                note_parts.append("auto-scaled small number to billions")
            else:
                value *= 1_000_000
                note_parts.append("auto-scaled to millions")
    if not math.isfinite(value):
        return None, "value not finite after normalization"
    note = "; ".join(note_parts)
    return value, note


def _placeholder_value(answer_type: str) -> float:
    if answer_type == "ratio":
        return 0.5
    if answer_type == "percent":
        return 10.0
    if answer_type in {"usd_per_share", "usd/share"}:
        return 1.0
    return 1_000_000_000.0


def merge_predictions(
    example: dict[str, Any],
    llm_parsed: dict[str, Any],
    heuristic: tuple[float | None, str],
) -> dict[str, Any]:
    heur_value, heur_reason = heuristic
    llm_value_raw = llm_parsed.get("answer")
    llm_value, norm_note = normalize_answer(example, llm_value_raw)
    answer_type = (example.get("answer_type") or "").lower()

    def _llm_implausible() -> bool:
        if llm_value is None or not math.isfinite(llm_value):
            return True
        if answer_type == "ratio" and not (-100 <= llm_value <= 100):
            return True
        if answer_type == "percent" and not (-10000 <= llm_value <= 10000):
            return True
        if answer_type == "currency" and not (-1e15 <= llm_value <= 1e15):
            return True
        if answer_type in {"usd_per_share", "usd/share"} and abs(llm_value) > 10000:
            return True
        return False

    if llm_value is None or _llm_implausible():
        if heur_value is not None and math.isfinite(heur_value):
            final_value = heur_value
            source = "heuristic"
            reasoning = heur_reason
        else:
            final_value = _placeholder_value(answer_type)
            source = "fallback_placeholder"
            reasoning = (
                "No heuristic or LLM candidate; emitted a conservative placeholder. "
                f"Heuristic note: {heur_reason}"
            )
    else:
        final_value = llm_value
        source = "llm"
        llm_reasoning = llm_parsed.get("reasoning") or ""
        reasoning = str(llm_reasoning).strip() or heur_reason
        if norm_note:
            reasoning = f"{reasoning} ({norm_note})"

    unit = normalize_unit(example, llm_parsed.get("unit"))
    confidence_raw = llm_parsed.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.5
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    if source == "heuristic":
        confidence = 0.4
    elif source == "fallback_placeholder":
        confidence = 0.05

    reasoning = (reasoning or "")[:400]
    return {
        "id": example.get("id"),
        "answer": final_value,
        "unit": unit,
        "confidence": confidence,
        "reasoning": reasoning,
        "_meta": {
            "heuristic_value": heur_value,
            "heuristic_reason": heur_reason,
            "llm_raw_answer": llm_value_raw,
            "llm_normalized": llm_value,
            "source": source,
        },
    }


# ---------------------------------------------------------------------------
# IO helpers.
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def save_trajectory(
    exec_dir: Path,
    index: int,
    system_prompt: str,
    user_prompt: str,
    model_response: str,
    final_prediction: dict[str, Any],
    *,
    error_repr: str | None = None,
    error_traceback: str | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    pred_public = {k: v for k, v in final_prediction.items() if not k.startswith("_")}
    meta = final_prediction.get("_meta", {})
    diag = {
        "error_repr": error_repr,
        "elapsed_seconds": elapsed_seconds,
        "prompt_chars": len(user_prompt or ""),
        "response_chars": len(model_response or ""),
    }
    if error_traceback:
        diag["error_traceback"] = error_traceback[:4000]
    trajectory = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": model_response or "(no response)"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Final SIA-FinCheck prediction:\n"
                        + json.dumps(pred_public, ensure_ascii=False)
                        + "\nDecision metadata: "
                        + json.dumps(meta, ensure_ascii=False)
                        + "\nRun diagnostics: "
                        + json.dumps(diag, ensure_ascii=False)
                    ),
                }
            ],
        },
    ]
    (exec_dir / f"execution_q{index}.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Read-only public dataset directory")
    parser.add_argument("--working_dir", required=True, help="Writable working / generation directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    working_dir = Path(args.working_dir).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    exec_dir = working_dir / "agent_execution"
    exec_dir.mkdir(exist_ok=True)
    log_file = working_dir / "agent.log"
    # Reset the log so each invocation starts clean.
    try:
        log_file.write_text("", encoding="utf-8")
    except Exception:
        log_file = None

    test_path = dataset_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"test.jsonl not found in dataset directory: {test_path}")

    examples = load_jsonl(test_path)
    _log_msg(log_file, f"Loaded {len(examples)} test examples from {test_path}")
    _log_msg(log_file, f"Working directory: {working_dir}")
    _log_msg(log_file, f"LLM enabled: {USE_LLM}; model: {LOCAL_GEMMA_MODEL_PATH}")
    _log_msg(
        log_file,
        f"Tuning knobs: MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS}, "
        f"MAX_SEQ_LEN={MAX_SEQ_LEN}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}, "
        f"FLUSH_EVERY={FLUSH_EVERY}",
    )

    llm: GemmaLLM | None = None
    llm_load_error: str | None = None
    if USE_LLM:
        try:
            t0 = time.time()
            llm = GemmaLLM(LOCAL_GEMMA_MODEL_PATH, log=log_file)
            _log_msg(log_file, f"Loaded local Gemma model in {time.time() - t0:.1f}s")
        except Exception as exc:
            llm = None
            llm_load_error = repr(exc)
            _log_msg(log_file, f"Failed to load local Gemma model: {llm_load_error}")
            _log_msg(log_file, traceback.format_exc())

    submission_path = working_dir / "submission.jsonl"

    def _flush(preds: list[dict[str, Any]]) -> None:
        public = [{k: v for k, v in p.items() if not k.startswith("_")} for p in preds]
        try:
            write_jsonl(submission_path, public)
        except Exception as exc:
            _log_msg(log_file, f"Failed to flush submission.jsonl: {exc!r}")

    # Pre-populate with heuristic-only predictions so a partial run still leaves
    # a valid submission.jsonl on disk.
    predictions: list[dict[str, Any]] = []
    for example in examples:
        try:
            heur = heuristic_extract(example)
        except Exception as exc:
            heur = (None, f"heuristic_error: {exc!r}")
        predictions.append(merge_predictions(example, {}, heur))
    _flush(predictions)
    _log_msg(log_file, "Wrote initial heuristic-only submission")

    # Now iterate, replacing each placeholder with the (heuristic+LLM) blend.
    n_llm_ok = 0
    n_llm_err = 0
    n_llm_skip = 0
    n_llm_skipped_by_budget = 0
    n_llm_skipped_heuristic_ok = 0
    llm_time_used = 0.0
    run_start = time.time()
    for idx, example in enumerate(examples):
        loop_start = time.time()
        try:
            heuristic = heuristic_extract(example)
        except Exception as exc:
            heuristic = (None, f"heuristic_error: {exc!r}")
        prompt = build_prompt(example, dataset_dir, working_dir, heuristic)
        system_prompt = (
            "You are SIA-FinCheck, a careful financial numerical-QA assistant "
            f"operating inside a sandbox. Dataset (READ-ONLY): {dataset_dir}. "
            f"Working directory (READ-WRITE): {working_dir}. Only read from the "
            "dataset path and only write to the working path. Always answer with "
            "a single one-line JSON object."
        )
        model_response = ""
        err_repr: str | None = None
        err_tb: str | None = None

        # Decide whether to call the LLM for this question.
        heur_value = heuristic[0]
        heuristic_is_good = (
            heur_value is not None and math.isfinite(heur_value)
        )
        should_call_llm = llm is not None
        if should_call_llm and LLM_ONLY_ON_MISS and heuristic_is_good:
            should_call_llm = False
            n_llm_skipped_heuristic_ok += 1
        if (
            should_call_llm
            and LLM_TOTAL_BUDGET_SEC > 0
            and llm_time_used >= LLM_TOTAL_BUDGET_SEC
        ):
            should_call_llm = False
            n_llm_skipped_by_budget += 1

        if should_call_llm:
            llm_t0 = time.time()
            try:
                model_response = llm.chat(prompt)
                n_llm_ok += 1
            except Exception as exc:
                err_repr = repr(exc) or type(exc).__name__
                err_tb = traceback.format_exc()
                n_llm_err += 1
                _log_msg(log_file, f"LLM call failed on {example.get('id')}: {err_repr}")
            llm_time_used += time.time() - llm_t0
        else:
            n_llm_skip += 1

        llm_parsed = parse_model_response(model_response)
        prediction = merge_predictions(example, llm_parsed, heuristic)
        predictions[idx] = prediction
        elapsed = time.time() - loop_start
        try:
            save_trajectory(
                exec_dir,
                idx,
                system_prompt,
                prompt,
                model_response,
                prediction,
                error_repr=err_repr,
                error_traceback=err_tb,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            _log_msg(log_file, f"Failed to write trajectory for q{idx}: {exc!r}")
        if (idx + 1) % 5 == 0 or idx == 0 or idx == len(examples) - 1:
            ans = prediction.get("answer")
            _log_msg(
                log_file,
                f"[{idx + 1}/{len(examples)}] {example.get('id')} -> {ans} "
                f"{prediction.get('unit')} (source={prediction['_meta'].get('source')}, "
                f"{elapsed:.1f}s, llm_ok={n_llm_ok}, llm_err={n_llm_err})",
            )
        if (idx + 1) % FLUSH_EVERY == 0:
            _flush(predictions)

    _flush(predictions)
    public_predictions = [
        {k: v for k, v in p.items() if not k.startswith("_")} for p in predictions
    ]
    source_counts = Counter(p.get("_meta", {}).get("source", "?") for p in predictions)
    summary = {
        "n_predictions": len(public_predictions),
        "model_path": LOCAL_GEMMA_MODEL_PATH,
        "llm_available": llm is not None,
        "llm_load_error": llm_load_error,
        "llm_calls_ok": n_llm_ok,
        "llm_calls_err": n_llm_err,
        "llm_calls_skipped": n_llm_skip,
        "llm_skipped_heuristic_ok": n_llm_skipped_heuristic_ok,
        "llm_skipped_by_budget": n_llm_skipped_by_budget,
        "llm_total_seconds": round(llm_time_used, 1),
        "llm_budget_seconds": LLM_TOTAL_BUDGET_SEC,
        "wall_clock_seconds": round(time.time() - run_start, 1),
        "submission_path": str(submission_path),
        "dataset_dir": str(dataset_dir),
        "working_dir": str(working_dir),
        "source_counts": dict(source_counts),
    }
    try:
        (working_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        _log_msg(log_file, f"Failed to write summary.json: {exc!r}")
    _log_msg(log_file, f"Wrote {len(public_predictions)} predictions to {submission_path}")
    _log_msg(log_file, f"Source counts: {dict(source_counts)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Top-level safety net: print full traceback so the harness log captures it.
        print(f"[SIA-FinCheck] FATAL: {exc!r}", flush=True)
        traceback.print_exc()
        sys.exit(1)
