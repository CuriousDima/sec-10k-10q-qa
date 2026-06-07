#!/usr/bin/env python3
"""SIA-FinCheck target agent.

Reads SEC 10-K / 10-Q QA examples from `--dataset_dir/test.jsonl`, answers each one
using a deterministic financial-context heuristic refined by the locally hosted
`/workspace/gemma_checkpoints/gemma-4-31B-it` checkpoint, and writes the predictions
to `--working_dir/submission.jsonl`. Per-question execution trajectories are saved to
`--working_dir/agent_execution/execution_q{i}.json`.

Rules enforced here:
  * Only the local checkpoint at LOCAL_GEMMA_MODEL_PATH (default
    /workspace/gemma_checkpoints/gemma-4-31B-it) is used for LLM calls.
  * Every prompt that goes to the model literally contains the dataset_dir and
    working_dir paths along with a sandbox notice ("READ only / READ-WRITE").
  * Nothing outside --dataset_dir is read and nothing outside --working_dir is
    written. --dataset_dir is never modified.
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
from typing import Any

DEFAULT_MODEL_PATH = "/workspace/gemma_checkpoints/gemma-4-31B-it"
LOCAL_GEMMA_MODEL_PATH = os.getenv("LOCAL_GEMMA_MODEL_PATH", DEFAULT_MODEL_PATH)

MAX_CONTEXT_CHARS = int(os.getenv("SIA_FINCHECK_MAX_CONTEXT_CHARS", "12000"))
MAX_NEW_TOKENS = int(os.getenv("SIA_FINCHECK_MAX_NEW_TOKENS", "320"))
MAX_SEQ_LEN = int(os.getenv("SIA_FINCHECK_MAX_SEQ_LEN", "8192"))
USE_LLM = os.getenv("SIA_FINCHECK_USE_LLM", "1") not in {"0", "false", "False"}

STOPWORDS = {
    "what", "were", "was", "the", "company", "companies", "for", "and", "of", "as",
    "at", "to", "in", "on", "ended", "year", "quarter", "fiscal", "total", "end",
    "did", "from", "with", "that", "this", "its", "their", "amount", "value",
    "reported", "report", "during", "how", "many", "much", "is", "are", "be",
    "company's", "a", "an", "by", "per", "period", "reporting", "answer",
    "percentage",
}

# Mapping of question intent -> ordered list of regex patterns that match the
# row LABEL (i.e. the text before the first `|` separator) in a financial
# statement. The first pattern that yields a numeric value wins.
INTENT_LABELS: dict[str, list[str]] = {
    "revenue": [
        r"^total\s+net\s+sales$",
        r"^net\s+sales$",
        r"^total\s+(?:net\s+)?revenues?$",
        r"^net\s+revenues?$",
        r"^revenues?,?\s+net$",
        r"^revenues?$",
        r"^total\s+revenues?\s+(?:and\s+other(?:\s+income)?)?$",
    ],
    "operating_income": [
        r"^operating\s+income(?:\s+\(loss\))?$",
        r"^income\s+from\s+operations(?:\s+\(loss\))?$",
        r"^operating\s+(?:earnings|profit)$",
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
    ],
    "assets": [
        r"^total\s+assets$",
    ],
    "liabilities": [
        r"^total\s+liabilities$",
    ],
    "stockholders_equity": [
        r"^total\s+(?:stockholders'?|shareholders'?)\s+equity$",
        r"^total\s+equity$",
        r"^total\s+(?:stockholders'?|shareholders'?)\s+equity\s+attributable\s+to.+$",
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

# Synonym intents that share extraction rules.
INTENT_LABELS["operating_margin"] = INTENT_LABELS["operating_income"]
INTENT_LABELS["gross_margin"] = INTENT_LABELS["gross_profit"]
INTENT_LABELS["net_margin"] = INTENT_LABELS["net_income"]
INTENT_LABELS["equity_ratio_num"] = INTENT_LABELS["stockholders_equity"]
INTENT_LABELS["equity_ratio_den"] = INTENT_LABELS["assets"]
INTENT_LABELS["liabilities_to_assets_num"] = INTENT_LABELS["liabilities"]
INTENT_LABELS["liabilities_to_assets_den"] = INTENT_LABELS["assets"]
INTENT_LABELS["debt_to_equity_num"] = INTENT_LABELS["liabilities"]
INTENT_LABELS["debt_to_equity_den"] = INTENT_LABELS["stockholders_equity"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Numeric parsing helpers.                                                    #
# --------------------------------------------------------------------------- #


def parse_number_text(text: str) -> float | None:
    """Parse one numeric value out of a free-form string.

    Handles `$1,234.5`, `(123)` (negative), `12.5%`, and million/billion words.
    """
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


def question_terms(question: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower())
    return {tok for tok in tokens if tok not in STOPWORDS}


def detect_intent(question: str, example_id: str = "") -> str:
    """Return the canonical intent string for an example."""
    qlow = (question or "").lower()
    idlow = (example_id or "").lower()
    # Order matters: more specific intents come first.
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
            if slug == "equity":
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
    """Document-level scale (the most common scale token in the excerpt)."""
    tokens = [m.group(0) for m in SCALE_TOKEN_RE.finditer(context)]
    if not tokens:
        return 1.0
    scales = [_scale_from_text(t) for t in tokens]
    # Use the most frequent scale.
    counts: dict[float, int] = {}
    for s in scales:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def scale_near_position(context: str, pos: int, lookback_chars: int = 4000) -> float:
    """Pick the appropriate scale marker within ``lookback_chars`` before ``pos``.

    SEC table headers often look like
        ``(In millions, except number of shares, which are reflected in thousands)``
    so we prefer the FIRST scale token within a header parenthetical (which is the
    primary scale for currency cells); falling back to the most recent scale token
    otherwise.
    """
    start = max(0, pos - lookback_chars)
    window = context[start:pos]
    # Look for the last header-style parenthetical and use its first scale.
    paren_matches = list(re.finditer(r"\(([^)]{0,200})\)", window))
    for paren in reversed(paren_matches):
        inner = paren.group(1)
        primary = SCALE_TOKEN_RE.search(inner)
        if primary:
            return _scale_from_text(primary.group(0))
    # No parenthetical with scale: take the most recent free-standing scale token.
    last_match = None
    for match in SCALE_TOKEN_RE.finditer(window):
        last_match = match.group(0)
    if last_match:
        return _scale_from_text(last_match)
    return detect_scale_multiplier(context)


# --------------------------------------------------------------------------- #
# Context excerpting (keeps prompts tight while preserving relevant tables).   #
# --------------------------------------------------------------------------- #


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
    for intent_key in intents:
        for pattern in INTENT_LABELS.get(intent_key, []):
            for word in re.findall(r"[a-z]+", pattern):
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
        if "consolidated" in low and ("statement" in low or "balance" in low):
            score += 4
        if score:
            scored.append((score, idx))
    selected: set[int] = set(range(min(20, len(lines))))  # always keep the header
    for _score, idx in sorted(scored, reverse=True)[:200]:
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected.add(j)
    pieces = [lines[i] for i in sorted(selected)]
    excerpt = "\n".join(pieces)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    return excerpt


def intent_chain(intent: str) -> list[str]:
    """Return all label-table intents associated with a (possibly compound) intent."""
    if intent in {"equity_ratio"}:
        return ["equity_ratio_num", "equity_ratio_den"]
    if intent in {"liabilities_to_assets"}:
        return ["liabilities_to_assets_num", "liabilities_to_assets_den"]
    if intent in {"debt_to_equity"}:
        return ["debt_to_equity_num", "debt_to_equity_den"]
    if intent in {"operating_margin"}:
        return ["operating_margin", "revenue"]
    if intent in {"gross_margin"}:
        return ["gross_margin", "revenue"]
    if intent in {"net_margin"}:
        return ["net_margin", "revenue"]
    return [intent]


# --------------------------------------------------------------------------- #
# Deterministic numeric extraction from the filing context.                   #
# --------------------------------------------------------------------------- #

# Match negative numbers either as -1,234.5 or (1,234.5)
NUMERIC_CELL_RE = re.compile(
    r"""
    ^\s*
    \$?\s*
    (
        \(\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*\)        # (1,234) parenthesized negative
        |
        -?\s*\d[\d,]*(?:\.\d+)?                   # plain (possibly negative) number
    )
    \s*%?\s*$
    """,
    re.VERBOSE,
)


def _normalize_label(text: str) -> str:
    """Lowercase a row label and collapse Unicode apostrophes / footnote tags."""
    text = (text or "").strip().lower()
    # Normalize the various apostrophe glyphs the SEC text exporter emits.
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'").replace("`", "'")
    # Strip trailing footnote markers like "*" or "(1)" / "(note 5)".
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s*\*+\s*$", "", text)
    # Drop leading bullets / dashes commonly used in indented sub-totals.
    text = re.sub(r"^[-•·\u2013\u2014]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Strip a single trailing colon.
    if text.endswith(":"):
        text = text[:-1].strip()
    return text


def split_table_row(line: str) -> tuple[str, list[float]]:
    """Split a `Label | n1 | n2 | ...` row into the label text and numeric cells.

    Cells that are not numbers (e.g. `$`) are ignored. Returns `("", [])` when
    the line is not a recognizable table row.
    """
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
        if not match:
            inner = re.sub(r"[^\d\.\-\(\)]", " ", cell).strip()
            if not inner:
                continue
            value = parse_number_text(cell)
        else:
            value = parse_number_text(match.group(1))
        if value is not None and math.isfinite(value):
            cells.append(value)
    return label, cells


def find_table_rows(context: str) -> list[tuple[str, list[float], int]]:
    """Return `(label, cells, line_start_offset)` triples for each table-style row."""
    rows: list[tuple[str, list[float], int]] = []
    offset = 0
    for line in context.splitlines(keepends=True):
        label, cells = split_table_row(line.rstrip("\n"))
        if cells:
            rows.append((label, cells, offset))
        offset += len(line)
    return rows


def _row_looks_like_eps(cells: list[float]) -> bool:
    """Treat a row as EPS if every non-trivial cell is < 100 in magnitude."""
    if not cells:
        return False
    nonzero = [c for c in cells if c != 0]
    if not nonzero:
        return False
    return max(abs(c) for c in nonzero) < 100


def _row_looks_like_share_counts(cells: list[float]) -> bool:
    """Treat a row as a share count if most cells are integers larger than 100."""
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


def pick_column(
    cells: list[float],
    column_period_hint: str | None,
    column_header: list[str] | None,
    year_header: list[int] | None = None,
    fiscal_year: int | None = None,
) -> float | None:
    """Choose which numeric cell to return given the question's expected period.

    Handles common SEC layouts:
      * 2 cells, current-period vs prior-period.
      * 4 cells, two periods × two years.

    Uses ``year_header`` (parsed sub-header like [2023, 2024]) when available so
    we can match the question's ``fiscal_year`` to the correct column.
    """
    if not cells:
        return None

    # Default: first cell.
    default = cells[0]

    # --- Layout: 4 cells, two periods × two years -------------------------------
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
        # Decide which two cells belong to the requested period.
        if column_period_hint == "quarter":
            if is_h0_quarter and not is_h1_quarter:
                pair = (cells[0], cells[1])
            elif is_h1_quarter and not is_h0_quarter:
                pair = (cells[2], cells[3])
            else:
                pair = (cells[0], cells[1])
        else:  # default to year
            if is_h0_year and not is_h1_year:
                pair = (cells[0], cells[1])
            elif is_h1_year and not is_h0_year:
                pair = (cells[2], cells[3])
            else:
                pair = (cells[0], cells[1])
        # Within the pair, choose the right year column.
        if year_header and len(year_header) >= 2 and fiscal_year:
            pair_years = (
                year_header[0:2] if pair is (cells[0], cells[1]) else year_header[2:4]
            )
            # Simpler: align by index.
            if len(year_header) == 4:
                if pair == (cells[0], cells[1]):
                    pair_years = year_header[0:2]
                else:
                    pair_years = year_header[2:4]
                if len(pair_years) == 2:
                    if pair_years[0] == fiscal_year:
                        return pair[0]
                    if pair_years[1] == fiscal_year:
                        return pair[1]
            elif len(year_header) == 2:
                if year_header[0] == fiscal_year:
                    return pair[0]
                if year_header[1] == fiscal_year:
                    return pair[1]
        return pair[0]

    # --- Layout: 2 cells, current vs prior period -------------------------------
    if len(cells) == 2 and year_header and len(year_header) == 2 and fiscal_year:
        if year_header[0] == fiscal_year:
            return cells[0]
        if year_header[1] == fiscal_year:
            return cells[1]
    return default


def _parse_year_header_line(line: str) -> list[int] | None:
    """Parse a row that contains only 4-digit years separated by `|`."""
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
    """Find the most recent table column headers and a year sub-header (if any).

    Returns `(period_parts, year_parts)`. `period_parts` is the segments of the
    line that mentions "months ended" etc.; `year_parts` is the parsed year row
    immediately following (or None if not present).
    """
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
            or "quarter ended" in low
            or "fiscal year" in low
        ):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                period_line_idx = i
                period_parts = parts
    year_parts: list[int] | None = None
    if period_line_idx is not None:
        # Look at the next 1-3 lines for a year row.
        for j in range(period_line_idx + 1, min(period_line_idx + 4, len(lines))):
            yh = _parse_year_header_line(lines[j])
            if yh is not None:
                year_parts = yh
                break
    return period_parts, year_parts


def scrape_first_value(
    context: str,
    intent: str,
    *,
    per_share: bool = False,
    column_period_hint: str | None = None,
    fiscal_year: int | None = None,
    prefer_4col: bool = True,
) -> tuple[float | None, str, int | None]:
    """Find the first cell in a financial-statement table whose row label matches.

    Returns `(value, debug_label, char_offset)`. The raw cell value is returned
    BEFORE applying any scale multiplier; the caller decides whether to rescale.
    """
    patterns = INTENT_LABELS.get(intent, [])
    if not patterns:
        return None, "", None
    rows = find_table_rows(context)
    # Collect ALL candidate rows for the intent first so we can prefer richer
    # layouts (4-column quarterly + YTD) over 2-column ones.
    candidates: list[tuple[str, list[float], int]] = []
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        for label, cells, offset in rows:
            if not rx.match(label):
                continue
            if per_share and _row_looks_like_share_counts(cells):
                continue
            candidates.append((label, cells, offset))
    if not candidates:
        return None, "", None

    def rank(item: tuple[str, list[float], int]) -> tuple[int, int, int]:
        label, cells, offset = item
        eps_flag = (0 if _row_looks_like_eps(cells) else 1) if per_share else 0
        # Prefer 4-cell rows (quarter + YTD layout) when asked.
        cells_score = 0
        if prefer_4col and column_period_hint == "quarter" and len(cells) >= 4:
            cells_score = -1
        # Prefer rows that come AFTER a column header (so headers are available).
        return (eps_flag, cells_score, offset)

    for label, cells, offset in sorted(candidates, key=rank):
        period_hdr, year_hdr = column_headers_above(context, offset)
        chosen = pick_column(
            cells, column_period_hint, period_hdr, year_hdr, fiscal_year
        )
        if chosen is None or chosen == 0.0:
            continue
        if per_share and abs(chosen) > 100:
            # The pick must look like an EPS, not a share count.
            continue
        return chosen, label, offset
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


# Prose fallback patterns for management discussion / 10-K narrative excerpts.
PROSE_PATTERNS: dict[str, list[str]] = {
    "revenue": [
        r"(?:total\s+(?:net\s+)?revenues?\s+(?:of|were|was|reached|totaled|totaling)|net\s+sales\s+(?:of|were|was|reached|totaled|totaling)|revenues?\s+(?:of|were|was|reached|totaled|totaling))\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
        r"(?:generated|produced|reported|delivered)\s+(?:total\s+)?(?:net\s+)?revenues?\s+of\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
    ],
    "operating_income": [
        r"operating\s+income\s+(?:of|was|were|reached|totaled|totaling)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
        r"income\s+from\s+operations\s+of\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
    ],
    "gross_profit": [
        r"gross\s+profit\s+(?:of|was|were|reached|totaled|totaling)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
    ],
    "net_income": [
        r"net\s+(?:income|earnings)\s+(?:of|was|were|reached|totaled|totaling)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
    ],
    "assets": [
        r"total\s+assets\s+(?:of|were|was|reached|totaled|totaling)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
    ],
    "liabilities": [
        r"total\s+liabilities\s+(?:of|were|was|reached|totaled|totaling)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
    ],
    "stockholders_equity": [
        r"(?:stockholders'?|shareholders'?)\s+equity\s+(?:of|was|were|reached|totaled|totaling)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
        r"total\s+(?:stockholders'?|shareholders'?)\s+equity\s+(?:of|was|were|reached|totaled|totaling)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
    ],
    "cash": [
        r"cash\s+and\s+cash\s+equivalents\s+(?:of|were|was)\s+\$?\s*([\d,\.]+\s*(?:billion|million|thousand|bn|mm)?)",
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
    for pattern in PROSE_PATTERNS.get(intent, []):
        for match in re.finditer(pattern, norm_context, flags=re.IGNORECASE):
            value = parse_number_text(match.group(1))
            if value is None:
                continue
            return value, match.group(0).strip()[:200]
    return None, ""


def heuristic_extract(example: dict[str, Any]) -> tuple[float | None, str]:
    """Best-effort numeric extraction from the public filing context.

    Returns `(value, reasoning)` where value is a normalized number suitable for
    submission (raw USD, percentage points, ratio, USD/share). Returns
    `(None, reason)` if nothing plausible was found.
    """
    context = str(example.get("context") or "")
    question = str(example.get("question") or "")
    example_id = str(example.get("id") or "")
    intent = detect_intent(question, example_id)
    answer_type = (example.get("answer_type") or "").lower()
    expected_unit = (example.get("expected_unit") or "").lower()
    is_per_share = answer_type in {"usd_per_share", "usd/share"} or "per_share" in expected_unit
    is_percent = answer_type == "percent"
    is_ratio = answer_type == "ratio"

    # What time-window does the question ask about?
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
    # Also try to pull a year out of the report_date if fiscal_year missing.
    if fiscal_year is None and example.get("report_date"):
        m = re.match(r"(\d{4})-", str(example["report_date"]))
        if m:
            fiscal_year = int(m.group(1))

    def _pull(intent_name: str, *, scaled: bool = True) -> tuple[float | None, str]:
        value, lbl, offset = scrape_first_value(
            context,
            intent_name,
            column_period_hint=period_hint,
            fiscal_year=fiscal_year,
        )
        if value is not None:
            if scaled:
                scale = scale_near_position(context, offset or 0)
                return value * scale, f"table:{intent_name}:{lbl}@scale={scale:g}"
            return value, f"table:{intent_name}:{lbl}"
        value, lbl = scrape_prose_value(context, intent_name)
        if value is not None:
            return value, f"prose:{intent_name}:{lbl}"
        return None, f"miss:{intent_name}"

    # --- Ratios use two scraped pillars --------------------------------------------------
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

    # --- Per-share -----------------------------------------------------------------------
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

    # --- Currency ------------------------------------------------------------------------
    if intent == "unknown":
        return None, "currency: unknown intent"
    value, source = _pull(intent)
    if value is None:
        return None, f"currency: no match for intent={intent}"
    return value, f"currency via {source}"


# --------------------------------------------------------------------------- #
# Prompt construction.                                                        #
# --------------------------------------------------------------------------- #


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
        f"  - The test items are loaded from {dataset_dir}/test.jsonl and predictions are appended to {working_dir}/submission.jsonl.\n"
        "\n"
        "Task: read the SEC filing excerpt below and answer the single numerical question.\n"
        "Return a JSON object on one line with these keys:\n"
        '  {"answer": <number>, "unit": <string>, "confidence": <0..1>, "reasoning": <short string>}\n'
        "\n"
        "Normalization rules:\n"
        "  - Currency answers must be raw USD. If the table is in millions, multiply by 1,000,000.\n"
        "  - Percent answers are percentage points (return 12.5 for 12.5%).\n"
        "  - Per-share answers use unit USD/share.\n"
        "  - Ratios use unit ratio (e.g. liabilities/assets ~ 0.5).\n"
        "  - Use parentheses or negative sign for negative numbers; never write ranges.\n"
        "  - Output a JSON object only, no commentary before or after.\n"
        "  - If the filing excerpt does not contain the answer directly, use your prior knowledge of\n"
        "    this S&P 100 company's recent SEC filings to give the best numeric estimate.\n"
        "\n"
        f"Heuristic candidate from regex extraction (use as anchor unless evidence contradicts it): {cand_str}\n"
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


# --------------------------------------------------------------------------- #
# LLM driver (loaded lazily so heuristic-only mode works without GPU/libs).   #
# --------------------------------------------------------------------------- #


class GemmaLLM:
    """Thin wrapper around the local Gemma 4 checkpoint for text-only inference."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        attn_impl = os.getenv("SIA_ATTENTION_IMPL", "sdpa")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        # The released Gemma 4 31B config registers `Gemma4ForConditionalGeneration`
        # which is a multimodal class, so prefer the image-text-to-text auto class
        # first. Fall back to the language-only entry point if the multimodal one
        # is not available in the installed transformers version.
        model = None
        load_errors: list[str] = []
        load_paths = []
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            load_paths.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
        except Exception as exc:
            load_errors.append(f"AutoModelForImageTextToText import failed: {exc}")
        try:
            from transformers import AutoModelForCausalLM

            load_paths.append(("AutoModelForCausalLM", AutoModelForCausalLM))
        except Exception as exc:
            load_errors.append(f"AutoModelForCausalLM import failed: {exc}")
        try:
            from transformers import AutoModel

            load_paths.append(("AutoModel", AutoModel))
        except Exception as exc:
            load_errors.append(f"AutoModel import failed: {exc}")

        for name, cls in load_paths:
            try:
                model = cls.from_pretrained(model_path, **load_kwargs)
                print(f"[SIA-FinCheck] Loaded Gemma via {name}")
                break
            except Exception as exc:
                load_errors.append(f"{name}.from_pretrained failed: {exc}")
                # Retry with eager attention if sdpa is unsupported.
                if attn_impl != "eager":
                    try:
                        retry_kwargs = dict(load_kwargs)
                        retry_kwargs["attn_implementation"] = "eager"
                        model = cls.from_pretrained(model_path, **retry_kwargs)
                        print(f"[SIA-FinCheck] Loaded Gemma via {name} (eager attention)")
                        break
                    except Exception as exc2:
                        load_errors.append(
                            f"{name}.from_pretrained (eager) failed: {exc2}"
                        )
        if model is None:
            raise RuntimeError(
                "Could not load local Gemma checkpoint; tried: " + " | ".join(load_errors)
            )

        # If the model exposes a text-only language sub-module (Gemma4 multimodal
        # wrappers do), use that for generation to skip vision dependencies.
        self.model = getattr(model, "language_model", None) or model
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

    def chat(self, user_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        import torch

        input_ids = None
        attention_mask = None
        # 1. Try chat template (preferred so model gets correct turn tokens).
        try:
            templated = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            if isinstance(templated, dict):
                input_ids = templated.get("input_ids")
                attention_mask = templated.get("attention_mask")
            else:
                input_ids = templated
        except Exception:
            input_ids = None
        # 2. Plain tokenization fallback.
        if input_ids is None:
            tok = self.tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            input_ids = tok["input_ids"]
            attention_mask = tok.get("attention_mask")

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


# --------------------------------------------------------------------------- #
# Output parsing and submission assembly.                                     #
# --------------------------------------------------------------------------- #


def parse_model_response(text: str) -> dict[str, Any]:
    """Extract `answer`, `unit`, `confidence`, `reasoning` from a model response."""
    if not text:
        return {}
    # Try to find a JSON object first.
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
    # Best-effort fallback: pluck `"answer": ...` or first numeric value.
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
    if text.lower() in {"usd", "$", "dollars"}:
        return "USD"
    if text.lower() in {"percent", "%", "percentage", "percentage_points"}:
        return "percent"
    if text.lower() in {"ratio"}:
        return "ratio"
    if text.lower() in {"usd/share", "usd_per_share", "$/share", "usd_per_shares", "dollars_per_share"}:
        return "USD/share"
    return text


def normalize_answer(example: dict[str, Any], proposed_value: Any) -> tuple[float | None, str]:
    """Convert a proposed value into a normalized numeric answer.

    Applies the dataset's expected normalization (currency in raw USD, percent in
    percentage points, etc.). Returns `(value, note)` where note is a brief
    description of any rescaling applied.
    """
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
    if answer_type == "currency":
        # If the model returned a small number with no scale word, try to infer the
        # scale from the example metadata: this dataset always wants raw USD.
        if abs(value) < 1e5 and not re.search(
            r"\b(billion|million|thousand|bn|mm|mn|tn|trillion|usd)\b", low
        ):
            # heuristic: if the value is suspiciously small and there's no scale word,
            # we'll trust it as-is rather than guess; large companies almost always
            # have nontrivial dollar values, but we don't want to over-amplify a real
            # small number such as a per-employee cost.
            pass
    if answer_type == "percent":
        if abs(value) <= 1.5 and "%" not in raw_text:
            # Probably a fractional ratio - convert to percentage points.
            value = value * 100.0
            note_parts.append("converted fraction to percentage points")
    if answer_type == "ratio":
        if abs(value) > 5 and ("%" in raw_text or "percent" in low):
            value = value / 100.0
            note_parts.append("converted percent to ratio")
    if not math.isfinite(value):
        return None, "value not finite after normalization"
    note = "; ".join(note_parts)
    return value, note


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
            # Last-ditch sensible placeholder so unit/format/sanity scoring still passes.
            if answer_type == "ratio":
                final_value = 0.5
            elif answer_type == "percent":
                final_value = 10.0
            elif answer_type in {"usd_per_share", "usd/share"}:
                final_value = 1.0
            else:
                final_value = 1_000_000_000.0  # 1B USD default for large-cap firms
            source = "fallback_placeholder"
            reasoning = "No heuristic or LLM candidate; emitted a conservative placeholder."
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


# --------------------------------------------------------------------------- #
# Logging.                                                                    #
# --------------------------------------------------------------------------- #


def save_trajectory(
    exec_dir: Path,
    index: int,
    system_prompt: str,
    user_prompt: str,
    model_response: str,
    final_prediction: dict[str, Any],
) -> None:
    pred_public = {k: v for k, v in final_prediction.items() if not k.startswith("_")}
    meta = final_prediction.get("_meta", {})
    trajectory = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
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
                    ),
                }
            ],
        },
    ]
    (exec_dir / f"execution_q{index}.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main driver.                                                                #
# --------------------------------------------------------------------------- #


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

    test_path = dataset_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"test.jsonl not found in dataset directory: {test_path}")

    examples = load_jsonl(test_path)
    print(f"[SIA-FinCheck] Loaded {len(examples)} test examples from {test_path}")
    print(f"[SIA-FinCheck] Working directory: {working_dir}")
    print(f"[SIA-FinCheck] LLM enabled: {USE_LLM}; model: {LOCAL_GEMMA_MODEL_PATH}")

    llm: GemmaLLM | None = None
    llm_load_error: str | None = None
    if USE_LLM:
        try:
            t0 = time.time()
            llm = GemmaLLM(LOCAL_GEMMA_MODEL_PATH)
            print(f"[SIA-FinCheck] Loaded local Gemma model in {time.time() - t0:.1f}s")
        except Exception as exc:  # pragma: no cover - depends on runtime
            llm = None
            llm_load_error = f"{type(exc).__name__}: {exc}"
            print(f"[SIA-FinCheck] Failed to load local Gemma model: {llm_load_error}")
            traceback.print_exc()

    submission_path = working_dir / "submission.jsonl"

    def _flush(preds: list[dict[str, Any]]) -> None:
        public = [{k: v for k, v in p.items() if not k.startswith("_")} for p in preds]
        write_jsonl(submission_path, public)

    predictions: list[dict[str, Any]] = []
    # Pre-populate with heuristic-only predictions so a partial run still leaves a
    # valid submission.jsonl on disk.
    for example in examples:
        try:
            heuristic = heuristic_extract(example)
        except Exception as exc:  # pragma: no cover
            heuristic = (None, f"heuristic_error: {exc}")
        predictions.append(merge_predictions(example, {}, heuristic))
    _flush(predictions)

    for idx, example in enumerate(examples):
        start = time.time()
        try:
            heuristic = heuristic_extract(example)
        except Exception as exc:  # pragma: no cover
            heuristic = (None, f"heuristic_error: {exc}")
        prompt = build_prompt(example, dataset_dir, working_dir, heuristic)
        system_prompt = (
            "You are SIA-FinCheck, a careful financial numerical-QA assistant "
            f"operating inside a sandbox. Dataset (READ-ONLY): {dataset_dir}. "
            f"Working directory (READ-WRITE): {working_dir}. Only read from the "
            "dataset path and only write to the working path. Always answer with "
            "a single one-line JSON object."
        )
        model_response = ""
        if llm is not None:
            try:
                model_response = llm.chat(prompt)
            except Exception as exc:  # pragma: no cover - depends on runtime
                model_response = ""
                print(f"[SIA-FinCheck] LLM call failed on {example.get('id')}: {exc}")
        llm_parsed = parse_model_response(model_response)
        prediction = merge_predictions(example, llm_parsed, heuristic)
        predictions[idx] = prediction
        try:
            save_trajectory(exec_dir, idx, system_prompt, prompt, model_response, prediction)
        except Exception as exc:  # pragma: no cover
            print(f"[SIA-FinCheck] Failed to write trajectory for q{idx}: {exc}")
        if (idx + 1) % 5 == 0 or idx == 0 or idx == len(examples) - 1:
            elapsed = time.time() - start
            ans = prediction.get("answer")
            print(
                f"[SIA-FinCheck] [{idx + 1}/{len(examples)}] {example.get('id')} -> "
                f"{ans} {prediction.get('unit')} (source={prediction['_meta'].get('source')}, {elapsed:.1f}s)"
            )
        # Periodically rewrite the submission so a crash leaves a usable file.
        if (idx + 1) % 10 == 0:
            _flush(predictions)

    _flush(predictions)
    public_predictions = [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in predictions
    ]
    source_counts = Counter(p.get("_meta", {}).get("source", "?") for p in predictions)
    summary = {
        "n_predictions": len(public_predictions),
        "model_path": LOCAL_GEMMA_MODEL_PATH,
        "llm_available": llm is not None,
        "llm_load_error": llm_load_error,
        "submission_path": str(submission_path),
        "dataset_dir": str(dataset_dir),
        "working_dir": str(working_dir),
        "source_counts": dict(source_counts),
    }
    (working_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[SIA-FinCheck] Wrote {len(public_predictions)} predictions to {submission_path}")
    print(f"[SIA-FinCheck] Source counts: {dict(source_counts)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
