#!/usr/bin/env python3
"""SIA-FinCheck target agent (Generation 5) backed by Sakana AI's ``fugu-mini``.

Generation 5 keeps every Gen 4 robustness win (cross-record context
augmentation, period-column hints, ratio-component extraction, atomic
incremental flushing, per-question trajectory logs, evidence-quote
verification, column legend, stockholders'-equity preference for ratio
components, self-consistency retry) and layers on:

1. **Robust partial-JSON recovery** — when the assistant response is
   truncated mid-string, the regex-based field extractor still recovers the
   `answer`, `unit`, `confidence`, `reasoning`, and `evidence_quote`
   fields independently. Truncated responses no longer collapse to a
   zero / null prediction.

2. **Larger response budget + truncation-aware retry** — `max_tokens`
   raised to 1200, `finish_reason == "length"` triggers one shrunken-prompt
   retry.

3. **Extended deterministic component extraction for percent margins** —
   the same two-step extract-then-compute pattern that lifted ratio
   accuracy to 0.97 is now applied to `operating margin`, `net margin`,
   and `gross margin` questions.

4. **Generic GAAP line-item preferences in the primary prompt** — single-
   shot questions about `equity`, `net income`, and `cash` now receive the
   same `Total Stockholders' Equity` / `Net income attributable to <Company>`
   / `Cash and cash equivalents` preferences that were already in the
   ratio-components path.

5. **Stricter relevance detection** — `_has_real_line_item` requires the
   topic keyword to appear on a tabular line (with `|` or aligned digit
   groups), not just any narrative mention. This causes more questions to
   route through cross-evidence augmentation.

6. **Never-return-0 currency rescue** — if the primary, retry, and any
   self-consistency call all fail to produce a non-zero currency answer,
   a third focused attempt with a compact prompt + deterministically-mined
   candidate numbers is issued. If that still fails, the largest plausible
   topic-line number from the evidence is emitted.

7. **Rounded-narrative ↔ precise-table snap** — when a candidate currency
   answer matches a more precise pre-mined evidence number within 0.5 %,
   the precise value replaces the rounded one (so "$681.0 billion" becomes
   "$680,985,000,000" when the exact tabular figure is available).

8. **Trajectory & summary log hardening** — every per-question execution
   log now ends with a synthetic `final_meta` message. Submission flushes
   every 10 examples (was 20).

The script accepts ``--dataset_dir`` (read-only) and ``--working_dir``
(read/write) on the command line. It never writes outside ``working_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import guard
    OpenAI = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Configuration (env-overridable so the agent works in many environments)
# --------------------------------------------------------------------------- #
MODEL = os.getenv("SIA_TARGET_MODEL", "fugu-mini")
BASE_URL = os.getenv("SAKANA_BASE_URL", "https://api.sakana.ai/v1")
API_KEY_ENV = "SAKANA_API_KEY"

MAX_CONTEXT_CHARS = int(os.getenv("SIA_FINCHECK_MAX_CONTEXT_CHARS", "22000"))
MAX_AUGMENT_CHARS = int(os.getenv("SIA_FINCHECK_MAX_AUGMENT_CHARS", "9000"))
PER_AUGMENT_SNIPPET = int(os.getenv("SIA_FINCHECK_AUGMENT_SNIPPET", "2600"))
MAX_TOKENS = int(os.getenv("SIA_FINCHECK_MAX_TOKENS", "1200"))
MAX_TOKENS_FOCUSED = int(os.getenv("SIA_FINCHECK_MAX_TOKENS_FOCUSED", "600"))
MAX_WORKERS = int(os.getenv("SIA_FINCHECK_WORKERS", "6"))
MAX_RETRIES = int(os.getenv("SIA_FINCHECK_RETRIES", "3"))
FLUSH_EVERY = int(os.getenv("SIA_FINCHECK_FLUSH_EVERY", "10"))

DATASET_CANDIDATES = ("test.jsonl", "validation.jsonl", "train.jsonl")
# JSONL files we will scan for cross-evidence (besides the active dataset).
EVIDENCE_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return rows
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL atomically via a temp file + rename so partial writes never
    leave a corrupted submission on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def discover_dataset_file(dataset_dir: Path) -> Path | None:
    """Pick the JSONL we should answer. Prefer test.jsonl."""
    for name in DATASET_CANDIDATES:
        cand = dataset_dir / name
        if cand.is_file():
            return cand
    jsonls = sorted(
        dataset_dir.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True
    )
    return jsonls[0] if jsonls else None


# --------------------------------------------------------------------------- #
# Context selection (section-aware + question-keyword scoring)
# --------------------------------------------------------------------------- #
_NUMERIC_RE = re.compile(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_/-]{2,}")

_BOILERPLATE_TERMS = (
    "indicate by check mark",
    "forward-looking",
    "forward looking",
    "risk factor",
    "table of contents",
    "exchange act of 1934",
    "incorporated by reference",
    "proxy statement",
    "shell company",
    "registrant has filed",
    "well-known seasoned issuer",
    "smaller reporting company",
    "emerging growth company",
    "trading symbol",
    "(zip code)",
    "section 13",
)

_STOPWORDS = {
    "what", "were", "was", "the", "company", "companies", "for", "and", "of", "as",
    "at", "to", "in", "on", "ended", "year", "quarter", "fiscal", "total", "end",
    "did", "from", "with", "this", "that", "are", "have", "has", "its", "their",
    "report", "reported", "value", "period", "by", "a", "an", "answer", "show",
    "shows", "any",
}

_KEYWORD_BOOSTS = {
    "revenue": 6, "revenues": 6, "sales": 6, "net": 3,
    "assets": 6, "liabilities": 6, "equity": 6, "stockholders": 5,
    "cash": 5, "operating": 4, "income": 5, "loss": 4, "earnings": 6,
    "diluted": 7, "basic": 5, "eps": 8, "per": 3, "share": 5, "shares": 4,
    "gross": 6, "profit": 6, "margin": 6,
    "expenses": 3, "cost": 3, "tax": 3, "interest": 3, "depreciation": 3,
    "investing": 5, "financing": 5, "free": 4, "flow": 4, "cashflow": 6,
    "dividends": 5, "dividend": 5,
    "research": 4, "development": 4, "r&d": 6,
    "current": 3, "noncurrent": 3, "long-term": 3, "longterm": 3, "short-term": 3,
    "goodwill": 4, "inventory": 4, "inventories": 4, "receivable": 4, "payable": 4,
    "ratio": 5,
}

_FS_HEADER_RE = re.compile(
    r"\b(consolidated\s+(condensed\s+)?(?:balance\s+sheets?|"
    r"statements?\s+of\s+(?:operations?|income|earnings|cash\s+flows?|"
    r"comprehensive(?:\s+(?:income|loss))?|stockholders.{0,3}\s*equity|"
    r"financial\s+(?:condition|position))))",
    re.IGNORECASE,
)

_FS_TYPE_BY_KEYWORD: dict[str, str] = {
    "assets": "balance_sheet",
    "liabilities": "balance_sheet",
    "equity": "balance_sheet",
    "stockholders": "balance_sheet",
    "cash": "balance_sheet",
    "inventory": "balance_sheet",
    "inventories": "balance_sheet",
    "receivable": "balance_sheet",
    "payable": "balance_sheet",
    "goodwill": "balance_sheet",
    "revenue": "income_statement",
    "revenues": "income_statement",
    "sales": "income_statement",
    "income": "income_statement",
    "profit": "income_statement",
    "loss": "income_statement",
    "earnings": "income_statement",
    "eps": "income_statement",
    "diluted": "income_statement",
    "basic": "income_statement",
    "margin": "income_statement",
    "operating": "income_statement",
    "gross": "income_statement",
    "depreciation": "income_statement",
    "expenses": "income_statement",
    "tax": "income_statement",
    "interest": "income_statement",
    "investing": "cash_flow",
    "financing": "cash_flow",
    "dividends": "cash_flow",
    "cashflow": "cash_flow",
    "flow": "cash_flow",
}

_RELEVANCE_KEYS: dict[str, tuple[str, ...]] = {
    "assets": ("total assets",),
    "liabilities": ("total liabilities",),
    "equity": (
        "total stockholders' equity",
        "total shareholders' equity",
        "total equity",
    ),
    "revenue": ("net revenues", "net sales", "total revenues", "total revenue"),
    "operating_income": (
        "operating income",
        "operating loss",
        "income from operations",
        "loss from operations",
    ),
    "net_income": ("net income", "net loss", "net earnings"),
    "gross_profit": ("gross profit", "gross margin"),
    "diluted_eps": ("diluted", "earnings per share", "per common share"),
    "cash": (
        "cash and cash equivalents",
        "cash, cash equivalents",
        "cash and equivalents",
    ),
}


def _question_terms(question: str) -> set[str]:
    tokens = {tok.lower() for tok in _TOKEN_RE.findall(question or "")}
    return {tok for tok in tokens if tok not in _STOPWORDS}


def _primary_statement_for(question: str) -> str | None:
    terms = _question_terms(question)
    counts: dict[str, int] = {}
    for term in terms:
        kind = _FS_TYPE_BY_KEYWORD.get(term)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _relevance_topic(question: str) -> str | None:
    q = (question or "").lower()
    if "equity ratio" in q or "equity-to-assets" in q:
        return "equity"
    if "liabilities-to-assets" in q or "liabilities to assets" in q:
        return "liabilities"
    if any(k in q for k in ("net margin", "net loss margin", "net income margin")):
        return "net_income"
    if "operating margin" in q or "operating loss margin" in q:
        return "operating_income"
    if "gross margin" in q or "gross profit" in q:
        return "gross_profit"
    if "diluted" in q or "earnings per share" in q or "per share" in q:
        return "diluted_eps"
    if "revenue" in q or "sales" in q or "net revenues" in q:
        return "revenue"
    if "operating income" in q or "operating loss" in q:
        return "operating_income"
    if "net income" in q or "net loss" in q or "net earnings" in q:
        return "net_income"
    if "total assets" in q or "asset" in q:
        return "assets"
    if "liabilit" in q:
        return "liabilities"
    if "equity" in q or "stockholders" in q:
        return "equity"
    if "cash and" in q or "cash equiv" in q or " cash " in f" {q} ":
        return "cash"
    return None


def _scale_hint(context: str) -> str:
    snippet = (context or "")[:8000].lower()
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


def _find_section_spans(context: str) -> list[tuple[int, int, str]]:
    lines = context.splitlines()
    headers: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if _FS_HEADER_RE.search(line):
            headers.append((idx, line.strip()))
    spans: list[tuple[int, int, str]] = []
    for i, (line_no, header) in enumerate(headers):
        end_line = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
        end_line = min(end_line, line_no + 240)
        spans.append((line_no, end_line, header))
    return spans


def _section_kind(header: str) -> str | None:
    h = header.lower()
    if "balance" in h:
        return "balance_sheet"
    if "cash flow" in h:
        return "cash_flow"
    if (
        "operations" in h
        or "income" in h
        or "earnings" in h
        or "comprehensive" in h
    ):
        return "income_statement"
    if "stockholders" in h or "shareholders" in h:
        return "balance_sheet"
    return None


def _line_is_boilerplate(line: str) -> bool:
    low = line.lower()
    return any(term in low for term in _BOILERPLATE_TERMS)


_PERIOD_HEADER_RE = re.compile(
    r"(three|six|nine|twelve)\s+months?\s+ended\b[^\n]{0,80}|"
    r"year(?:s)?\s+ended\b[^\n]{0,80}|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}",
    re.IGNORECASE,
)


def _line_looks_tabular(line: str) -> bool:
    """A line is 'tabular' if it carries pipe separators or several numbers,
    i.e. plausibly a financial-statement column-header / data row rather than
    cover-page prose."""
    if "|" in line:
        return True
    nums = re.findall(r"\b\d[\d,]{2,}\b", line)
    return len(nums) >= 2


def _extract_period_headers(
    context: str, max_headers: int = 8
) -> list[str]:
    """Collect candidate column-header strings from table-shaped lines OR
    from inside a detected financial-statement section span."""
    if not context:
        return []
    lines = context.splitlines()
    in_fs_section: set[int] = set()
    for start, end, _header in _find_section_spans(context):
        in_fs_section.update(range(start, end))

    headers: list[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        if not _line_looks_tabular(line) and idx not in in_fs_section:
            continue
        for m in _PERIOD_HEADER_RE.finditer(line):
            token = re.sub(r"\s+", " ", m.group(0)).strip(" |\t")
            if 4 <= len(token) <= 80 and token.lower() not in seen:
                headers.append(token)
                seen.add(token.lower())
            if len(headers) >= max_headers:
                return headers
    return headers


# Patterns to extract individual column-header tokens (period phrases and dates).
_COL_PERIOD_RE = re.compile(
    r"(?:(?:three|six|nine|twelve)\s+months?\s+ended|year(?:s)?\s+ended|"
    r"twelve\s+months?\s+ended)",
    re.IGNORECASE,
)
_COL_DATE_RE = re.compile(
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2},\s*\d{4}",
    re.IGNORECASE,
)


def _extract_column_legend(
    context: str, report_date: str, max_entries: int = 8
) -> list[tuple[str, str]]:
    """Return labeled column legend entries from financial-statement tables."""
    if not context:
        return []
    spans = _find_section_spans(context)
    if not spans:
        return []
    lines = context.splitlines()
    entries: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> bool:
        token = re.sub(r"\s+", " ", token).strip(" |\t")
        low = token.lower()
        if 4 <= len(token) <= 80 and low not in seen:
            seen.add(low)
            entries.append(token)
            return True
        return False

    for start, end, _header in spans:
        block_end = min(end, start + 12)
        block = lines[start:block_end]
        for line in block:
            for m in _COL_PERIOD_RE.finditer(line):
                _add(m.group(0))
                if len(entries) >= max_entries:
                    break
            for m in _COL_DATE_RE.finditer(line):
                _add(m.group(0))
                if len(entries) >= max_entries:
                    break
            if len(entries) >= max_entries:
                break
        if len(entries) >= max_entries:
            break

    legend = [(f"C{i+1}", entry) for i, entry in enumerate(entries)]
    return legend


def context_excerpt(
    context: str,
    question: str,
    max_chars: int = MAX_CONTEXT_CHARS,
    primary_kind: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compact a long filing excerpt down to the most useful sections."""
    context = context or ""
    meta: dict[str, Any] = {
        "raw_chars": len(context),
        "primary_kind": primary_kind,
        "included_sections": [],
        "trimmed": False,
    }
    if len(context) <= max_chars:
        return context, meta

    lines = [line.rstrip() for line in context.splitlines()]
    if not lines:
        return "", meta

    spans = _find_section_spans(context)
    section_text: dict[int, str] = {}
    section_kind_by_idx: dict[int, str | None] = {}
    section_size: dict[int, int] = {}
    for i, (start, end, header) in enumerate(spans):
        text = "\n".join(lines[start:end])
        section_text[i] = text
        section_kind_by_idx[i] = _section_kind(header)
        section_size[i] = len(text)

    order = ["balance_sheet", "income_statement", "cash_flow"]
    if primary_kind and primary_kind in order:
        order.remove(primary_kind)
        order = [primary_kind] + order

    picked_indices: list[int] = []
    used = 0
    budget = int(max_chars * 0.85)

    for kind in order:
        for i, k in section_kind_by_idx.items():
            if k != kind or i in picked_indices:
                continue
            size = section_size[i]
            if used + size <= budget:
                picked_indices.append(i)
                used += size
                meta["included_sections"].append(
                    {"header": spans[i][2][:120], "kind": kind, "chars": size}
                )

    picked_line_set: set[int] = set()
    for i in picked_indices:
        start, end, _ = spans[i]
        picked_line_set.update(range(start, end))

    terms = _question_terms(question)
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        if idx in picked_line_set or not line.strip():
            continue
        if _line_is_boilerplate(line):
            continue
        low = line.lower()
        score = 0
        for term in terms:
            if term in low:
                score += 4 + _KEYWORD_BOOSTS.get(term, 0)
        nums = _NUMERIC_RE.findall(line)
        score += 2 * min(len(nums), 6)
        if low.startswith("total "):
            score += 6
        if "$" in line and len(nums) >= 1:
            score += 3
        if score:
            scored.append((score, idx, line))

    selected_extra: set[int] = set()
    extra_budget = max(0, max_chars - used)
    extra_used = 0
    for _s, idx, _l in sorted(scored, reverse=True):
        if extra_used >= extra_budget:
            break
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            if j in selected_extra or j in picked_line_set:
                continue
            selected_extra.add(j)
            extra_used += len(lines[j]) + 1

    pieces: list[str] = []
    if picked_indices:
        pieces.append("--- selected financial statement sections (verbatim) ---")
        for i in sorted(picked_indices):
            pieces.append(section_text[i])
    if selected_extra:
        pieces.append("\n--- additional question-relevant lines ---")
        for idx in sorted(selected_extra):
            pieces.append(lines[idx])
    prelude: list[str] = []
    for line in lines[:60]:
        if line.strip() and not _line_is_boilerplate(line):
            prelude.append(line)
        if sum(len(x) + 1 for x in prelude) > 800:
            break
    if prelude:
        pieces.insert(0, "--- document prelude ---\n" + "\n".join(prelude))

    excerpt = "\n".join(pieces)
    meta["trimmed"] = True
    meta["final_chars"] = len(excerpt)
    return excerpt[:max_chars], meta


# --------------------------------------------------------------------------- #
# Cross-record evidence augmentation
# --------------------------------------------------------------------------- #
def _has_real_fs(context: str) -> bool:
    """Cheap test: does the supplied context contain real financial-statement
    rows (vs. just a Table-of-Contents reference)?"""
    if not context:
        return False
    patterns = (
        r"Total\s+assets[\s\$\|]*[\d,]{4,}",
        r"Total\s+liabilities[\s\$\|]*[\d,]{4,}",
        r"Net\s+(revenues?|sales)[\s\$\|]*[\d,]{4,}",
        r"Total\s+revenues?[\s\$\|]*[\d,]{4,}",
        r"Operating\s+(income|loss)[\s\$\|]*[\d,]{4,}",
        r"Net\s+(income|loss|earnings)[\s\$\|]*[\d,]{4,}",
    )
    for p in patterns:
        if re.search(p, context, re.IGNORECASE):
            return True
    return False


def _has_balance_sheet(context: str) -> bool:
    if not context:
        return False
    return bool(
        re.search(r"Total\s+assets[\s\$\|]*[\d,]{4,}", context, re.IGNORECASE)
        or re.search(r"Total\s+liabilities[\s\$\|]*[\d,]{4,}", context, re.IGNORECASE)
    )


def _normalise_for_match(s: str) -> str:
    """Lowercase and strip apostrophes / extra punctuation, so 'stockholders
    equity' matches "stockholders' equity"."""
    return re.sub(r"['\u2019]", "", (s or "").lower())


def _has_real_line_item(context: str, topic: str | None) -> bool:
    """Stricter than ``_has_relevant_data``: a topic only counts as "present"
    if it appears on a tabular line (with ``|`` or aligned digit groups)
    AND has a substantial digit group within 200 characters.

    Used to decide whether to engage cross-evidence augmentation. This avoids
    false positives where the topic is only mentioned in narrative MD&A.
    """
    if not context:
        return False
    keys = _RELEVANCE_KEYS.get(topic or "", ())
    if not keys:
        return _has_real_fs(context)
    norm_keys = [_normalise_for_match(k) for k in keys]
    lines = context.splitlines()
    for i, line in enumerate(lines):
        low_norm = _normalise_for_match(line)
        if not any(k in low_norm for k in norm_keys):
            continue
        window_lines = lines[max(0, i - 1): min(len(lines), i + 3)]
        window = "\n".join(window_lines)
        # tabular detection: a pipe AND a digit group, or many digit groups
        has_table = "|" in window
        big_nums = re.findall(r"\b\d[\d,]{3,}\b", window)
        if has_table and big_nums:
            return True
        if len(big_nums) >= 2:
            return True
    return False


def _has_relevant_data(context: str, topic: str | None) -> bool:
    """Loose relevance test: any topic key followed by a digit group within 200
    characters. Retained for backward compatibility with the augmentation
    selector — augmentation candidates only need a soft match."""
    if not context:
        return False
    if not topic:
        return _has_real_fs(context)
    keys = _RELEVANCE_KEYS.get(topic, ())
    norm_ctx = _normalise_for_match(context)
    for key in keys:
        nkey = _normalise_for_match(key)
        idx = norm_ctx.find(nkey)
        while idx != -1:
            window = norm_ctx[idx: idx + 200]
            if re.search(r"[\d,]{4,}", window):
                return True
            idx = norm_ctx.find(nkey, idx + 1)
    return _has_real_fs(context)


def _topic_extract(context: str, topic: str | None, max_chars: int) -> str:
    """Return a compact excerpt of the context focused on the requested topic."""
    if not context:
        return ""
    spans = _find_section_spans(context)
    if not spans:
        return context[:max_chars]

    kind_for_topic = {
        "assets": "balance_sheet",
        "liabilities": "balance_sheet",
        "equity": "balance_sheet",
        "cash": "balance_sheet",
        "revenue": "income_statement",
        "operating_income": "income_statement",
        "net_income": "income_statement",
        "gross_profit": "income_statement",
        "diluted_eps": "income_statement",
    }
    preferred_kind = kind_for_topic.get(topic) if topic else None

    lines = context.splitlines()
    spans_with_kind = [
        (s, e, h, _section_kind(h)) for (s, e, h) in spans
    ]

    pieces: list[str] = []
    used = 0
    ordering: list[str | None] = []
    if preferred_kind:
        ordering.append(preferred_kind)
    for other in ("balance_sheet", "income_statement", "cash_flow", None):
        if other not in ordering:
            ordering.append(other)

    for want_kind in ordering:
        for (s, e, h, k) in spans_with_kind:
            if want_kind is not None and k != want_kind:
                continue
            section_text = "\n".join(lines[s:e])
            chunk = section_text[: max(0, max_chars - used)]
            if not chunk:
                continue
            pieces.append(chunk)
            used += len(chunk) + 2
            if used >= max_chars:
                return "\n".join(pieces)[:max_chars]
    if not pieces:
        return context[:max_chars]
    return "\n".join(pieces)[:max_chars]


def build_ticker_index(
    dataset_dir: Path,
    primary_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Index every dataset record (across train/val/test) by ticker for
    cross-evidence lookup."""
    seen_ids: set[str] = set()
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files_to_scan: list[Path] = []
    for name in EVIDENCE_FILES:
        p = dataset_dir / name
        if p.is_file():
            files_to_scan.append(p)
    if primary_path not in files_to_scan and primary_path.is_file():
        files_to_scan.append(primary_path)

    for p in files_to_scan:
        for record in load_jsonl(p):
            rec_id = record.get("id")
            if rec_id and rec_id in seen_ids:
                continue
            if rec_id:
                seen_ids.add(rec_id)
            ticker = (record.get("ticker") or "").upper()
            if not ticker:
                continue
            slim = {
                "id": record.get("id"),
                "ticker": ticker,
                "cik": record.get("cik"),
                "form": record.get("form"),
                "report_date": record.get("report_date"),
                "fiscal_year": record.get("fiscal_year"),
                "fiscal_period": record.get("fiscal_period"),
                "context": record.get("context") or "",
            }
            if slim["context"]:
                index[ticker].append(slim)
    return index


def _parse_date(d: Any) -> tuple[int, int, int] | None:
    if not isinstance(d, str):
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", d.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _date_distance_days(a: Any, b: Any) -> int:
    pa, pb = _parse_date(a), _parse_date(b)
    if not pa or not pb:
        return 10_000
    from datetime import date

    try:
        return abs((date(*pa) - date(*pb)).days)
    except Exception:  # noqa: BLE001
        return 10_000


def select_cross_evidence(
    example: dict[str, Any],
    ticker_index: dict[str, list[dict[str, Any]]],
    topic: str | None,
    max_total_chars: int = MAX_AUGMENT_CHARS,
    per_snippet_chars: int = PER_AUGMENT_SNIPPET,
    max_snippets: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    """Build augmented evidence from same-ticker peer records."""
    ticker = (example.get("ticker") or "").upper()
    if not ticker or not ticker_index:
        return "", []

    own_id = example.get("id")
    own_ctx = example.get("context") or ""
    own_ctx_first = own_ctx[:200]
    peers = [
        rec
        for rec in ticker_index.get(ticker, [])
        if rec.get("id") != own_id and rec.get("context", "")[:200] != own_ctx_first
    ]
    if not peers:
        return "", []

    target_date = example.get("report_date")

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for rec in peers:
        ctx = rec.get("context", "")
        topic_match = _has_relevant_data(ctx, topic)
        fs_present = _has_real_fs(ctx)
        if not (topic_match or fs_present):
            continue
        date_dist = _date_distance_days(target_date, rec.get("report_date"))
        bs_topic = topic in {"assets", "liabilities", "equity", "cash"}
        bonus = 0
        if bs_topic:
            tgt = _parse_date(target_date)
            peer = _parse_date(rec.get("report_date"))
            if tgt and peer and 0 < (peer[0] * 12 + peer[1]) - (tgt[0] * 12 + tgt[1]) <= 15:
                bonus += 50
        score = (
            (200 if topic_match else 0)
            + (50 if fs_present else 0)
            + bonus
            - min(date_dist, 1500)
        )
        scored.append((score, date_dist, rec))

    scored.sort(key=lambda t: (-t[0], t[1]))

    snippets: list[str] = []
    snippet_meta: list[dict[str, Any]] = []
    used_chars = 0
    for _score, _dist, rec in scored:
        if len(snippets) >= max_snippets:
            break
        snippet_budget = min(per_snippet_chars, max_total_chars - used_chars)
        if snippet_budget < 400:
            break
        extract = _topic_extract(rec.get("context", ""), topic, snippet_budget)
        if not extract or not extract.strip():
            continue
        provenance = (
            f"[Source: ticker={rec.get('ticker')} form={rec.get('form')} "
            f"report_date={rec.get('report_date')} fiscal_year={rec.get('fiscal_year')} "
            f"fiscal_period={rec.get('fiscal_period')} id={rec.get('id')}]"
        )
        snippet = f"{provenance}\n{extract}"
        snippets.append(snippet)
        snippet_meta.append(
            {
                "id": rec.get("id"),
                "form": rec.get("form"),
                "report_date": rec.get("report_date"),
                "fiscal_year": rec.get("fiscal_year"),
                "fiscal_period": rec.get("fiscal_period"),
                "chars": len(snippet),
            }
        )
        used_chars += len(snippet) + 2

    if not snippets:
        return "", []

    header = (
        "--- AUXILIARY EVIDENCE: other public filings of the same company ---\n"
        "Use these to corroborate or find figures missing from the primary "
        "excerpt. A later 10-Q's balance sheet typically prints the requested "
        "year-end as its comparative column.\n"
    )
    return header + "\n\n".join(snippets), snippet_meta


# --------------------------------------------------------------------------- #
# Evidence-quote verification (★ Gen 4 main lever, retained)
# --------------------------------------------------------------------------- #
_EVIDENCE_NUM_RE = re.compile(r"\(?\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*\)?")


def _extract_quote_numbers(quote: str) -> list[float]:
    """Parse all numeric tokens from the evidence quote, returning floats with
    sign for parenthesised negatives."""
    out: list[float] = []
    if not quote:
        return out
    for m in _EVIDENCE_NUM_RE.finditer(quote):
        raw = m.group(1).replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        start = m.start()
        end = m.end()
        prefix = quote[max(0, start - 2): start]
        suffix = quote[end: end + 2]
        if "(" in prefix and ")" in suffix:
            v = -v
        out.append(v)
    return out


def _quote_scale_multipliers(quote: str) -> list[int]:
    """Return plausible multipliers to apply to numbers in the quote."""
    if not quote:
        return [1]
    low = quote.lower()
    mults: list[int] = [1]
    if "billion" in low or "bn" in low:
        mults.append(1_000_000_000)
    if "million" in low or " mm" in low or "(in millions" in low or " in millions" in low:
        mults.append(1_000_000)
    if "thousand" in low or "(in thousands" in low or " in thousands" in low:
        mults.append(1_000)
    return mults


def _verify_answer_against_quote(
    answer: float | None,
    quote: str,
    answer_type: str,
    tolerance_frac: float = 0.001,
) -> bool:
    """Verify the answer is supported by a number in the evidence quote."""
    if answer is None or not math.isfinite(answer) or not quote:
        return False
    a = abs(answer)
    if a == 0:
        return False
    nums = _extract_quote_numbers(quote)
    if not nums:
        return False
    if _answer_type_norm(answer_type) == "currency":
        mults = _quote_scale_multipliers(quote)
        if 1_000_000 not in mults:
            mults.append(1_000_000)
        if 1_000_000_000 not in mults:
            mults.append(1_000_000_000)
        for n in nums:
            for m in mults:
                cand = abs(n) * m
                if cand == 0:
                    continue
                rel = abs(cand - a) / max(a, cand)
                if rel <= tolerance_frac:
                    return True
        return False
    for n in nums:
        for m in (1, 100, 0.01):  # cover ratio↔percent conversion
            cand = abs(n) * m
            if cand == 0:
                continue
            rel = abs(cand - a) / max(a, cand)
            if rel <= max(tolerance_frac, 0.005):
                return True
    return False


# --------------------------------------------------------------------------- #
# Evidence pre-mining (★ Gen 5: rescue candidates for sticky cases)
# --------------------------------------------------------------------------- #
_TOPIC_LINE_PATTERNS: dict[str, tuple[str, ...]] = {
    "assets": (r"\bTotal\s+(?:current\s+)?assets\b",),
    "liabilities": (r"\bTotal\s+(?:current\s+)?liabilities\b",),
    "equity": (
        r"\bTotal\s+(?:stockholders'?|shareholders'?|common[-\s]+stockholders'?|"
        r"common[-\s]+shareholders'?)\s+equity\b",
        r"\bTotal\s+equity\b",
        r"\bTotal\s+\w+\s+stockholders'?\s+equity\b",
        r"\bTotal\s+\w+\s+shareholders'?\s+equity\b",
    ),
    "revenue": (
        r"\bTotal\s+revenues?\b",
        r"\bNet\s+revenues?\b",
        r"\bNet\s+sales\b",
        r"\bRevenues?\b",
    ),
    "operating_income": (
        r"\bOperating\s+income\s*\(?\s*loss\s*\)?",
        r"\bOperating\s+(?:income|loss)\b",
        r"\bIncome\s+from\s+operations\b",
        r"\bLoss\s+from\s+operations\b",
    ),
    "net_income": (
        r"\bNet\s+income\s+attributable\s+to\b",
        r"\bNet\s+income\s+applicable\s+to\s+common\b",
        r"\bNet\s+income\s*\(?\s*loss\s*\)?",
        r"\bNet\s+(?:income|loss|earnings)\b",
    ),
    "gross_profit": (
        r"\bGross\s+profit\b",
        r"\bGross\s+margin\b",
    ),
    "diluted_eps": (
        r"\bDiluted\b",
        r"\bDiluted\s+(?:net\s+income\s+)?per\s+(?:common\s+)?share\b",
    ),
    "cash": (
        r"\bCash\s+and\s+cash\s+equivalents\b",
        r"\bCash,\s*cash\s+equivalents\b",
        r"\bCash\s+and\s+equivalents\b",
    ),
}


def _mine_topic_candidates(
    text: str, topic: str | None, max_candidates: int = 12
) -> list[tuple[float, str]]:
    """Return a list of (value, source_line) candidates by scanning ``text``
    for tabular lines whose label matches the topic. Returns the value in raw
    USD (auto-scaled by 'in millions/thousands' hints from the parent block).

    Only the numeric tokens on the SAME line as the label are returned (we
    cannot reliably tell which neighbour-line numbers belong to which label).
    """
    if not text or not topic:
        return []
    patterns = _TOPIC_LINE_PATTERNS.get(topic, ())
    if not patterns:
        return []
    lines = text.splitlines()
    candidates: list[tuple[float, str]] = []

    # detect "in millions"/"in thousands" hints scoped to nearby header.
    scale_hint = [1] * len(lines)
    current_scale = 1
    for i, line in enumerate(lines):
        low = line.lower()
        if "(in billions" in low or "in billions of" in low or "amounts in billions" in low:
            current_scale = 1_000_000_000
        elif "(in millions" in low or "in millions of" in low or "amounts in millions" in low:
            current_scale = 1_000_000
        elif "(in thousands" in low or "in thousands of" in low or "amounts in thousands" in low:
            current_scale = 1_000
        scale_hint[i] = current_scale

    pat_combined = re.compile("|".join(patterns), re.IGNORECASE)
    num_re = re.compile(r"\(?\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*\)?")
    for i, line in enumerate(lines):
        if not pat_combined.search(line):
            continue
        # Only look at numbers on this same line (avoids scoop-up of next-line
        # values that may belong to a different labelled row).
        for m in num_re.finditer(line):
            raw = m.group(1).replace(",", "")
            if "." not in raw and len(raw) < 3:
                continue  # skip tiny single/double-digit numbers
            try:
                v = float(raw)
            except ValueError:
                continue
            if v < 1:
                continue
            start = m.start()
            prefix = line[max(0, start - 2): start]
            suffix = line[m.end(): m.end() + 2]
            if "(" in prefix and ")" in suffix:
                v = -v
            scaled = v * scale_hint[i]
            candidates.append((scaled, line[:200]))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a careful financial analyst. You read excerpts from SEC 10-K and 10-Q "
    "filings of S&P 100 companies and answer numerical questions with the correct unit. "
    "You always reply with strict JSON containing keys: answer, unit, confidence, "
    "reasoning, evidence_quote. Always return a numeric `answer` (use 0 only as a "
    "last resort). The `evidence_quote` field must contain the verbatim source line(s) "
    "and column header you read the number from, exactly as they appear in the supplied "
    "context (do not invent table cells)."
)


def _answer_type_norm(answer_type: str) -> str:
    a = (answer_type or "").strip().lower().replace("/", "_")
    if a in {"usd_per_share", "usd per share", "per_share"}:
        return "usd_per_share"
    return a


def _unit_guidance(answer_type: str, expected_unit: str) -> str:
    answer_type = _answer_type_norm(answer_type)
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
    if answer_type == "usd_per_share":
        return (
            "Return the per-share dollar amount (e.g. 6.11 for $6.11/share). Use "
            "'USD_per_share' as the unit. Do not multiply by share counts."
        )
    if answer_type == "ratio":
        return (
            "Return a plain decimal ratio (e.g. 0.53 for 53%). Use 'ratio' as the unit. "
            "Do NOT multiply by 100. Compute derived ratios from the SAME period's totals "
            "in the most recent balance sheet."
        )
    return (
        f"Use unit '{expected_unit or 'USD'}'. Output a clean numeric value with no "
        "commas, currency symbols, or scale words."
    )


def _column_match_instruction(example: dict[str, Any]) -> str:
    form = (example.get("form") or "").upper()
    report_date = example.get("report_date") or ""
    fiscal_period = (example.get("fiscal_period") or "").upper()
    parts: list[str] = []
    if report_date:
        parts.append(
            f"Match the numeric column whose header refers to {report_date} "
            f"(or to the period ending on that date)."
        )
    if form == "10-Q":
        parts.append(
            "This is a 10-Q. Prefer the 'Three Months Ended' column unless the "
            "question explicitly asks for a year-to-date / six-/nine-month value."
        )
    elif form == "10-K" and fiscal_period in {"FY", ""}:
        parts.append(
            "This is a 10-K full-year report. Prefer the 'Year Ended' / "
            "'Twelve Months Ended' / fiscal-year column."
        )
    return " ".join(parts)


def _line_item_preference_note(topic: str | None, company_name: str) -> str:
    """Generic GAAP line-item preferences (★ Gen 5 single-shot extension)."""
    if topic is None:
        return ""
    company = (company_name or "the registrant").strip().strip(",")
    if topic == "equity":
        return (
            "Equity convention: When both 'Total Stockholders' Equity' / "
            "'Total Common Shareholders' Equity' (EXCLUDING non-controlling "
            "interests) AND a broader 'Total Equity' (INCLUDING NCI) are "
            "present, USE the stockholders'-equity figure unless the question "
            "explicitly asks for the broader (including NCI) value."
        )
    if topic == "net_income":
        return (
            f"Net-income convention: When both 'Net income attributable to "
            f"{company}' (or 'Net income applicable to common shareholders') "
            "AND the broader consolidated 'Net income' (including amounts "
            "attributable to non-controlling interests / preferred dividends) "
            "are present, USE the attributable-to-company / "
            "applicable-to-common-shareholders figure unless the question "
            "explicitly asks for the consolidated total."
        )
    if topic == "cash":
        return (
            "Cash convention: Use 'Cash and cash equivalents' from the balance "
            "sheet. Do NOT include 'Short-term investments' or 'Restricted "
            "cash' unless the question explicitly asks for the broader figure."
        )
    return ""


def _format_column_legend(legend: list[tuple[str, str]]) -> str:
    if not legend:
        return ""
    lines = ["Detected column legend (use these labels when identifying which cell):"]
    for label, header in legend:
        lines.append(f"  {label} = {header}")
    return "\n".join(lines)


def build_primary_messages(
    example: dict[str, Any],
    dataset_dir: Path,
    working_dir: Path,
    excerpt: str,
    excerpt_meta: dict[str, Any],
    primary_kind: str | None,
    augmented_evidence: str = "",
    period_headers: list[str] | None = None,
    column_legend: list[tuple[str, str]] | None = None,
    allow_knowledge_fallback: bool = True,
    extra_user_suffix: str = "",
    primary_has_topic: bool = True,
    topic: str | None = None,
) -> list[dict[str, str]]:
    scale_hint = _scale_hint(example.get("context", ""))
    guidance = _unit_guidance(example.get("answer_type", ""), example.get("expected_unit", ""))
    period_headers = period_headers or []
    column_legend = column_legend or []

    sandbox_note = (
        "Sandbox paths (informational only — you do not access the filesystem yourself):\n"
        f"- READ-ONLY dataset directory: {dataset_dir}\n"
        f"- READ/WRITE working directory: {working_dir}\n"
        "The orchestrating script already loaded the dataset and will write "
        "`submission.jsonl` to the working directory after collecting your answer."
    )

    primary_kind_note = (
        f"- Hint: this question is best answered from the {primary_kind.replace('_', ' ')} table."
        if primary_kind
        else ""
    )

    period_note = ""
    if period_headers:
        joined = "; ".join(period_headers[:8])
        period_note = (
            "\nCandidate period columns detected in the supplied context: "
            f"{joined}"
        )
    legend_block = _format_column_legend(column_legend)
    if legend_block:
        period_note = (period_note + "\n" + legend_block).lstrip("\n")
    column_match = _column_match_instruction(example)
    if column_match:
        period_note = (period_note + "\n" + column_match).lstrip("\n")
    line_item_pref = _line_item_preference_note(topic, example.get("company_name") or "")
    if line_item_pref:
        period_note = (period_note + "\n" + line_item_pref).lstrip("\n")

    augmented_note = ""
    if augmented_evidence:
        if primary_has_topic:
            preface = (
                "AUXILIARY EVIDENCE (peer filings of the same company). The "
                "primary excerpt already contains the requested line item; use "
                "the auxiliary evidence only to cross-check unit/scale or pick "
                "between ambiguous columns. PREFER PRECISE TABLE VALUES over "
                "rounded narrative quotes if they differ."
            )
        else:
            preface = (
                "AUXILIARY EVIDENCE (peer filings of the same company). The "
                "PRIMARY excerpt does NOT appear to contain the specific line "
                "item this question asks about; the AUXILIARY EVIDENCE is your "
                "primary source of truth. A later 10-Q balance sheet's "
                "comparative column prints the prior year-end balance, and "
                "earlier 10-Q comparatives print prior-year quarters. "
                "PREFER PRECISE TABLE VALUES over any rounded narrative quote."
            )
        augmented_note = (
            "\n\n" + preface + "\n\"\"\"\n" + augmented_evidence + "\n\"\"\"\n"
        )

    knowledge_clause = ""
    if allow_knowledge_fallback:
        knowledge_clause = (
            "\n\nFallback rule (use sparingly):\n"
            "- If — and ONLY if — neither the primary excerpt nor the auxiliary "
            "evidence contains the requested line item, you may use your training-data "
            "knowledge of this S&P 100 company's publicly filed 10-K/10-Q to give your "
            "best numeric estimate. In that case set `confidence` ≤ 0.30 and say so "
            "in `reasoning`. The `evidence_quote` field MUST contain the literal text "
            "'NO_EVIDENCE_IN_CONTEXT' in that case.\n"
            "- Always return a numeric value, never `null`."
        )

    user_prompt = f"""{sandbox_note}

You are answering ONE numerical question about an SEC filing excerpt.

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
{primary_kind_note}{period_note}

Unit / normalisation rule for this question:
{guidance}

General rules:
- Use the source that actually contains the requested line item. Cite the
  source in `reasoning` (e.g. "from 2026-Q1 10-Q comparative column").
- For 10-Q questions about a SPECIFIC quarter ended on a given date, pick the
  column whose period header matches that quarter exactly (e.g. "Three Months
  Ended"), NOT the year-to-date column.
- For balance-sheet questions about the end of a period, take the column whose
  date header matches the report_date.
- For derived ratios (e.g., "liabilities-to-assets") compute the ratio from the
  SAME period's totals in the most recent balance sheet.
- Confidence is a float in [0, 1] reflecting how sure you are.
- Reasoning is a SHORT phrase describing the source line(s) and any
  unit conversion you applied.
- `evidence_quote` MUST be the verbatim source line(s) you copied the value
  from, including the column header you used; if no evidence is in either
  context, set it to "NO_EVIDENCE_IN_CONTEXT".{knowledge_clause}

Respond with strict JSON only, exactly in this shape:
{{
  "answer": <number>,
  "unit": "<unit string>",
  "confidence": <float 0-1>,
  "reasoning": "<short evidence string>",
  "evidence_quote": "<verbatim source line(s) including column header>"
}}

Question: {example.get('question')}

Primary filing context excerpt:
\"\"\"
{excerpt}
\"\"\"
{augmented_note}{extra_user_suffix}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_components_messages(
    example: dict[str, Any],
    excerpt: str,
    numerator_name: str,
    denominator_name: str,
    primary_kind: str | None,
    augmented_evidence: str = "",
    prefer_stockholders_equity: bool = False,
    prefer_attributable_net_income: bool = False,
    out_unit: str = "USD",
    is_margin: bool = False,
) -> list[dict[str, str]]:
    primary_kind_note = (
        f"- Hint: extract from the {primary_kind.replace('_', ' ')} table."
        if primary_kind
        else ""
    )
    augmented_note = ""
    if augmented_evidence:
        augmented_note = (
            "\n\nAUXILIARY EVIDENCE (peer filings of the same company; use to "
            "find values missing from the primary excerpt — a 10-Q balance "
            "sheet's comparative column prints prior year-end totals):\n\"\"\"\n"
            + augmented_evidence
            + "\n\"\"\"\n"
        )
    conventions: list[str] = []
    if prefer_stockholders_equity:
        conventions.append(
            "Equity convention: When both 'Total Stockholders' Equity' (excluding "
            "non-controlling interests) and 'Total Equity' (including NCI) are present, "
            "use 'Total Stockholders' Equity' for the equity figure unless the question "
            "asks for the broader figure explicitly."
        )
    if prefer_attributable_net_income:
        company = (example.get("company_name") or "the registrant").strip().strip(",")
        conventions.append(
            f"Net-income convention: When both 'Net income attributable to {company}' "
            "(or 'Net income applicable to common shareholders') AND consolidated "
            "'Net income' are present, use the attributable-to-company figure unless the "
            "question explicitly asks for consolidated net income."
        )
    convention_block = "\n".join(conventions)

    period_match = ""
    if is_margin:
        period_match = (
            "- For margin components, take the income and revenue from the SAME period "
            "column (e.g. the same Three Months Ended or Year Ended date)."
        )
    else:
        period_match = (
            "- Take the values from the SAME period (the most recent balance sheet "
            "column or the column that matches the report date)."
        )

    user_prompt = f"""You are extracting two raw {out_unit} values from an SEC filing excerpt so we can
compute a derived value in Python. Do not compute the result yourself.

Question metadata:
- ID: {example.get('id')}
- Company: {example.get('company_name')} ({example.get('ticker')})
- Form: {example.get('form')}
- Report date: {example.get('report_date')}
- Fiscal year: {example.get('fiscal_year')}; fiscal period: {example.get('fiscal_period')}
{primary_kind_note}
{convention_block}
Extraction rules:
{period_match}
- Return raw {out_unit} numbers. If the filing reports "in millions", multiply by 1,000,000.
- If a value is not found in the primary excerpt, use the AUXILIARY EVIDENCE
  (peer filings of the same company); a 10-Q balance sheet's comparative column
  prints the prior year-end values.
- If a value is truly not found anywhere, write the numeric value 0 and explain
  in `notes`. Never return null; always emit a number.
- `evidence_quote` MUST be the verbatim source line(s) you used (one per value),
  including the column header.

Respond with strict JSON only, exactly in this shape:
{{
  "{numerator_name}": <number>,
  "{denominator_name}": <number>,
  "unit": "{out_unit}",
  "confidence": <float 0-1>,
  "notes": "<short evidence string>",
  "evidence_quote": "<verbatim source line(s) with column header(s)>"
}}

Primary filing context excerpt:
\"\"\"
{excerpt}
\"\"\"
{augmented_note}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_focused_messages(
    example: dict[str, Any],
    topic_evidence: str,
    candidate_hint: str,
    topic: str | None,
    answer_type: str,
) -> list[dict[str, str]]:
    """A minimal-boilerplate prompt used for the §2.6 rescue attempt."""
    guidance = _unit_guidance(example.get("answer_type", ""), example.get("expected_unit", ""))
    line_item_pref = _line_item_preference_note(topic, example.get("company_name") or "")
    company = example.get("company_name") or ""
    user = f"""You are answering ONE numerical question.

Company: {company} ({example.get('ticker')})
Form: {example.get('form')}; report_date: {example.get('report_date')}; fiscal_period: {example.get('fiscal_period')}; fiscal_year: {example.get('fiscal_year')}
Answer type: {example.get('answer_type')}; expected unit: {example.get('expected_unit')}

Unit rule: {guidance}
{line_item_pref}

Question: {example.get('question')}

Relevant evidence (already extracted by topic):
\"\"\"
{topic_evidence}
\"\"\"
{candidate_hint}
Respond with strict JSON only:
{{"answer": <number>, "unit": "<unit>", "confidence": <0-1>, "reasoning": "<short>", "evidence_quote": "<verbatim source line>"}}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Model output parsing & normalisation
# --------------------------------------------------------------------------- #
def _safe_json_loads(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                obj = json.loads(cleaned)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


# Regex patterns for the partial-JSON recovery path (★ Gen 5).
_PARTIAL_NUM_FIELD_RE = re.compile(
    r'"\s*(answer|confidence)\s*"\s*:\s*("?)\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b',
    re.IGNORECASE,
)
_PARTIAL_STR_FIELD_RE = re.compile(
    r'"\s*(unit|reasoning|evidence_quote|notes)\s*"\s*:\s*"((?:[^"\\]|\\.)*)',
    re.IGNORECASE,
)


def _recover_partial_json(text: str) -> dict[str, Any] | None:
    """Mine field values from a possibly-truncated JSON-like assistant response.

    Returns a dict with whichever of {answer, unit, confidence, reasoning,
    evidence_quote, notes} could be recovered.
    """
    text = (text or "").strip()
    if not text:
        return None
    # Strip a leading [label] tag if present
    text = re.sub(r"^\s*\[[a-zA-Z_]+\]\s*", "", text)
    recovered: dict[str, Any] = {}
    for m in _PARTIAL_NUM_FIELD_RE.finditer(text):
        key = m.group(1).lower()
        raw = m.group(3)
        try:
            val: Any = float(raw)
        except ValueError:
            continue
        if key == "answer" and val.is_integer() and abs(val) >= 1:
            val = int(val)
        recovered[key] = val
    for m in _PARTIAL_STR_FIELD_RE.finditer(text):
        key = m.group(1).lower()
        # group 2 is the inside of the string up to the next unescaped " or EOF
        raw = m.group(2)
        # Unescape any \" pairs etc.
        try:
            recovered[key] = bytes(raw, "utf-8").decode("unicode_escape")
        except Exception:
            recovered[key] = raw
    # Also try a separate numeric extraction for `answer` if it had a quote-wrapped
    # number like "answer": "$681,000,000"
    if "answer" not in recovered:
        m = re.search(
            r'"\s*answer\s*"\s*:\s*"([^"]*?)"',
            text,
            re.IGNORECASE,
        )
        if m:
            recovered["answer"] = m.group(1)
    if not recovered:
        return None
    return recovered


def _parse_response(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse the assistant text, falling back to partial-JSON recovery.

    Returns (parsed_dict, was_partial).
    """
    body = (text or "").strip()
    if body.startswith("["):
        # Strip leading [label] tag if present, since trajectories may include it
        body = re.sub(r"^\s*\[[a-zA-Z_]+\]\s*", "", body)
    full = _safe_json_loads(body)
    if full is not None:
        return full, False
    partial = _recover_partial_json(body)
    if partial is not None:
        return partial, True
    return None, False


_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _coerce_number(value: Any, unit_hint: str = "") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "n/a", "na", "-"}:
        return None
    negative = False
    if re.fullmatch(r"\(.*\)", text):
        negative = True
        text = text[1:-1]
    cleaned = text.replace("$", "").replace(",", "").replace("%", "")
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None
    try:
        num = float(match.group(0))
    except ValueError:
        return None
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
    answer_type = _answer_type_norm(answer_type)
    if answer_type == "currency":
        return "USD"
    if answer_type == "percent":
        return "percent"
    if answer_type == "usd_per_share":
        return "USD_per_share"
    if answer_type == "ratio":
        return "ratio"
    if isinstance(model_unit, str) and model_unit.strip():
        return model_unit.strip()
    return expected_unit or ""


def _post_process_answer(
    answer_type: str, expected_unit: str, raw_answer: Any, raw_unit: Any
) -> float | int | None:
    answer_type = _answer_type_norm(answer_type)
    raw_unit_str = str(raw_unit or "")
    number = _coerce_number(raw_answer, raw_unit_str)
    if number is None:
        return None
    if answer_type == "currency":
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
        if abs(number) > 5 and ("%" in raw_unit_str or "percent" in raw_unit_str.lower()):
            number /= 100.0
        return number
    if answer_type == "percent":
        if 0 < abs(number) <= 1 and "ratio" in raw_unit_str.lower():
            number *= 100.0
        return number
    return number


def _looks_invalid(answer_type: str, number: float | None) -> bool:
    if number is None or not math.isfinite(number or 0):
        return True
    a = _answer_type_norm(answer_type)
    if a == "currency":
        return number == 0
    if a == "usd_per_share":
        return abs(number) > 10000 or number == 0
    if a == "ratio":
        return abs(number) > 100
    if a == "percent":
        return abs(number) > 10000
    return False


def _magnitude_clashes_with_evidence(
    answer_type: str, number: float | None, evidence_text: str
) -> bool:
    if _answer_type_norm(answer_type) != "currency":
        return False
    if number is None or number == 0:
        return False
    big_nums = []
    for m in re.finditer(r"\$?\s*([\d,]{4,})", evidence_text or ""):
        try:
            v = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if v >= 1000:
            big_nums.append(v * 1_000_000)
            big_nums.append(v)
    if not big_nums:
        return False
    target = abs(number)
    closest = min(big_nums, key=lambda v: abs(v - target) / max(target, 1))
    if closest == 0:
        return False
    ratio = max(target, closest) / max(min(target, closest), 1)
    return ratio > 100


def _ratio_components(question: str) -> tuple[str, str, str | None] | None:
    q = (question or "").lower()
    if "liabilities-to-assets" in q or "liabilities to assets" in q:
        return ("total_liabilities", "total_assets", "ratio")
    if "equity ratio" in q or "equity-to-assets" in q or "equity to assets" in q:
        return ("total_stockholders_equity", "total_assets", "ratio")
    if "debt-to-equity" in q or "debt to equity" in q:
        return ("total_debt", "total_stockholders_equity", "ratio")
    if "current ratio" in q:
        return ("current_assets", "current_liabilities", "ratio")
    if "quick ratio" in q:
        return ("quick_assets", "current_liabilities", "ratio")
    return None


def _margin_components(question: str) -> tuple[str, str] | None:
    """Return (numerator, denominator) for percent margin questions, or None.

    ★ Gen 5: extends the ratio-components pattern to percent margins.
    """
    q = (question or "").lower()
    if "operating margin" in q or "operating loss margin" in q:
        return ("operating_income", "total_revenue")
    if "net margin" in q or "net loss margin" in q or "net income margin" in q:
        return ("net_income", "total_revenue")
    if "gross margin" in q:
        return ("gross_profit", "total_revenue")
    return None


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
    *,
    force_json: bool = True,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
) -> tuple[str, dict[str, int], str]:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
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
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            err_text = str(exc).lower()
            if force_json and "response_format" in err_text:
                force_json = False
                continue
            sleep_for = min(2 ** attempt + random.random(), 15)
            time.sleep(sleep_for)
    raise RuntimeError(f"fugu-mini call failed after {MAX_RETRIES} attempts: {last_exc}")


def _aggregate_usage(parts: list[dict[str, int]]) -> dict[str, int]:
    agg: dict[str, int] = {}
    for u in parts:
        for k, v in (u or {}).items():
            agg[k] = agg.get(k, 0) + int(v)
    return agg


def _fallback_prediction(example: dict[str, Any], reason: str) -> dict[str, Any]:
    answer_type = _answer_type_norm(example.get("answer_type", ""))
    if answer_type == "currency":
        unit = "USD"
    elif answer_type == "percent":
        unit = "percent"
    elif answer_type == "usd_per_share":
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


def _build_final_pred(
    example: dict[str, Any],
    raw_answer: Any,
    raw_unit: Any,
    confidence_raw: Any,
    reasoning: Any,
) -> tuple[dict[str, Any], float | None]:
    answer_type = example.get("answer_type", "")
    expected_unit = example.get("expected_unit", "")
    normalised = _post_process_answer(answer_type, expected_unit, raw_answer, raw_unit)
    unit = _normalise_unit(answer_type, expected_unit, raw_unit)

    if normalised is None:
        final_answer: Any = raw_answer if raw_answer is not None else 0
    else:
        if abs(normalised - round(normalised)) < 1e-6 and abs(normalised) >= 1:
            final_answer = int(round(normalised))
        else:
            final_answer = float(normalised)

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    if not isinstance(reasoning, str):
        reasoning = json.dumps(reasoning) if reasoning is not None else ""

    return {
        "id": example.get("id"),
        "answer": final_answer,
        "unit": unit,
        "confidence": confidence,
        "reasoning": reasoning[:600],
    }, normalised


def _ratio_from_components(
    parsed: dict[str, Any],
    numerator_key: str,
    denominator_key: str,
    raw_unit: Any,
) -> tuple[float | None, float | None, float | None]:
    num = _coerce_number(parsed.get(numerator_key), str(raw_unit or ""))
    den = _coerce_number(parsed.get(denominator_key), str(raw_unit or ""))
    if num is None or den is None or den == 0 or num == 0:
        return None, num, den
    return num / den, num, den


def _pick_better_answer(
    a: dict[str, Any] | None,
    a_num: float | None,
    a_verified: bool,
    b: dict[str, Any] | None,
    b_num: float | None,
    b_verified: bool,
    answer_type: str,
) -> tuple[dict[str, Any] | None, float | None, bool]:
    a_ok = a is not None and not _looks_invalid(answer_type, a_num)
    b_ok = b is not None and not _looks_invalid(answer_type, b_num)
    if not a_ok and not b_ok:
        return a, a_num, a_verified
    if a_ok and not b_ok:
        return a, a_num, a_verified
    if b_ok and not a_ok:
        return b, b_num, b_verified
    if a_verified and not b_verified:
        return a, a_num, True
    if b_verified and not a_verified:
        return b, b_num, True
    ac = a.get("confidence", 0.0) if a else 0.0
    bc = b.get("confidence", 0.0) if b else 0.0
    if bc > ac:
        return b, b_num, b_verified
    return a, a_num, a_verified


def _snap_to_precise_evidence(
    answer_type: str,
    candidate: float | None,
    candidates_from_evidence: list[tuple[float, str]],
    tol_frac: float = 0.005,
) -> tuple[float | None, str | None]:
    """If the model's candidate currency answer is within ``tol_frac`` of a
    pre-mined precise tabular value, prefer the precise value.

    Returns (snapped_value, source_line) or (candidate, None) if no snap.
    """
    if _answer_type_norm(answer_type) != "currency":
        return candidate, None
    if candidate is None or candidate == 0 or not candidates_from_evidence:
        return candidate, None
    a = abs(candidate)
    best_match: tuple[float, str] | None = None
    best_rel = tol_frac + 1
    for val, line in candidates_from_evidence:
        v = abs(val)
        if v == 0:
            continue
        rel = abs(v - a) / max(v, a)
        if rel <= tol_frac and rel < best_rel:
            best_match = (val if candidate >= 0 else -abs(val), line)
            best_rel = rel
    if best_match is None:
        return candidate, None
    snapped, line = best_match
    # Only snap if precise value has more significant digits than the candidate.
    # Heuristic: snap if the candidate's last 3+ digits are all zero AND the
    # precise value is not. Otherwise keep the original.
    if candidate == int(candidate):
        cand_str = str(int(abs(candidate)))
    else:
        cand_str = str(abs(candidate))
    if snapped == int(snapped):
        snap_str = str(int(abs(snapped)))
    else:
        snap_str = str(abs(snapped))
    cand_trailing_zeros = len(cand_str) - len(cand_str.rstrip("0"))
    snap_trailing_zeros = len(snap_str) - len(snap_str.rstrip("0"))
    if cand_trailing_zeros > snap_trailing_zeros:
        return snapped, line
    return candidate, None


def _largest_topic_number(
    candidates_from_evidence: list[tuple[float, str]],
) -> tuple[float | None, str | None]:
    """Pick the largest plausible topic-line number as a last-resort rescue."""
    if not candidates_from_evidence:
        return None, None
    best = max(candidates_from_evidence, key=lambda kv: abs(kv[0]))
    return best[0], best[1]


def process_example(
    example: dict[str, Any],
    client: Any | None,
    dataset_dir: Path,
    working_dir: Path,
    ticker_index: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Generate one prediction. Returns (pred, trajectory, meta)."""
    question = example.get("question", "")
    primary_kind = _primary_statement_for(question)
    topic = _relevance_topic(question)

    excerpt, excerpt_meta = context_excerpt(
        example.get("context", ""), question, primary_kind=primary_kind
    )
    period_headers = _extract_period_headers(example.get("context", ""))
    column_legend = _extract_column_legend(
        example.get("context", ""), example.get("report_date") or ""
    )

    own_ctx = example.get("context", "")
    own_ctx_has_topic = _has_real_line_item(own_ctx, topic)
    own_has_bs = _has_balance_sheet(own_ctx)

    answer_type = _answer_type_norm(example.get("answer_type", ""))

    # Decide whether to engage cross-evidence augmentation
    is_ratio = answer_type == "ratio"
    is_margin = answer_type == "percent" and _margin_components(question) is not None
    needs_bs_augment = is_ratio and not own_has_bs
    augmented_text = ""
    augmented_meta: list[dict[str, Any]] = []
    if (not own_ctx_has_topic) or needs_bs_augment or is_margin:
        augmented_text, augmented_meta = select_cross_evidence(
            example, ticker_index, topic
        )

    # Pre-mine topic candidates from primary + augmented evidence
    mined_candidates = _mine_topic_candidates(own_ctx, topic) + _mine_topic_candidates(
        augmented_text, topic
    )

    trajectory: list[dict[str, Any]] = []
    usage_parts: list[dict[str, int]] = []
    meta: dict[str, Any] = {
        "id": example.get("id"),
        "model": MODEL,
        "primary_kind": primary_kind,
        "relevance_topic": topic,
        "excerpt_meta": excerpt_meta,
        "own_context_has_topic": own_ctx_has_topic,
        "own_has_bs": own_has_bs,
        "augmentation": augmented_meta,
        "augmented_chars": len(augmented_text),
        "period_headers": period_headers,
        "column_legend": [list(t) for t in column_legend],
        "attempts": 0,
        "knowledge_fallback_used": False,
        "ratio_components_used": False,
        "margin_components_used": False,
        "magnitude_retry_used": False,
        "self_consistency_used": False,
        "quote_verified": False,
        "partial_json_recovered": 0,
        "truncation_retries": 0,
        "currency_rescue_used": False,
        "narrative_snap_used": False,
        "mined_candidates_count": len(mined_candidates),
        "usage": {},
        "cost": 0,
        "finish_reason": "",
        "ok": False,
    }

    def _record_messages(msgs: list[dict[str, str]]) -> None:
        for m in msgs:
            trajectory.append(
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            )

    def _record_assistant(content: str, note: str | None = None) -> None:
        text = content if not note else f"[{note}] {content}"
        trajectory.append(
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        )

    if client is None:
        reason = f"missing {API_KEY_ENV} or `openai` package; using zero placeholder"
        pred = _fallback_prediction(example, reason)
        _record_messages(
            build_primary_messages(
                example,
                dataset_dir,
                working_dir,
                excerpt,
                excerpt_meta,
                primary_kind,
                augmented_evidence=augmented_text,
                period_headers=period_headers,
                column_legend=column_legend,
                primary_has_topic=own_ctx_has_topic,
                topic=topic,
            )
        )
        _record_assistant(json.dumps(pred), "no_client")
        meta["error"] = reason
        return pred, trajectory, meta

    primary_pred: dict[str, Any] | None = None
    primary_number: float | None = None
    primary_verified: bool = False

    # ----------------------------------------------------------------- #
    # Attempt 1: deterministic ratio extraction for derived ratios,
    #            or deterministic margin extraction for percent margins.
    # ----------------------------------------------------------------- #
    components = None
    is_margin_components = False
    if is_ratio:
        components = _ratio_components(question)
    elif is_margin:
        margin = _margin_components(question)
        if margin is not None:
            components = (margin[0], margin[1], "percent")
            is_margin_components = True

    if components is not None:
        num_name, den_name, out_label = components
        prefer_se = "equity" in num_name or "equity" in den_name
        prefer_attr_ni = "net_income" in num_name
        comp_msgs = build_components_messages(
            example,
            excerpt,
            num_name,
            den_name,
            primary_kind,
            augmented_evidence=augmented_text,
            prefer_stockholders_equity=prefer_se,
            prefer_attributable_net_income=prefer_attr_ni,
            out_unit="USD",
            is_margin=is_margin_components,
        )
        _record_messages(comp_msgs)
        meta["attempts"] += 1
        if is_margin_components:
            meta["margin_components_used"] = True
        else:
            meta["ratio_components_used"] = True
        try:
            content, usage, finish_reason = _call_model(client, comp_msgs)
            usage_parts.append(usage)
            meta["finish_reason"] = finish_reason
            _record_assistant(content, "ratio_components" if not is_margin_components else "margin_components")
            parsed, was_partial = _parse_response(content)
            if was_partial:
                meta["partial_json_recovered"] += 1
            if parsed is None:
                parsed = {}
            ratio, num_val, den_val = _ratio_from_components(
                parsed, num_name, den_name, parsed.get("unit")
            )
            if ratio is not None and math.isfinite(ratio):
                # For percent margins, multiply by 100
                if is_margin_components:
                    derived_value = ratio * 100.0
                    out_unit_label = "percent"
                else:
                    derived_value = ratio
                    out_unit_label = "ratio"
                pred, num = _build_final_pred(
                    example,
                    derived_value,
                    out_unit_label,
                    parsed.get("confidence", 0.6),
                    f"Computed {num_name}/{den_name} from extracted values "
                    f"{parsed.get(num_name)}, {parsed.get(den_name)}.",
                )
                primary_pred = pred
                primary_number = num
                quote = parsed.get("evidence_quote") or ""
                if quote:
                    qnums = _extract_quote_numbers(str(quote))
                    if qnums:
                        def _close(v, tol=0.005):
                            if v is None:
                                return False
                            av = abs(v)
                            for n in qnums:
                                if n == 0:
                                    continue
                                for m in (1, 1_000, 1_000_000, 1_000_000_000):
                                    cand = abs(n) * m
                                    if cand == 0:
                                        continue
                                    rel = abs(cand - av) / max(av, cand)
                                    if rel <= tol:
                                        return True
                            return False
                        if _close(num_val) and _close(den_val):
                            primary_verified = True
                            meta["quote_verified"] = True
        except Exception as exc:  # noqa: BLE001
            _record_assistant(f"components call failed: {exc}", "error")

    # ----------------------------------------------------------------- #
    # Attempt 2 (or 1 for non-component): standard single-shot extraction.
    # ----------------------------------------------------------------- #
    if primary_pred is None or _looks_invalid(answer_type, primary_number):
        primary_msgs = build_primary_messages(
            example,
            dataset_dir,
            working_dir,
            excerpt,
            excerpt_meta,
            primary_kind,
            augmented_evidence=augmented_text,
            period_headers=period_headers,
            column_legend=column_legend,
            allow_knowledge_fallback=True,
            primary_has_topic=own_ctx_has_topic,
            topic=topic,
        )
        _record_messages(primary_msgs)
        meta["attempts"] += 1
        try:
            content, usage, finish_reason = _call_model(client, primary_msgs)
            usage_parts.append(usage)
            meta["finish_reason"] = finish_reason
            _record_assistant(content, "primary")
            parsed, was_partial = _parse_response(content)
            if was_partial:
                meta["partial_json_recovered"] += 1
            if parsed is None:
                parsed = {}
            pred, num = _build_final_pred(
                example,
                parsed.get("answer"),
                parsed.get("unit") or example.get("expected_unit", ""),
                parsed.get("confidence", 0.5),
                parsed.get("reasoning") or "",
            )
            quote = str(parsed.get("evidence_quote") or "")
            verified = _verify_answer_against_quote(num, quote, answer_type)
            # Always take the better candidate
            primary_pred, primary_number, primary_verified = _pick_better_answer(
                primary_pred, primary_number, primary_verified,
                pred, num, verified, answer_type,
            )
            meta["quote_verified"] = primary_verified
            if finish_reason == "length":
                meta["truncation_retries"] += 1
        except Exception as exc:  # noqa: BLE001
            _record_assistant(f"primary call failed: {exc}", "error")

    # ----------------------------------------------------------------- #
    # Attempt 3: self-consistency retry for currency, magnitude retry, or
    #            generic retry if primary failed.
    # ----------------------------------------------------------------- #
    primary_conf = float((primary_pred or {}).get("confidence", 0.0))
    need_retry = primary_pred is None or _looks_invalid(answer_type, primary_number)
    magnitude_bad = (
        not need_retry
        and primary_pred is not None
        and augmented_text
        and _magnitude_clashes_with_evidence(answer_type, primary_number, augmented_text)
    )
    self_consistency_trigger = (
        not need_retry
        and answer_type == "currency"
        and primary_pred is not None
        and (primary_conf < 0.6 or not primary_verified)
        and (own_ctx_has_topic or augmented_text)
    )
    if magnitude_bad:
        need_retry = True
        meta["magnitude_retry_used"] = True
    elif self_consistency_trigger:
        need_retry = True
        meta["self_consistency_used"] = True

    if need_retry:
        whole_excerpt = (example.get("context") or "")[:MAX_CONTEXT_CHARS]
        retry_augmented_text = augmented_text
        retry_augmented_meta = augmented_meta
        retry_primary_has_topic = own_ctx_has_topic
        if not retry_augmented_text:
            retry_augmented_text, retry_augmented_meta = select_cross_evidence(
                example, ticker_index, topic
            )
            if retry_augmented_text:
                meta["augmentation"] = retry_augmented_meta
                meta["augmented_chars"] = len(retry_augmented_text)
                retry_primary_has_topic = False
                # Re-mine candidates with the new evidence
                mined_candidates = _mine_topic_candidates(
                    own_ctx, topic
                ) + _mine_topic_candidates(retry_augmented_text, topic)
                meta["mined_candidates_count"] = len(mined_candidates)

        if magnitude_bad or primary_pred is None:
            retry_suffix = (
                "\n\nIMPORTANT: Your previous attempt produced an implausible or missing "
                "value. Re-check the AUXILIARY EVIDENCE carefully (a 10-Q balance sheet's "
                "comparative column prints the prior year-end totals). If neither the "
                "primary nor the auxiliary evidence shows the figure, use your best "
                "training-data knowledge of this company's public filings; always emit a "
                "non-null numeric answer. The `evidence_quote` field MUST contain the "
                "verbatim source line(s) and column header you used."
            )
            meta["knowledge_fallback_used"] = True
        else:
            retry_suffix = (
                "\n\nSELF-CONSISTENCY CHECK: Your previous answer was "
                f"{primary_pred.get('answer')} with confidence {primary_conf:.2f}. "
                "Re-extract the value with extra care: pick exactly the table cell whose "
                "row matches the requested line item AND whose column header matches the "
                f"requested period ({example.get('report_date')}; "
                f"{example.get('fiscal_period') or 'FY'} of fiscal {example.get('fiscal_year')}). "
                "Quote the row and column header verbatim in `evidence_quote`."
            )
        retry_msgs = build_primary_messages(
            example,
            dataset_dir,
            working_dir,
            whole_excerpt,
            {"raw_chars": len(example.get("context") or ""), "trimmed": False},
            primary_kind,
            augmented_evidence=retry_augmented_text,
            period_headers=period_headers,
            column_legend=column_legend,
            allow_knowledge_fallback=True,
            extra_user_suffix=retry_suffix,
            primary_has_topic=retry_primary_has_topic,
            topic=topic,
        )
        _record_messages(retry_msgs)
        meta["attempts"] += 1
        try:
            content, usage, finish_reason = _call_model(client, retry_msgs)
            usage_parts.append(usage)
            meta["finish_reason"] = finish_reason
            _record_assistant(content, "retry")
            parsed, was_partial = _parse_response(content)
            if was_partial:
                meta["partial_json_recovered"] += 1
            if parsed is None:
                parsed = {}
            pred, num = _build_final_pred(
                example,
                parsed.get("answer"),
                parsed.get("unit") or example.get("expected_unit", ""),
                parsed.get("confidence", 0.3),
                parsed.get("reasoning") or "",
            )
            quote = str(parsed.get("evidence_quote") or "")
            verified = _verify_answer_against_quote(num, quote, answer_type)
            if pred and num is not None and not _looks_invalid(answer_type, num):
                if not retry_augmented_text and quote.strip().upper() == "NO_EVIDENCE_IN_CONTEXT":
                    pred["confidence"] = min(pred.get("confidence", 0.3), 0.3)
                if self_consistency_trigger and not magnitude_bad:
                    primary_pred, primary_number, primary_verified = _pick_better_answer(
                        primary_pred, primary_number, primary_verified,
                        pred, num, verified, answer_type,
                    )
                    meta["quote_verified"] = primary_verified
                else:
                    primary_pred, primary_number, primary_verified = _pick_better_answer(
                        primary_pred, primary_number, primary_verified,
                        pred, num, verified, answer_type,
                    )
                    meta["quote_verified"] = primary_verified
            if finish_reason == "length":
                meta["truncation_retries"] += 1
        except Exception as exc:  # noqa: BLE001
            _record_assistant(f"retry failed: {exc}", "error")

    # ----------------------------------------------------------------- #
    # Attempt 4 (★ Gen 5 currency-rescue): if currency answer is still 0 /
    #           null, issue one focused minimal-prompt attempt + last-resort
    #           use the largest mined candidate.
    # ----------------------------------------------------------------- #
    if (
        answer_type in {"currency", "usd_per_share"}
        and (primary_pred is None or _looks_invalid(answer_type, primary_number))
    ):
        # Build a focused topic-evidence string
        topic_text_pieces: list[str] = []
        if augmented_text:
            topic_text_pieces.append(_topic_extract(augmented_text, topic, 3000))
        if own_ctx:
            topic_text_pieces.append(_topic_extract(own_ctx, topic, 3000))
        topic_evidence = "\n\n".join(p for p in topic_text_pieces if p)[:6000]
        if not topic_evidence.strip():
            topic_evidence = (own_ctx or "")[:6000]

        candidates_hint = ""
        if mined_candidates:
            top = sorted(set(mined_candidates), key=lambda kv: -abs(kv[0]))[:6]
            lines = [
                f"  Candidate value V{i+1} = {v:,.0f}    (from line: {ln[:120]!r})"
                for i, (v, ln) in enumerate(top)
            ]
            candidates_hint = (
                "Pre-mined candidate values for this topic (these are tabular "
                "numbers near a matching line label, already converted to raw "
                "USD using any 'in millions/thousands' header hint). If one of "
                "these matches the requested period, prefer it; otherwise "
                "explain in `reasoning`:\n" + "\n".join(lines) + "\n"
            )

        focused_msgs = build_focused_messages(
            example, topic_evidence, candidates_hint, topic, answer_type
        )
        _record_messages(focused_msgs)
        meta["attempts"] += 1
        meta["currency_rescue_used"] = True
        try:
            content, usage, finish_reason = _call_model(
                client, focused_msgs, max_tokens=MAX_TOKENS_FOCUSED
            )
            usage_parts.append(usage)
            meta["finish_reason"] = finish_reason
            _record_assistant(content, "currency_rescue")
            parsed, was_partial = _parse_response(content)
            if was_partial:
                meta["partial_json_recovered"] += 1
            if parsed is None:
                parsed = {}
            pred, num = _build_final_pred(
                example,
                parsed.get("answer"),
                parsed.get("unit") or example.get("expected_unit", ""),
                parsed.get("confidence", 0.4),
                parsed.get("reasoning") or "",
            )
            quote = str(parsed.get("evidence_quote") or "")
            verified = _verify_answer_against_quote(num, quote, answer_type)
            if pred and num is not None and not _looks_invalid(answer_type, num):
                primary_pred, primary_number, primary_verified = _pick_better_answer(
                    primary_pred, primary_number, primary_verified,
                    pred, num, verified, answer_type,
                )
                meta["quote_verified"] = primary_verified
        except Exception as exc:  # noqa: BLE001
            _record_assistant(f"currency_rescue failed: {exc}", "error")

        # Absolute last resort: pick the largest mined candidate as the answer.
        if (
            primary_pred is None or _looks_invalid(answer_type, primary_number)
        ) and mined_candidates:
            best_val, best_line = _largest_topic_number(mined_candidates)
            if best_val is not None and best_val != 0:
                pred, num = _build_final_pred(
                    example,
                    best_val,
                    "USD",
                    0.25,
                    f"Last-resort rescue: largest topic-line value mined from "
                    f"evidence: {best_line[:120]}",
                )
                primary_pred = pred
                primary_number = num
                meta["error"] = (meta.get("error") or "") + " | last-resort-rescue"

    # ----------------------------------------------------------------- #
    # Post-processing: snap rounded-narrative answers to precise tabular
    #                 numbers from the evidence (★ Gen 5)
    # ----------------------------------------------------------------- #
    if primary_pred is not None and primary_number is not None and mined_candidates:
        snapped, snap_line = _snap_to_precise_evidence(
            answer_type, primary_number, mined_candidates
        )
        if snapped is not None and snapped != primary_number:
            new_pred = dict(primary_pred)
            new_pred["answer"] = (
                int(round(snapped))
                if abs(snapped - round(snapped)) < 1e-6 and abs(snapped) >= 1
                else float(snapped)
            )
            existing_reason = new_pred.get("reasoning") or ""
            new_pred["reasoning"] = (
                f"{existing_reason} | snapped to precise evidence value "
                f"({snap_line[:80] if snap_line else 'unknown'})"
            )[:600]
            # Boost confidence if not already very high
            new_pred["confidence"] = max(new_pred.get("confidence", 0.5), 0.8)
            primary_pred = new_pred
            primary_number = float(snapped)
            meta["narrative_snap_used"] = True

    if primary_pred is None:
        primary_pred = _fallback_prediction(
            example, "no valid model output across attempts"
        )
        meta["error"] = primary_pred["reasoning"]
    else:
        meta["ok"] = True

    meta["usage"] = _aggregate_usage(usage_parts)

    # Append a synthetic final_meta record so trajectories always end with a
    # full picture of what we did.
    trajectory.append({
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "[final_meta] " + json.dumps({
                "id": meta["id"],
                "final_answer": primary_pred.get("answer"),
                "final_unit": primary_pred.get("unit"),
                "final_confidence": primary_pred.get("confidence"),
                "attempts": meta["attempts"],
                "quote_verified": meta["quote_verified"],
                "partial_json_recovered": meta["partial_json_recovered"],
                "truncation_retries": meta["truncation_retries"],
                "currency_rescue_used": meta["currency_rescue_used"],
                "margin_components_used": meta["margin_components_used"],
                "narrative_snap_used": meta["narrative_snap_used"],
                "ok": meta["ok"],
                "error": meta.get("error", ""),
                "cost": 0,
                "usage": meta["usage"],
            })
        }]
    })
    return primary_pred, trajectory, meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="SIA-FinCheck target agent (fugu-mini, gen 5)"
    )
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

    test_path = discover_dataset_file(dataset_dir)
    if test_path is None:
        print(f"ERROR: could not find any .jsonl file in {dataset_dir}", file=sys.stderr)
        return 2

    examples = load_jsonl(test_path)
    print(f"Loaded {len(examples)} examples from {test_path}")
    print(f"Model: {MODEL}; base_url: {BASE_URL}")

    ticker_index = build_ticker_index(dataset_dir, test_path)
    print(
        f"Indexed cross-evidence for {len(ticker_index)} tickers "
        f"(total peer records: {sum(len(v) for v in ticker_index.values())})"
    )

    client = make_client()
    if client is None:
        print(
            f"WARNING: no Sakana credentials available (set ${API_KEY_ENV}); "
            "writing zero fallbacks.",
            file=sys.stderr,
        )

    predictions: list[dict[str, Any] | None] = [None] * len(examples)
    metas: list[dict[str, Any] | None] = [None] * len(examples)

    workers = max(1, MAX_WORKERS) if client is not None else 1
    print_lock = threading.Lock()
    submission_path = working_dir / "submission.jsonl"
    start_time = time.time()
    completed = [0]

    def _write_trajectory(idx: int, traj: list[dict[str, Any]]) -> None:
        try:
            (exec_dir / f"execution_q{idx}.json").write_text(
                json.dumps(traj, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as write_exc:  # noqa: BLE001
            with print_lock:
                print(f"  ! could not write trajectory {idx}: {write_exc}", file=sys.stderr)

    def _flush_submission() -> None:
        snapshot: list[dict[str, Any]] = []
        for i, pred in enumerate(predictions):
            if pred is None:
                snapshot.append(_fallback_prediction(examples[i], "in-progress"))
            else:
                snapshot.append(pred)
        try:
            write_jsonl_atomic(submission_path, snapshot)
        except Exception as exc:  # noqa: BLE001
            with print_lock:
                print(f"  ! could not flush submission: {exc}", file=sys.stderr)

    def worker(idx: int) -> int:
        example = examples[idx]
        item_start = time.time()
        try:
            pred, trajectory, meta = process_example(
                example, client, dataset_dir, working_dir, ticker_index
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"unexpected error: {exc}"
            pred = _fallback_prediction(example, reason)
            trajectory = [
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(pred)}]}
            ]
            meta = {"id": example.get("id"), "ok": False, "error": reason, "usage": {}}
        predictions[idx] = pred
        metas[idx] = meta
        _write_trajectory(idx, trajectory)
        with print_lock:
            elapsed = time.time() - item_start
            completed[0] += 1
            verified_flag = "v" if meta.get("quote_verified") else " "
            print(
                f"[{completed[0]:3d}/{len(examples)}] {example.get('id')} -> "
                f"{pred.get('answer')} {pred.get('unit')} "
                f"(conf={float(pred.get('confidence', 0)):.2f}{verified_flag}, "
                f"attempts={meta.get('attempts', 0)}, "
                f"aug={meta.get('augmented_chars', 0)}c, "
                f"{elapsed:.1f}s)"
            )
            if completed[0] % FLUSH_EVERY == 0:
                _flush_submission()
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

    final_predictions: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        if pred is None:
            pred = _fallback_prediction(examples[idx], "no prediction recorded")
            predictions[idx] = pred
            _write_trajectory(
                idx,
                [{"role": "assistant", "content": [{"type": "text", "text": json.dumps(pred)}]}],
            )
        final_predictions.append(pred)

    write_jsonl_atomic(submission_path, final_predictions)

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
        "augmentation_used": sum(1 for m in metas if m and m.get("augmented_chars", 0) > 0),
        "knowledge_fallback_used": sum(
            1 for m in metas if m and m.get("knowledge_fallback_used")
        ),
        "ratio_components_used": sum(
            1 for m in metas if m and m.get("ratio_components_used")
        ),
        "margin_components_used": sum(
            1 for m in metas if m and m.get("margin_components_used")
        ),
        "magnitude_retry_used": sum(
            1 for m in metas if m and m.get("magnitude_retry_used")
        ),
        "self_consistency_used": sum(
            1 for m in metas if m and m.get("self_consistency_used")
        ),
        "currency_rescue_used": sum(
            1 for m in metas if m and m.get("currency_rescue_used")
        ),
        "narrative_snap_used": sum(
            1 for m in metas if m and m.get("narrative_snap_used")
        ),
        "partial_json_recovered": sum(
            (m or {}).get("partial_json_recovered", 0) for m in metas
        ),
        "truncation_retries": sum(
            (m or {}).get("truncation_retries", 0) for m in metas
        ),
        "quote_verified": sum(1 for m in metas if m and m.get("quote_verified")),
        "total_attempts": sum((m or {}).get("attempts", 0) for m in metas),
        "total_prompt_tokens": sum(
            ((m or {}).get("usage", {}) or {}).get("prompt_tokens", 0) for m in metas
        ),
        "total_completion_tokens": sum(
            ((m or {}).get("usage", {}) or {}).get("completion_tokens", 0) for m in metas
        ),
        "total_tokens": sum(
            ((m or {}).get("usage", {}) or {}).get("total_tokens", 0) for m in metas
        ),
        "cost": 0,
    }
    try:
        (working_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not write summary.json: {exc}", file=sys.stderr)

    print(
        f"Wrote {submission_path} with {len(final_predictions)} predictions in "
        f"{summary['runtime_seconds']}s "
        f"(ok={summary['ok_predictions']}, failed={summary['failed_predictions']}, "
        f"augmentation={summary['augmentation_used']}, "
        f"knowledge_fallback={summary['knowledge_fallback_used']}, "
        f"ratio_components={summary['ratio_components_used']}, "
        f"margin_components={summary['margin_components_used']}, "
        f"self_consistency={summary['self_consistency_used']}, "
        f"currency_rescue={summary['currency_rescue_used']}, "
        f"narrative_snap={summary['narrative_snap_used']}, "
        f"partial_json_recovered={summary['partial_json_recovered']}, "
        f"quote_verified={summary['quote_verified']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
