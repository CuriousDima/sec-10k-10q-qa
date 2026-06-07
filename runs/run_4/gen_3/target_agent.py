#!/usr/bin/env python3
"""SIA-FinCheck target agent (Generation 3) backed by Sakana AI's `fugu-mini`.

Generation 3 keeps every robustness win from Generation 2 and adds one main
structural improvement plus several smaller refinements:

1. **Cross-record context augmentation** — at startup we index every JSONL
   record in the dataset directory (train, validation, test, sample) by
   `ticker`/`cik`. When the supplied context for a question lacks the relevant
   financial-statement row, we synthesise auxiliary evidence by stitching in
   compact extracts from same-ticker peer records that DO contain the relevant
   table, tagged with their report date so the model can pick the right
   comparative column. The labels in train/validation are intentionally hidden
   by the dataset; we only ever read the *context* fields, never labels.

2. **Period-column hints** — we surface the candidate column headers we
   detected near each financial-statement section ("December 31, 2024 |
   December 31, 2023", "Three Months Ended September 27, 2025") so the model
   knows which numeric column maps to the requested `report_date`.

3. **Smarter answer-validity check** — flag answers whose magnitude clashes
   wildly with anything found in the augmented context, then re-prompt once.

4. **Atomic incremental flushing** — `submission.jsonl` is rewritten via a
   temp file + rename every 20 examples; per-question trajectories are
   written immediately, so a crash never loses progress.

The script accepts ``--dataset_dir`` (read-only) and ``--working_dir``
(read/write) on the command line. It never writes outside `working_dir`.
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
MAX_TOKENS = int(os.getenv("SIA_FINCHECK_MAX_TOKENS", "700"))
MAX_WORKERS = int(os.getenv("SIA_FINCHECK_WORKERS", "6"))
MAX_RETRIES = int(os.getenv("SIA_FINCHECK_RETRIES", "3"))
FLUSH_EVERY = int(os.getenv("SIA_FINCHECK_FLUSH_EVERY", "20"))

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

# Words that hint at the *kind* of figure being requested. Used both for
# section prioritisation and for evidence-relevance scoring during augmentation.
# All keys are matched case-insensitively, so we keep them lower-case here.
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


# Period header patterns frequently seen in 10-K/10-Q financial-statement tables.
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
    # Two or more multi-digit numbers separated by whitespace is also tabular.
    nums = re.findall(r"\b\d[\d,]{2,}\b", line)
    return len(nums) >= 2


def _extract_period_headers(
    context: str, max_headers: int = 8
) -> list[str]:
    """Collect candidate column-header strings, but only from lines that look
    like table rows / column headers OR from inside detected financial-statement
    section spans. This avoids picking up cover-page noise such as the float
    measurement date or the share-count record date in a 10-K."""
    if not context:
        return []
    # Build a set of line indices that are inside a financial-statement span.
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
# Cross-record evidence augmentation (the main Gen 3 lever)
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
    text = context
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _has_relevant_data(context: str, topic: str | None) -> bool:
    """Does this context plausibly contain the *specific* figure the question
    asks about?"""
    if not context:
        return False
    if not topic:
        return _has_real_fs(context)
    keys = _RELEVANCE_KEYS.get(topic, ())
    low = context.lower()
    for key in keys:
        # Require the keyword AND a near-by number.
        idx = low.find(key.lower())
        while idx != -1:
            window = context[idx : idx + 200]
            if re.search(r"[\d,]{4,}", window):
                return True
            idx = low.find(key.lower(), idx + 1)
    return _has_real_fs(context)


def _topic_extract(context: str, topic: str | None, max_chars: int) -> str:
    """Return a compact excerpt of the context focused on the requested topic.
    Falls back to the FS section excerpt + page prelude."""
    if not context:
        return ""
    # Prefer the financial-statement sections themselves.
    spans = _find_section_spans(context)
    if not spans:
        return context[:max_chars]

    # Map topic -> preferred section kind.
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
    # First include sections of the preferred kind.
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
    cross-evidence lookup. Records without a ticker are dropped."""
    seen_ids: set[str] = set()
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files_to_scan: list[Path] = []
    for name in EVIDENCE_FILES:
        p = dataset_dir / name
        if p.is_file():
            files_to_scan.append(p)
    # Also include the actively-answered file even if its name differs.
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
            # Drop heavy fields we don't need to keep memory small but keep context.
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
    """Approximate distance in days; returns a big number if either is unparseable."""
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
    """Build augmented evidence from same-ticker peer records.

    Returns (augmented_text, snippet_metadata).
    """
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

    # Score peers: prefer ones that contain the requested topic; then by date
    # proximity to the example's report_date.
    target_date = example.get("report_date")

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for rec in peers:
        ctx = rec.get("context", "")
        topic_match = _has_relevant_data(ctx, topic)
        fs_present = _has_real_fs(ctx)
        if not (topic_match or fs_present):
            continue
        date_dist = _date_distance_days(target_date, rec.get("report_date"))
        # Bonus when the peer's date suggests its comparative column matches.
        bs_topic = topic in {"assets", "liabilities", "equity", "cash"}
        bonus = 0
        if bs_topic:
            # A 10-Q whose report_date is in the year right AFTER target_date
            # typically has the target year-end as its comparative column.
            tgt = _parse_date(target_date)
            peer = _parse_date(rec.get("report_date"))
            if tgt and peer and 0 < (peer[0] * 12 + peer[1]) - (tgt[0] * 12 + tgt[1]) <= 15:
                bonus += 50
        score = (
            (200 if topic_match else 0)
            + (50 if fs_present else 0)
            + bonus
            - min(date_dist, 1500)  # closer dates rank higher
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
        "Use these ONLY to corroborate the figure the question asks about. The "
        "BALANCE SHEET in a later 10-Q typically prints the requested year-end "
        "as its comparative column.\n"
    )
    return header + "\n\n".join(snippets), snippet_meta


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a careful financial analyst. You read excerpts from SEC 10-K and 10-Q "
    "filings of S&P 100 companies and answer numerical questions with the correct unit. "
    "You always reply with strict JSON containing keys: answer, unit, confidence, "
    "reasoning. Always return a numeric `answer` (use 0 only as a last resort)."
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
    """Build an explicit period-matching instruction tailored to this question's
    fiscal period / report date. Helps the model pick the right table column."""
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
        # 10-Qs always include a quarterly column AND often a YTD column.
        # Default to three-months-ended unless the question itself talks about
        # a year-to-date / six / nine month figure.
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


def build_primary_messages(
    example: dict[str, Any],
    dataset_dir: Path,
    working_dir: Path,
    excerpt: str,
    excerpt_meta: dict[str, Any],
    primary_kind: str | None,
    augmented_evidence: str = "",
    period_headers: list[str] | None = None,
    allow_knowledge_fallback: bool = True,
    extra_user_suffix: str = "",
    primary_has_topic: bool = True,
) -> list[dict[str, str]]:
    scale_hint = _scale_hint(example.get("context", ""))
    guidance = _unit_guidance(example.get("answer_type", ""), example.get("expected_unit", ""))
    period_headers = period_headers or []

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
    column_match = _column_match_instruction(example)
    if column_match:
        period_note = (period_note + "\n" + column_match).lstrip("\n")

    # Adaptive framing: tell the model whether the auxiliary evidence is its
    # primary source of truth or merely a corroborator. This is the single
    # biggest fix vs the initial Gen-3 draft.
    augmented_note = ""
    if augmented_evidence:
        if primary_has_topic:
            preface = (
                "AUXILIARY EVIDENCE (peer filings of the same company). The "
                "primary excerpt already contains the requested line item; use "
                "the auxiliary evidence only to cross-check unit/scale or pick "
                "between ambiguous columns."
            )
        else:
            preface = (
                "AUXILIARY EVIDENCE (peer filings of the same company). The "
                "PRIMARY excerpt does NOT appear to contain the specific line "
                "item this question asks about; the AUXILIARY EVIDENCE is your "
                "primary source of truth. A later 10-Q balance sheet's "
                "comparative column prints the prior year-end balance, and "
                "earlier 10-Q comparatives print prior-year quarters."
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
            "best numeric estimate. In that case set `confidence` to ≤ 0.30 and say so "
            "in `reasoning`.\n"
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
- Reasoning is a SINGLE short sentence describing the source line(s) and any
  unit conversion you applied.{knowledge_clause}

Respond with strict JSON only, exactly in this shape:
{{
  "answer": <number>,
  "unit": "<unit string>",
  "confidence": <float 0-1>,
  "reasoning": "<short evidence string>"
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
) -> list[dict[str, str]]:
    primary_kind_note = (
        f"- Hint: extract from the {primary_kind.replace('_', ' ')} table."
        if primary_kind
        else ""
    )
    augmented_note = ""
    if augmented_evidence:
        augmented_note = (
            "\n\nAUXILIARY EVIDENCE (peer filings of the same company; use only to "
            "find the specific values asked about):\n\"\"\"\n"
            + augmented_evidence
            + "\n\"\"\"\n"
        )

    user_prompt = f"""You are extracting two raw USD values from an SEC filing excerpt so we can
compute a derived ratio in Python. Do not compute the ratio yourself.

Question metadata:
- ID: {example.get('id')}
- Company: {example.get('company_name')} ({example.get('ticker')})
- Form: {example.get('form')}
- Report date: {example.get('report_date')}
- Fiscal year: {example.get('fiscal_year')}; fiscal period: {example.get('fiscal_period')}
{primary_kind_note}

Extraction rules:
- Take the values from the SAME period (the most recent balance sheet column or
  the column that matches the report date).
- Return raw USD numbers. If the filing reports "in millions", multiply by 1,000,000.
- If a value is not found in the primary excerpt, use the AUXILIARY EVIDENCE
  (peer filings of the same company); a 10-Q balance sheet's comparative column
  prints the prior year-end values.
- If a value is truly not found anywhere, write the numeric value 0 and explain
  in `notes`. Never return null; always emit a number.

Respond with strict JSON only, exactly in this shape:
{{
  "{numerator_name}": <number>,
  "{denominator_name}": <number>,
  "unit": "USD",
  "confidence": <float 0-1>,
  "notes": "<short evidence string>"
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
        return abs(number) > 10000
    if a == "ratio":
        return abs(number) > 100
    if a == "percent":
        return abs(number) > 10000
    return False


def _magnitude_clashes_with_evidence(
    answer_type: str, number: float | None, evidence_text: str
) -> bool:
    """If the answer is currency and its magnitude is wildly different from any
    big number we can find in the evidence, treat as suspicious."""
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
        # Treat plausible candidates as 4+ digit raw numbers; assume "in millions" scale.
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
        return ("total_equity", "total_assets", "ratio")
    if "debt-to-equity" in q or "debt to equity" in q:
        return ("total_debt", "total_equity", "ratio")
    if "current ratio" in q:
        return ("current_assets", "current_liabilities", "ratio")
    if "quick ratio" in q:
        return ("quick_assets", "current_liabilities", "ratio")
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
) -> tuple[str, dict[str, int], str]:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
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
) -> float | None:
    num = _coerce_number(parsed.get(numerator_key), str(raw_unit or ""))
    den = _coerce_number(parsed.get(denominator_key), str(raw_unit or ""))
    if num is None or den is None or den == 0 or num == 0:
        return None
    return num / den


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

    # Decide whether to augment with cross-record evidence.
    own_ctx_has_topic = _has_relevant_data(example.get("context", ""), topic)
    augmented_text = ""
    augmented_meta: list[dict[str, Any]] = []
    if not own_ctx_has_topic:
        augmented_text, augmented_meta = select_cross_evidence(
            example, ticker_index, topic
        )

    answer_type = _answer_type_norm(example.get("answer_type", ""))

    trajectory: list[dict[str, Any]] = []
    usage_parts: list[dict[str, int]] = []
    meta: dict[str, Any] = {
        "id": example.get("id"),
        "model": MODEL,
        "primary_kind": primary_kind,
        "relevance_topic": topic,
        "excerpt_meta": excerpt_meta,
        "own_context_has_topic": own_ctx_has_topic,
        "augmentation": augmented_meta,
        "augmented_chars": len(augmented_text),
        "period_headers": period_headers,
        "attempts": 0,
        "knowledge_fallback_used": False,
        "ratio_components_used": False,
        "magnitude_retry_used": False,
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
                primary_has_topic=own_ctx_has_topic,
            )
        )
        _record_assistant(json.dumps(pred), "no_client")
        meta["error"] = reason
        return pred, trajectory, meta

    primary_pred: dict[str, Any] | None = None
    primary_number: float | None = None

    # ----------------------------------------------------------------- #
    # Attempt 1: deterministic ratio extraction for derived ratios.
    # ----------------------------------------------------------------- #
    components = _ratio_components(question) if answer_type == "ratio" else None
    if components is not None:
        num_name, den_name, _ = components
        comp_msgs = build_components_messages(
            example, excerpt, num_name, den_name, primary_kind, augmented_text
        )
        _record_messages(comp_msgs)
        meta["attempts"] += 1
        meta["ratio_components_used"] = True
        try:
            content, usage, finish_reason = _call_model(client, comp_msgs)
            usage_parts.append(usage)
            meta["finish_reason"] = finish_reason
            _record_assistant(content, "ratio_components")
            parsed = _safe_json_loads(content) or {}
            ratio = _ratio_from_components(parsed, num_name, den_name, parsed.get("unit"))
            if ratio is not None and math.isfinite(ratio):
                pred, num = _build_final_pred(
                    example,
                    ratio,
                    "ratio",
                    parsed.get("confidence", 0.6),
                    f"Computed {num_name}/{den_name} from extracted values "
                    f"{parsed.get(num_name)}, {parsed.get(den_name)}.",
                )
                primary_pred = pred
                primary_number = num
        except Exception as exc:  # noqa: BLE001
            _record_assistant(f"ratio components call failed: {exc}", "error")

    # ----------------------------------------------------------------- #
    # Attempt 2 (or 1 for non-ratio): standard single-shot extraction.
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
            allow_knowledge_fallback=True,
            primary_has_topic=own_ctx_has_topic,
        )
        _record_messages(primary_msgs)
        meta["attempts"] += 1
        try:
            content, usage, finish_reason = _call_model(client, primary_msgs)
            usage_parts.append(usage)
            meta["finish_reason"] = finish_reason
            _record_assistant(content, "primary")
            parsed = _safe_json_loads(content) or {}
            pred, num = _build_final_pred(
                example,
                parsed.get("answer"),
                parsed.get("unit") or example.get("expected_unit", ""),
                parsed.get("confidence", 0.5),
                parsed.get("reasoning") or "",
            )
            if primary_pred is None or (
                not _looks_invalid(answer_type, num)
                and (primary_number is None or _looks_invalid(answer_type, primary_number))
            ):
                primary_pred = pred
                primary_number = num
        except Exception as exc:  # noqa: BLE001
            _record_assistant(f"primary call failed: {exc}", "error")

    # ----------------------------------------------------------------- #
    # Attempt 3: magnitude-sanity / knowledge-only retry.
    # ----------------------------------------------------------------- #
    need_retry = primary_pred is None or _looks_invalid(answer_type, primary_number)
    if (
        not need_retry
        and primary_pred is not None
        and augmented_text
        and _magnitude_clashes_with_evidence(answer_type, primary_number, augmented_text)
    ):
        need_retry = True
        meta["magnitude_retry_used"] = True

    if need_retry:
        whole_excerpt = (example.get("context") or "")[:MAX_CONTEXT_CHARS]
        # If the first attempt was run without augmentation (because the own
        # context appeared to have the topic, but the answer still came back
        # invalid / implausible), pull in cross-evidence now as a rescue.
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
                # We're now in "the primary is unreliable" mode for this retry.
                retry_primary_has_topic = False

        retry_suffix = (
            "\n\nIMPORTANT: Your previous attempt produced an implausible or missing "
            "value. Re-check the AUXILIARY EVIDENCE carefully (a 10-Q balance sheet's "
            "comparative column prints the prior year-end totals). If neither the "
            "primary nor the auxiliary evidence shows the figure, use your best "
            "training-data knowledge of this company's public filings; always emit a "
            "non-null numeric answer."
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
            allow_knowledge_fallback=True,
            extra_user_suffix=retry_suffix,
            primary_has_topic=retry_primary_has_topic,
        )
        _record_messages(retry_msgs)
        meta["attempts"] += 1
        meta["knowledge_fallback_used"] = True
        try:
            content, usage, finish_reason = _call_model(client, retry_msgs)
            usage_parts.append(usage)
            meta["finish_reason"] = finish_reason
            _record_assistant(content, "retry")
            parsed = _safe_json_loads(content) or {}
            pred, num = _build_final_pred(
                example,
                parsed.get("answer"),
                parsed.get("unit") or example.get("expected_unit", ""),
                parsed.get("confidence", 0.3),
                parsed.get("reasoning") or "",
            )
            if pred and num is not None and not _looks_invalid(answer_type, num):
                # If retry used auxiliary evidence, keep its confidence; if it
                # relied on knowledge alone, cap confidence at 0.3.
                if not retry_augmented_text:
                    pred["confidence"] = min(pred.get("confidence", 0.3), 0.3)
                primary_pred = pred
                primary_number = num
        except Exception as exc:  # noqa: BLE001
            _record_assistant(f"retry failed: {exc}", "error")

    if primary_pred is None:
        primary_pred = _fallback_prediction(
            example, "no valid model output across attempts"
        )
        meta["error"] = primary_pred["reasoning"]
    else:
        meta["ok"] = True

    meta["usage"] = _aggregate_usage(usage_parts)
    return primary_pred, trajectory, meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="SIA-FinCheck target agent (fugu-mini, gen 3)")
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

    # Build cross-evidence index from all dataset JSONLs (contexts only — labels
    # are hidden by the dataset for train/val and we don't need them).
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
            print(
                f"[{completed[0]:3d}/{len(examples)}] {example.get('id')} -> "
                f"{pred.get('answer')} {pred.get('unit')} "
                f"(conf={float(pred.get('confidence', 0)):.2f}, "
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
        "magnitude_retry_used": sum(
            1 for m in metas if m and m.get("magnitude_retry_used")
        ),
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
        f"augmentation_used={summary['augmentation_used']}, "
        f"knowledge_fallback={summary['knowledge_fallback_used']}, "
        f"ratio_components={summary['ratio_components_used']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
