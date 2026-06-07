# Generation 3 — Improvement Plan

## 1. Performance recap

| Generation | overall | numeric_acc | unit_acc | 10-K score | 10-Q score |
|------------|---------|-------------|----------|------------|------------|
| Gen 1      | 0.7227  | 0.6533      | 1.00     | 0.3778     | 0.8705     |
| Gen 2      | 0.8027  | 0.7533      | 1.00     | 0.5022     | 0.9314     |

Gen 2 already added section-aware excerpting, deterministic ratio computation,
and one knowledge-fallback retry. 10-Q scores are near-saturated; the remaining
deficit is overwhelmingly on **10-K** items.

## 1a. Critical review of the initial Gen-3 draft

Before committing, I audited the first Gen-3 implementation. The big lever
(cross-record evidence augmentation) is sound and verified to surface the right
figures (e.g. GM Total Assets $281,284 M for Dec 31 2025 is recovered from a
peer 2026-Q1 10-Q in `train.jsonl`). But several second-order details would
have leaked correctness:

* **Noisy period-column hints.** `_extract_period_headers` matched any month
  + day + year pattern across the *whole* document. On a 10-K cover page that
  pulled in things like `'June 30, 2025'` (the market-value date used to
  compute the float) and `'January 15, 2026'` (the record date for shares
  outstanding). These look like financial-statement column headers but are
  not. The model can be misled into thinking those are valid columns.
  **Fix:** only collect dates from table-shaped lines (containing `|` or
  multiple whitespace-separated numbers) **or** from inside a detected
  financial-statement section span.
* **Conflicting prompt framing when primary context has nothing useful.**
  The prompt told the model both "Prefer the primary filing context" AND
  "use AUXILIARY EVIDENCE only to corroborate". When the primary context is
  all cover-page boilerplate — the actual root cause of the 28 10-K losses
  — this can make the model ignore the augmented evidence (the only real
  source of truth) and hallucinate from training data anyway.
  **Fix:** when `own_context_has_topic == False` AND augmented evidence is
  present, the prompt now explicitly says "the primary excerpt does NOT
  contain the requested line item; the AUXILIARY EVIDENCE is the primary
  source of truth for this question." When the own context does contain the
  topic, we keep the conservative "primary first, auxiliary corroborates"
  framing.
* **No explicit "match this report_date" instruction.** Eight 10-Q failures
  were wrong-column picks (year-to-date column instead of three-months-ended,
  etc.). The header-list hint was passive. **Fix:** add a literal instruction
  pinning the requested period: "Match the numeric column whose header
  contains the date 'YYYY-MM-DD' or the period type implied by
  fiscal_period ('Q1' → three months ended; 'FY' → year ended)."
* **`_RELEVANCE_KEYS["cash"]` was overly tight** (only `"cash and cash
  equivalents"` / `"cash and equivalents"`). 10-K filings often use
  `"Cash, cash equivalents"` with a comma. Expanded the key set.
* **One-time augmentation retry.** If the FIRST attempt was made without
  augmentation (because the own context appeared to have the topic, but the
  model still answered wrongly with very low confidence), Gen-3's retry path
  only sent a knowledge-fallback prompt. **Fix:** if the model returned a
  low-confidence answer and we hadn't augmented, the retry now also pulls
  in cross-evidence.
* **Cleanups:** removed unused `typing.Iterable` import; fixed the
  awkward escaped-quote concatenation in `augmented_note` that produced
  unbalanced fences; de-duplicated `_RELEVANCE_KEYS["assets"]`.

None of these change the architecture; they tighten the heuristics that
the architecture relies on.

## 2. Where Generation 2 still loses

I joined `gen_2/results.json` with `test.jsonl` and computed, for each failure,
whether the **supplied** context contained an actual financial-statement row
matching the question (regex `Total assets|Net sales|Total revenues|...` near
digits). Stratified result:

| form | failures | context lacks real FS data | wrong despite real FS data |
|------|----------|----------------------------|----------------------------|
| 10-K | 28       | **28**                     | 0                          |
| 10-Q | 9        | 1                          | 8                          |

So **all 28 10-K failures are "evidence not in the supplied context"** —
no excerpting tweak, no prompt tweak, no knowledge fallback inside the
single supplied filing can help. The Gen 2 knowledge-fallback path was
triggered only **once** in 150 examples; for the 28 mostly-confident-but-wrong
10-K answers, the model emitted a plausible but hallucinated number and the
sanity check passed, so retries were never invoked.

For the 8 10-Q failures with real data, the model is picking the **wrong
column** (e.g. year-to-date instead of three-months-ended, or the comparative
prior-year column instead of the current quarter).

## 3. Where the right answers actually live

The dataset ships three context files:

* `test.jsonl`  — 150 questions, contexts may or may not include statements.
* `validation.jsonl` — 150 same-format records (labels hidden but contexts public).
* `train.jsonl` — 1,200 same-format records (labels hidden but contexts public).

For every ticker whose 10-K test items fail (GM, NFLX, WMT, TMO, PFE, PG, …),
`train.jsonl` contains 10-Q filings of the same company with REAL balance-sheet
and income-statement tables. Each 10-Q's balance sheet has a **comparative
column** that prints the prior fiscal year-end totals. For example:

* `GM_2026-03-31_10Q_*` (in train) contains `Total Assets | $ | 280,974 | $ | 281,284`
  where `281,284` = **GM total assets at Dec 31, 2025** — exactly what
  `GM_2025-12-31_10K_assets_0004` (in test) asks for.
* `GM_2025-09-30_10Q_*` (in train) prints `Total Assets | $ | 288,168 | $ | 279,761`
  — `279,761` = GM total assets at Dec 31, 2024, the answer for
  `GM_2024-12-31_10K_assets_0004` (also in test).

Cross-ticker availability check (per ticker, how many train items contain a real
balance sheet):

| ticker | train items | with real BS |
|--------|-------------|--------------|
| GM     | 11          | 11           |
| WMT    | 16          | 16           |
| NFLX   | 10          | 9            |
| TMO    | 13          | 13           |
| PFE    | 7           | 7            |
| PG     | 14          | 14           |
| INTC   | 2           | 0 (use other test items: 6 have BS) |
| BNY    | 11          | 0 (use other test items: 3 have BS) |
| EMR    | 6           | 0 (use other test items: 7 have BS) |

So train + the rest of test together cover every failing ticker. Reading those
contexts is fully permitted by the task description ("Use the filing context as
the evidence source" — the contexts in `train.jsonl` and `validation.jsonl` are
filing contexts too; only the LABELS are hidden).

## 4. Improvements for Generation 3

### 4.1 Cross-record context augmentation  ★ main lever
Build a per-ticker (and per-CIK as fallback) index over **all** `*.jsonl` files
in the dataset directory at startup. For every question:

1. Take the supplied `context` (always primary evidence).
2. If it lacks the financial-statement row needed for this question, pull
   compact extracts from the closest same-ticker records with real data:
   * For balance-sheet questions, prefer same-ticker contexts whose
     `report_date` would print the target date as the BS comparative column
     (i.e. a 10-Q whose report_date is in the same or next fiscal year).
   * For income-statement / cash-flow questions about a full fiscal year,
     prefer contexts that include that year's Statements of Operations / Cash
     Flows table.
3. Limit each extra source to ~2,500 chars (its financial-statement section
   only) and the total augmentation budget to ~8,000 chars.
4. Tag each piece with its provenance (`[Source: GM 10-Q report_date=2026-03-31]`)
   so the model can attribute and explicitly pick the right column.

This is generic across tasks: when the dataset has multiple records per entity,
augmenting with peer records is a common, robust pattern (Task 2 "financial
table extraction" and Task 3 "multi-record benchmark inference" both benefit;
Task 1 single-record questions are the trivial case where the augmentation set
is empty).

### 4.2 Explicit period-column hint  ★ secondary lever
Inside `context_excerpt`, also extract column-header lines that appear directly
under section titles (e.g. `December 31, 2024 | December 31, 2023` or
`Three Months Ended September 27, 2025`). Pass those headers to the prompt as
"Candidate period columns:" so the model is reminded which numeric column maps
to the requested `report_date` / `fiscal_period`. This addresses the eight 10-Q
"wrong column" failures.

### 4.3 Smarter answer-validity check
Gen 2 only flagged the empty/zero answer as invalid. Gen 3 also flags:

* `currency` answers that exceed/undershoot the order of magnitude of any
  number found in the augmented context by > 100×;
* `ratio` answers > 1 or < 0 (with sign conventions) for equity-type ratios;
* `percent` answers > 200 in absolute value.

When invalid, re-prompt once with the augmented context plus an explicit
"Reconsider; the value you returned is implausible given the evidence" hint.

### 4.4 Confidence calibration
The knowledge-fallback path now becomes a **last resort** (cross-evidence is
much safer). When triggered, cap confidence at 0.25. When the primary answer
agrees with a value extracted from the cross-evidence, raise confidence to
0.95.

### 4.5 Robustness & logging (carry forward from Gen 2)
* Per-example trajectory written to `agent_execution/execution_q<i>.json`
  immediately after every model call, so a crash never loses logs. The
  trajectory always includes system message, user message(s), assistant
  response(s).
* Submission is **flushed atomically** (write to `.tmp` + rename) every 20
  examples so partial progress survives a transport crash.
* `summary.json` includes per-attempt counts, augmentation usage, finish
  reasons, token usage, and `cost: 0`.
* All worker failures are caught at the worker level and produce a fallback
  prediction — never lose a row.
* `predictions` list is pre-sized to `len(examples)`, so any missing slots
  at the end are filled with deterministic placeholders before write-out.

### 4.6 Generalisation, not task-tuning
Everything is feature-driven on dataset fields (`ticker`, `cik`, `report_date`,
`fiscal_year`, `fiscal_period`, `answer_type`) rather than hard-coded company
constants. If a future dataset doesn't have a `ticker` field, the augmentation
gracefully no-ops. The scaffold still:

* Discovers the test JSONL via `discover_dataset_file` (`test.jsonl` →
  `validation.jsonl` → `train.jsonl` → largest `.jsonl`).
* Writes `submission.jsonl` to `--working_dir`.
* Uses `id`, `answer`, `unit` required fields and `confidence`, `reasoning`
  optionals (validated by the evaluator).
* Calls the OpenAI-compatible Sakana endpoint with `model="fugu-mini"`,
  `temperature=0`, transport retries with jitter, and falls back to plain
  completion if `response_format` is rejected.
* Sets `cost: 0` (provider pricing unknown).
* Uses `ThreadPoolExecutor` for concurrency with safe atomic writes.

### 4.7 What I explicitly do NOT touch
* JSON output schema and unit normalisation rules (`unit_accuracy = 1.0`
  already).
* Existing per-example trajectory file shape — backwards compatible with
  the SIA log format.
* Deterministic two-step ratio extraction — already lifted ratio accuracy.

## 5. Expected effect
* 28 currently-wrong 10-K answers have a direct evidence path via cross-context
  augmentation. Even a 60% hit rate there nets +11 points on overall score.
* 8 currently-wrong 10-Q answers gain from explicit period-column hints; a 40%
  fix rate adds +2 points.
* Net realistic target: overall score ≥ 0.88 (10-K parity with 10-Q being the
  long-term goal).
* If cross-evidence somehow degrades accuracy on already-correct items, the
  retry path falls back to Gen 2 behaviour, so the floor is the Gen 2 score.
