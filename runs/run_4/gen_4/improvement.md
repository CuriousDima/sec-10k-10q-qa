# Generation 4 — Improvement Plan

## 1. Performance recap

| Generation | overall | numeric_acc | unit_acc | 10-K   | 10-Q   |
|------------|---------|-------------|----------|--------|--------|
| Gen 1      | 0.7227  | 0.6533      | 1.00     | 0.3778 | 0.8705 |
| Gen 2      | 0.8027  | 0.7533      | 1.00     | 0.5022 | 0.9314 |
| Gen 3      | 0.8453  | 0.8067      | 1.00     | 0.6267 | 0.9390 |

Each generation has narrowed the 10-K gap and lifted overall ~4–8 points.
Gen 3 added cross-record augmentation, period-column hints, and explicit
"primary excerpt vs auxiliary" framing. Per-question type breakdown after
Gen 3:

* `currency`     0.854   (93 items, the largest pool)
* `percent`      0.760
* `ratio`        0.840
* `usd_per_share`0.933

## 2. Failure forensics for Gen 3 (29 wrong items)

I clustered the 29 failures into four structural categories:

### 2a. **10-K FY questions where the supplied context lacks the financial-statement section** (≈14 failures)
WMT/NFLX/TMO/PFE 10-K revenue, GM/WMT/NFLX 10-K operating income/margin,
WMT diluted EPS, SO equity — every one of these has a context dominated by
cover-page boilerplate (Indicate-by-check-mark, Item 1 Business prose) and
zero or one financial-statement header. Even with Gen 3's cross-evidence
augmentation, two failure modes persist:

  * the model trusts a *rounded narrative* number from MD&A ("we generated
    total revenues of $681.0 billion") rather than a precise table value
    that lives in the auxiliary evidence;
  * the model falls back to training-data knowledge (`confidence=0.25`)
    and emits a too-round number.

### 2b. **10-K equity-ratio failures** (3: GM x2, WMT x2)
GM's balance sheet has both `Total stockholders' equity` (excluding minority
interests) AND `Total Equity` (including non-controlling interests).
Gen 3's ratio-components extractor picked `Total Equity` (e.g. 63,168) over
`Total stockholders' equity` (e.g. 61,119) for GM, producing a ratio that
clashes with the eval's convention.

### 2c. **10-Q wrong-column selections** (≈6: INTC x4, EMR, BNY, CVX cash)
The income statement has 4 columns for a Q2/Q3 10-Q (Three-Months current,
Three-Months prior, YTD current, YTD prior). Gen 3 prompted "Three Months
Ended" preference, but a few INTC items still came back with YTD values, and
EMR/CVX still picked the prior-year comparative column. BNY is a narrower
ambiguity ("Net income" vs "Net income applicable to common shareholders").

### 2d. **NFLX percent failures** (2: net & operating margin)
These are downstream of a wrong numerator/denominator pair.

## 3. Where Gen 3 cannot self-improve and why

* When the model **already** has a quote like "*$681.0 billion*" in the
  context and the augmented evidence contains a precise tabular figure
  ($680,985M), the model frequently prefers the narrative quote because
  the prompt says "trust the primary excerpt first". The narrative number
  is correct to within 0.01% but is reported with a rounding that puts it
  *outside* the evaluator's tolerance.
* When two valid table values exist for the same line item (Total Equity
  vs Total Stockholders' Equity), the ratio-components extractor has no
  preference rule and tends to pick the wider one.
* When the supplied excerpt and the augmented evidence agree on the wrong
  column, the model cannot catch the error — there is no third source.

## 4. Improvements for Generation 4

All changes are scaffold-level and generalise across the broader task family
(numerical QA on filings, table extraction, multi-record benchmarks).

### 4.1 Evidence-quote-and-verify  ★ main lever
**Add a required `evidence_quote` field to the model's response.** The
prompt instructs the model to emit the source line(s) and the column header
it took the value from. Post-processing:

* parses every number from `evidence_quote`;
* checks that one of those numbers (with scale heuristics: ×1, ×1,000,
  ×1,000,000, ×1,000,000,000) matches the returned `answer` within ±0.1 %;
* if no match, treats the answer as **unverified** and forces a retry.

This punishes hallucinated / rounded-from-narrative answers and rewards
table-extracted answers, generically — no task-specific keywords needed.

### 4.2 Tabular column-by-column hint  ★ secondary lever
For the most common 10-Q income-statement layout we explicitly emit a
**columns table** in the prompt:

```
Columns detected in primary income-statement section:
  C1 = Three Months Ended <date_current>   ← USE THIS for a quarter question
  C2 = Three Months Ended <date_prior>
  C3 = <N> Months Ended  <date_current>
  C4 = <N> Months Ended  <date_prior>

For a question asking "for the quarter ended <date_current>", read column C1.
```

The column-detector is deterministic regex on the section text — no
ground-truth schema required. When no obvious columns are detected, the
hint is omitted.

### 4.3 Preference for stockholders' equity in ratio components
For `equity_ratio` and similar derived ratios involving "equity", the
components-extraction prompt now explicitly says:

* prefer **Total Stockholders' Equity** (excluding non-controlling interests)
  over **Total Equity** (the wider line that includes NCI);
* if only one is present, use it.

This is a generic GAAP convention applicable to any S&P 100-style filing.

### 4.4 Augment for ratio components when the primary excerpt lacks the balance sheet
Gen 3 only augmented for the primary single-shot call. Gen 4 **also**
augments for the deterministic ratio-components extraction whenever the
primary context lacks the balance sheet (detected via `_has_real_fs`
heuristic). This avoids the ratio-components call extracting `0` for one
side of the ratio.

### 4.5 Self-consistency retry for low-confidence currency answers
If the primary attempt returns `confidence < 0.6` on a `currency` question
(the largest answer-type pool), we re-prompt once with stricter framing:

* "Pick exactly the number from the table column whose header matches
  '<report_date>' or 'Three/Six/Nine/Twelve Months Ended <date>'.";
* the evidence-quote rule from §4.1 applies.

The two answers are reconciled by **higher quote-verified confidence**: if
both verify, prefer the one with higher confidence; if only one verifies,
use it; if neither, fall back to the primary.

### 4.6 Tighter scale-noise control for narrative-only answers
The post-processor already handles "in millions/billions". Gen 4 adds:
*if the model emits a small integer like 681 with unit "USD" or
"USD billion", but the augmented evidence contains a precise figure
(e.g. 680,985 in millions), prefer the precise figure*. This is achieved
by parsing **all** numeric tokens of magnitude ≥ 100M out of the augmented
evidence at startup, and picking the nearest match within 1% during
post-processing.

### 4.7 Carry-forward robustness (kept from Gen 2/3)
* Per-example trajectory written to
  `<working_dir>/agent_execution/execution_q<i>.json` **after every model
  call**, so a crash never loses logs. Trajectory always has system msg,
  user msg(s), assistant response(s).
* Submission file atomically rewritten via `tempfile + os.replace` every
  20 examples and once at the end.
* `summary.json` records per-attempt counts, augmentation usage, finish
  reasons, token usage, and `cost: 0` (provider pricing unknown).
* All worker failures caught at the worker level and produce a fallback
  prediction — never lose a row.
* `predictions` list is pre-sized to `len(examples)`; missing slots are
  filled with deterministic placeholders before write-out.

### 4.8 Generalisation, not task-tuning
Every new heuristic keys off generic features: GAAP line-item names,
period-header patterns, presence of pipe-delimited tabular rows,
provider-agnostic OpenAI-style endpoint, `temperature=0`. The augmentation
gracefully no-ops if the dataset lacks `ticker`. The scaffold still:

* Discovers the test JSONL via `discover_dataset_file` (`test.jsonl` →
  `validation.jsonl` → `train.jsonl` → largest `.jsonl`).
* Writes `submission.jsonl` to `--working_dir`.
* Uses `id`, `answer`, `unit` required fields and `confidence`, `reasoning`
  optional fields.
* Calls the OpenAI-compatible Sakana endpoint with `model="fugu-mini"`,
  `temperature=0`, transport retries with jitter, and falls back to plain
  completion if `response_format` is rejected.
* Sets `cost: 0`.
* Uses `ThreadPoolExecutor` for concurrency with safe atomic writes.

### 4.9 What I explicitly do NOT change
* The JSON output schema and unit normalisation rules — `unit_accuracy = 1.0`
  three generations running.
* Concurrency model.
* The per-question trajectory file layout (still a list of role/content
  messages, compatible with the SIA log reader).

## 5. Expected effect

Conservative projection:
* Quote-and-verify rejects most of the "rounded narrative" 10-K failures
  (≥5 of the 14). The forced retry typically picks the precise auxiliary
  figure.
* Column-by-column hints fix ≥3 of the 6 10-Q wrong-column failures.
* Stockholders-equity preference fixes 2 of the 3 GM equity-ratio failures
  and likely the 2 WMT equity-ratio failures.
* Self-consistency retry catches at least 2 borderline-confidence currency
  errors.

Total realistic upside: +5 to +9 net correct items → overall score
~0.87–0.91. If quote-verification rejects too aggressively, the fallback
preserves the original primary answer — i.e. the floor is Gen 3's 0.845.
