# Generation 2 — Improvement Plan

## 1. Summary of Generation 1

| Metric | Gen 1 |
|---|---|
| overall_score | **0.664** |
| numeric_accuracy | 0.58 |
| unit / format / sanity | 1.00 / 1.00 / 1.00 |
| 10-Q score | **0.863** |
| 10-K score | **0.200** |
| LLM contribution | **0 / 150 calls succeeded** |
| Source counts | `heuristic = 100`, `fallback_placeholder = 50` |

The agent never benefited from the LLM at all – every one of the 150
`llm.chat(...)` calls raised an exception that the harness silently logged as
`"LLM call failed on …: "` with an empty error message.  The whole score came
from the regex / table-row heuristic.  10-Q rows scored 0.86 (the table
extractor mostly works), but 10-K rows scored 0.20 because the supplied
contexts only include the cover page + the first ~18 kB of business
description – the financial statements are never in the excerpt.

## 2. Root-cause analysis

### 2.1 Why every LLM call failed silently

`HuggingFace.apply_chat_template(..., return_tensors="pt")` returns a
`transformers.tokenization_utils_base.BatchEncoding`, **not** a `dict`.
The gen-1 code is:

```python
if isinstance(templated, dict):
    input_ids = templated.get("input_ids")
else:
    input_ids = templated          # <- BatchEncoding goes here
```

`BatchEncoding` *behaves* like a dict but `isinstance(_, dict)` is `False`, so
`input_ids` becomes the whole `BatchEncoding`.  `BatchEncoding.to(device)`
happens to work, but the next line –

```python
new_tokens = output_ids[0, input_ids.shape[1]:]
```

– blows up with `AttributeError: shape` (raised inside `__getattr__` with no
message string).  The outer except clause prints `f"... failed on {id}: {exc}"`
which yields an *empty* error.  All 150 trajectories show the same pattern:
`"text": "(no response)"`.

### 2.2 Other extraction issues that hurt the heuristic

* "stockholders' equity" questions are answered by the **noncontrolling
  interest-inclusive** *Total Equity* row, not the parent attributable row
  (CVX, IBM, EMR Q1/Q3 cases).
* 10-K contexts in this dataset are truncated to ~18 kB and contain only the
  cover page + business overview, not the consolidated statements.  Most 10-K
  cover pages still include a single narrative sentence such as *"During fiscal
  2026, we generated total revenues of $713.2 billion, which primarily
  comprised net sales of $706.4 billion."* – we never tried hard enough to
  scrape these.
* The 4-column (quarter + YTD) layout selection is sometimes wrong.  When the
  question's `fiscal_period` is `Q3`, we want the YTD column (9 months ended)
  for revenue / income but the latest column for balance-sheet items.

### 2.3 Wasted-budget gen-2 seed

The auto-generated gen-2 seed switched to a totally different architecture
(`accelerate launch ... LoRA training`) that was never validated against this
dataset and would likely hang or produce worse predictions.  We are
deliberately discarding it and continuing the proven heuristic + Gemma route.

## 3. Goals for Generation 2

1. **Fix the LLM path so it actually contributes** – this alone should add a
   meaningful chunk of score because the LLM can solve cases where the
   heuristic returns the fallback placeholder.
2. **Make extraction layout-aware** – pick the *parent-attributable* equity
   row, prefer YTD column when appropriate, and handle Gemma's BatchEncoding
   output correctly.
3. **Expand 10-K prose patterns** so we recover at least the company-level
   narrative numbers ("revenues of $713.2 billion") on cover pages.
4. **Robust error reporting** – any exception must be captured with
   `repr()`/`traceback.format_exc()` so future generations can diagnose
   failures from the log rather than from silence.
5. **Keep the agent generic and resumable** – periodic flushes, per-question
   trajectories, never blow up on a single bad example, never reach outside
   `--dataset_dir` or `--working_dir`.

## 4. Structural changes

### 4.1 LLM driver rewrite

* Replace `AutoModelFor*` cascade with a single direct import of
  `Gemma4ForConditionalGeneration` (the actual architecture in the
  checkpoint).  Tested locally – it generates correctly end-to-end.
* Use the parent (multimodal) module for `.generate` – never reach into
  `.language_model`.  Generation works as a plain causal LM because we never
  pass `pixel_values` / `input_features`.
* Accept any mapping-like object returned by `apply_chat_template`
  (`BatchEncoding` *or* `dict`) via `isinstance(_, (dict, BatchEncoding))`,
  and additionally via the duck-typed `"input_ids" in templated`.
* Wrap every per-question call in
  ```python
  try: ...
  except Exception as exc:
      err = repr(exc) or type(exc).__name__
      log.write(f"[{id}] {err}\n{traceback.format_exc()}\n")
  ```
* Always also log the prompt length and elapsed seconds to detect OOM /
  truncation problems.

### 4.2 Heuristic improvements (still 100 % deterministic, no labels touched)

* New intent group `stockholders_equity_attributable` that prefers rows
  matching `Total\s+[A-Z][a-z]+(?:\s+\w+)*\s+Stockholders'?\s+Equity` or
  `…Stockholders'?\s+Equity\s+attributable\s+to.*`.
* Order of preference for the *equity* intent:
  1. Parent-attributable row (e.g. *"Total Chevron Corporation Stockholders'
     Equity"*, *"Total IBM stockholders' equity"*, *"Total stockholders'
     equity attributable to Apple Inc."*).
  2. Plain *"Total stockholders' equity"*.
  3. Generic *"Total equity"* (last resort because it includes
     noncontrolling interests).
* Period-aware column picking for currency/percent intents:
  * When `period_hint == "quarter"` and the row has 4 cells, prefer the
    *quarter* pair using header words `three|thirteen|quarter`.
  * When `period_hint == "year"` (10-K / FY), prefer the year pair using
    `nine|six|twelve|year|fiscal`.
  * When the question explicitly contains *"for the quarter ended YYYY-MM-DD"*
    and the table has both quarter and YTD columns, pick the quarter column
    for the matching year.
* Generalised prose patterns capturing
  * `(total\s+)?(net\s+)?revenues?\s+(of|were|was|reached|totaling|totaled|amounted to)\s+(approximately\s+)?\$?\s*([\d,.]+\s*(billion|million|thousand|bn|mm))`
  * `total\s+assets\s+(of|were|was)\s+\$?\s*([\d,.]+\s*(billion|million|thousand|bn|mm))`
  * `net\s+(income|earnings)\s+(of|was|were)\s+\$?\s*([\d,.]+\s*(billion|million|thousand|bn|mm))`
  * `(diluted|basic)\s+(earnings\s+per\s+(common\s+)?share|eps)\s+(of|was|were)\s+\$?\s*([\d.]+)`
* A new fallback that looks for big standalone scaled numbers on the same
  line as the intent keyword, e.g. *"We had total assets of $52.2 billion
  as of June 30, 2024"*.

### 4.3 Resilience / logging

* Always write a placeholder submission line for every test row **before** any
  LLM work, so even a crash leaves a valid file.
* Flush after every 5 predictions (was 10).
* Per-trajectory JSON now includes `error_repr`, `error_traceback`,
  `prompt_chars`, `response_chars`, `elapsed_seconds` plus the existing
  conversation triples.
* A consolidated `agent.log` is also written to the working directory.

### 4.4 Generalisation – not over-fitting

Everything new is generic: prose extraction, equity disambiguation, period
selection are common in any financial-table QA dataset.  No ticker-specific
lookup tables, no company-level constants, no per-example overrides.
The agent still answers any 10-K / 10-Q numerical QA item that follows the
SIA-FinCheck schema (`id`, `context`, `answer_type`, `expected_unit`,
`question`).

## 5. Compatibility with the harness

* Only writes to `--working_dir`; never touches `--dataset_dir`.
* Submission path is `<working_dir>/submission.jsonl`, prediction order
  preserved.
* Per-question trajectories at `<working_dir>/agent_execution/execution_q*.json`
  (the SIA-required layout).
* A `summary.json` with source counts, LLM availability, and run timing is
  written at the end.
* No network calls; only the local checkpoint at
  `LOCAL_GEMMA_MODEL_PATH` (defaults to `/workspace/gemma_checkpoints/gemma-4-31B-it`).
* Setting `SIA_FINCHECK_USE_LLM=0` falls back to heuristic-only mode (useful
  for fast smoke tests).

## 6. Expected outcome

* LLM path now functional → expect at least double-digit improvement on the
  50 fallback-placeholder cases.
* Equity disambiguation fixes ~5 misclassifications (CVX/IBM/EMR equity).
* Better prose coverage gives a fighting chance on ~30 10-K cases that were
  pure placeholders before.

## 7. Things we explicitly *did not* change

* No new third-party packages beyond what gen 1 already installed.
* No external network access.
* No LoRA / fine-tuning – tried in the auto-seed but discarded because there is
  no signal that it would help in the available time budget and it would risk
  blowing past the wall-clock limit before producing a `submission.jsonl`.
