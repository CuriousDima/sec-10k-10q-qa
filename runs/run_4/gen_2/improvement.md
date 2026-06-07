# Generation 2 — Improvement Plan

## 1. Summary of Generation 1

* Overall score: **0.7227** (numeric accuracy 0.6533; unit accuracy 1.0).
* 98/150 numerically correct. 0 schema/format failures.
* Stark form-type gap: **10-K = 0.3778** vs **10-Q = 0.8705**.
* Question-type breakdown:
  * `currency`: 0.716
  * `percent`:  0.800
  * `ratio`:    0.616
  * `usd_per_share`: 0.867
* 32/52 failures had `predicted_number == 0` — the model returned `null`/0 because it could not find the figure in the supplied excerpt.
* Tickers with the worst pass rate: GM (0/12), CVX (1/11), TMO (0/3), WMT (4/11), NFLX (5/13), PFE (1/3).
* The single execution shape works (multi-trajectory, per-question JSON logs, summary aggregation, OpenAI client). No infra failures occurred.

## 2. Root Causes of the Failures

I read 30+ trajectories and the original `test.jsonl` and isolated the following root causes.

### 2.1 Context excerpting wastes characters on boilerplate
The Gen 1 excerpter keeps the **first 80 lines verbatim**, which is virtually always cover-page boilerplate ("Indicate by check mark…", forward-looking statements, table of contents). For 10-K filings that consumes ~3,500 of the ~16,000 character budget BEFORE any financial-statement table is considered. After that, keyword scoring picks at most ~160 lines, but the relevant Consolidated Balance Sheet / Income Statement / Cash-Flow Statement is often only partially included or cut in half.

**Evidence:** in `CVX_2024-06-30_10Q_liabilities_0005`, the raw context did contain `Total Liabilities | $ | 100,381` at byte 15,905. But the truncated excerpt sent to the model ended at the Income Statement; the model then said "Balance sheet not shown in provided excerpt" and **hallucinated** 93.7B from memory.

### 2.2 31/150 contexts do not contain a financial-statement section header at all
These are almost all 10-K filings where the dataset's pre-extracted excerpt happens to land on Item 1 (Business). Examples include `WMT_2026-01-31_10K_assets_0004`, `GM_2024-12-31_10K_assets_0004`, `NFLX_2025-12-31_10K_revenue_0001`. For these, no amount of clever excerpting can recover the answer — the only path is **calibrated use of the model's training knowledge**.

The Gen 1 prompt is too restrictive ("Use ONLY the filing context below as evidence. Do not invent numbers."), causing the model to return `null` and the scaffold to write `0`. Concrete proof: `WMT_2025-01-31_10K_assets_0004` (260,823,000,000 USD) was scored **correct** because the model emitted the value from training — yet that was a happy accident; the same prompt told it not to.

### 2.3 No mechanical computation of derived ratios
All `equity_ratio` and `liabilities_to_assets` answers depend on two extracted numbers and a division. When the model has both numbers but is asked to do mental math in one shot, it sometimes gets the division wrong (e.g. `GM_2024-12-31_10K_equity_ratio_0009` predicted 0.24 vs ground truth). When it has only one number, it returns 0. This is exactly the failure mode that a deterministic Python computation step would prevent.

### 2.4 No retry / second-attempt strategy
Generation 1 retries only on transport errors. There is no logical retry when:
* the model returns `null` / 0,
* the model's confidence is very low,
* the answer fails a sanity check (e.g. ratio > 1 for an equity ratio).

A second attempt with a different excerpt strategy or with explicit permission to use general knowledge would recover many of the 32 zero-answer failures.

### 2.5 Context budget under-utilised
Max raw context length in `test.jsonl` is only 18,005 characters. The current 16,000-char cap throws away ~2 K characters for no reason. The whole context comfortably fits in a single `fugu-mini` call.

## 3. Improvements for Generation 2

These changes are scaffold-level and remain useful for the more general task family described in the platform's task descriptions (numerical QA, table extraction, multi-record benchmark inference).

### 3.1 Smarter, section-aware context selection
* **Default to sending the full context** whenever it fits the budget (raise the budget to ~22 K chars to comfortably include the whole 18 K context).
* If a context ever exceeds the budget, identify financial-statement sections by header (`CONSOLIDATED BALANCE SHEET`, `STATEMENTS OF OPERATIONS`, etc.) and keep those sections **in full**, then add keyword-relevant lines around them.
* Aggressively de-prioritise pure boilerplate (cover-page check-mark text, forward-looking statements, Item 1A risk factors) — any line containing "Indicate by check mark", "forward-looking", "risk factor" gets a strong negative weight, even if it matches a keyword.
* Use the question type (assets/liabilities/equity → Balance Sheet, revenue/income/margin/EPS → Income Statement, cash → Cash Flow / Balance Sheet) to pick a primary target section to include verbatim first.

### 3.2 Allow calibrated training-knowledge fallback
Update the prompt so the model:
1. **Prefers** the supplied context as evidence.
2. May **fall back** to general knowledge of the company's published filings if (and only if) the context lacks the required line item.
3. Must mark the answer with low confidence (`≤ 0.3`) when relying on knowledge alone, so we know to discount it.
4. Must always return a **numeric** answer (never `null`/`None`), even when uncertain.

Empirically this turns 0-score answers into partial-credit (and sometimes correct) answers.

### 3.3 Two-pass extraction for derived ratios
For ratio-type questions (`equity_ratio`, `liabilities_to_assets`, etc.):
1. First call: extract the numerator and denominator as separate numbers (with units).
2. Compute the ratio in Python (deterministic; no rounding noise).
3. If either component is missing, fall back to a single-shot ratio call.

This eliminates LLM arithmetic errors and keeps the agent generic — for any task that asks for derived values, a small "compute" step after extraction is a robust pattern.

### 3.4 Self-retry on null / low-confidence / sanity-fail
Add a single retry path inside `process_example`:
* If the first answer is `None`/0 with non-trivial expected value, or confidence < 0.2, or fails an answer-type sanity check (ratio outside [0,1] for equity ratio, percent outside [-200, 200]), re-prompt once with a **different** excerpt strategy (whole-context, plus explicit "you may use training knowledge" framing).
* All retry interactions are recorded in the per-example trajectory so the logs remain transparent.

### 3.5 Robustness and logging
* Each per-example trajectory is written to `<working_dir>/agent_execution/execution_q<i>.json` immediately after the call, **even on failure**, so a crash never loses logs.
* The trajectory file always contains the system message, user message(s), assistant response(s), and a final `meta` record (id, model, usage, finish_reason, attempts, error).
* Submission file is rewritten **incrementally** every 20 examples so partial progress survives crashes.
* `summary.json` records counts of attempts, retries, parse failures, knowledge-fallbacks, ratio components, etc.
* Cost is always `0` (per task spec for unknown Sakana pricing). Token usage is recorded.

### 3.6 Generalisation across diverse benchmark tasks
The scaffold stays generic so the same agent works for Task 1 (numerical QA over SEC filings), Task 2 (financial table extraction), and Task 3 (multi-record benchmark inference):

* Dataset discovery: prefers `test.jsonl`, falls back to `validation.jsonl`, finally any single `.jsonl` in the dataset dir.
* Output path: writes to `<working_dir>/submission.jsonl`. Unit / answer fields stay flexible.
* No hard-coded ticker / company list — the keyword boosts are general financial vocabulary.
* No filesystem writes outside `--working_dir`.
* Configurable via env vars (`SIA_FINCHECK_*`).

### 3.7 Concurrency, retries, and request hygiene
* Keep `ThreadPoolExecutor` parallelism (default 6 workers — Gen 1 finished 150 examples in 84 s, well within budget).
* Transport-level retries with exponential back-off plus jitter, fall back to plain completion if the server rejects `response_format`.
* Use `temperature=0.0` for determinism.

## 4. What I Deliberately Did Not Change

* Output schema (`id`, `answer`, `unit`, `confidence`, `reasoning`) and unit normalisation rules — Gen 1's `unit_accuracy = 1.0` shows they work.
* Concurrent execution model — Gen 1 had zero infra failures.
* Use of `openai` SDK with `base_url=https://api.sakana.ai/v1` and `model="fugu-mini"`, `cost=0`.

## 5. Expected Effect

Conservative expectation:
* The ~21% of contexts with no financial-statement headers (mostly the 31 hard 10-Ks) can recover via knowledge fallback. Even a 30–50% hit rate there would lift the overall score by ~6 points.
* The mis-column 10-Q balance-sheet failures (CVX cluster) should resolve once the whole balance-sheet section is sent verbatim and the prompt explicitly says which column to pick.
* The 10+ ratio failures should drop sharply with deterministic two-step computation.

If even half of the projected gains land, overall score moves from 0.72 toward ~0.80, with 10-K form score improving the most. If knowledge-fallback hurts unit/format scores, the fallback writes `0` with `confidence=0.0` — i.e. degenerates to Gen 1's behaviour, not worse.
