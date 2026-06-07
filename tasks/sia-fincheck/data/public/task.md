# SIA-FinCheck Task

Answer numerical questions about recent SEC 10-K and 10-Q filing excerpts for current S&P 100 companies.

## Files available to the target agent

The target agent receives `--dataset_dir` pointing at this `data/public/` directory. It may read:

- `train.jsonl` — 1,200 public training examples, labels hidden.
- `validation.jsonl` — 150 public validation examples, labels hidden.
- `test.jsonl` — 150 public test examples to answer for SIA evaluation.
- `sample_submission.jsonl` — example JSONL output format.
- `metadata/` — public dataset metadata.

Private labels are not available to the target agent. They are used only by `evaluate.py`.

## Public example schema

Each public JSONL record includes filing metadata, a question, an expected answer unit, and filing-derived context. Private labels are intentionally omitted.

Important fields include:

- `id`
- `ticker`
- `company_name`
- `form` (`10-K` or `10-Q`)
- `filing_date`
- `report_date`
- `fiscal_year`
- `fiscal_period`
- `question`
- `answer_type` (`currency`, `ratio`, `percent`, or `USD/share`)
- `expected_unit`
- `context`

## Required output

For SIA evaluation, answer every row in `test.jsonl` and write predictions to:

```text
<working_dir>/submission.jsonl
```

The target agent receives `--working_dir` as a writable output directory.

Write one JSON object per line, in the same order as `test.jsonl` if possible:

```json
{"id":"AAPL_2024_10K_revenue_0001","answer":391035000000,"unit":"USD","confidence":0.82,"reasoning":"The filing reports net sales; I normalized to raw USD."}
```

Required fields:

- `id`
- `answer`
- `unit`

Optional fields:

- `confidence`
- `reasoning`

## Units and normalization

Use raw numeric answers unless the question clearly asks for a ratio, percent, or per-share value.

- Currency labels are raw USD values. If a filing says `$391,035 million`, output `391035000000` with unit `USD`.
- Percent labels are percentage points. If the answer is 12.5%, output `12.5` with unit `percent`.
- Per-share answers should use unit `USD/share` or `USD_per_share`.
- Ratios should use unit `ratio`.

The evaluator is tolerant of common numeric formatting such as commas, `$`, parentheses for negatives, `%`, and million/billion scale words, but best practice is to output clean numeric values.

## Scoring

The evaluator uses a weighted score per item:

- 80% numeric correctness within tolerance
- 10% unit correctness
- 5% valid required output fields
- 5% derived-answer sanity checks

The main metric is `overall_score` / `accuracy` in `results.json`.

## Constraints

- Use the filing context as the evidence source.
- Do not try to read private labels or any paths outside `--dataset_dir` and `--working_dir`.
- Do not modify files in `--dataset_dir`.
- Save execution logs in `agent_execution/` or `agent_execution.json` as required by SIA.
