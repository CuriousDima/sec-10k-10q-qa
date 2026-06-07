# Sample Related Task Descriptions

## Task 1: Numerical QA over SEC filings

Given public excerpts from 10-K and 10-Q filings, answer a list of numerical finance questions. Each question may ask for a directly reported number such as revenue, assets, liabilities, cash flow, EPS, or gross profit, or a simple derived value such as a ratio or percentage. The output must be JSONL with one answer per input ID.

-----

## Task 2: Financial table extraction and normalization

Read company filing text containing financial statements and footnotes. Extract the requested numeric value, normalize it to the required unit, and return a concise machine-readable answer. Currency answers should be raw USD; percentages should be percentage points; per-share values should use USD/share.

-----

## Task 3: Multi-record benchmark inference

Process many independent benchmark examples. For each example, use only its supplied filing context and metadata, avoid private labels, produce an answer with a unit, and save all predictions to `submission.jsonl` in the writable working directory.
