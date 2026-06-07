# SIA-FinCheck Dataset Card

## Summary
SIA-FinCheck is a filing-text numerical QA benchmark built from current S&P 100 companies. Public examples contain SEC filing excerpts and questions. Private labels contain SEC XBRL companyfacts-derived numeric answers with source-fact provenance.

## Retrieval date
Build timestamp: 2026-06-06T20:54:13+00:00.

## Data sources
- S&P 100 constituent snapshot: Wikipedia S&P 100 page, cached under `data/raw/sp100/` and exported to `data/metadata/sp100_snapshot.csv`.
- Ticker/CIK mapping: SEC `company_tickers_exchange.json`.
- SEC submissions: `https://data.sec.gov/submissions/CIK##########.json`.
- SEC filing HTML: primary 10-K/10-Q document from SEC EDGAR Archives.
- SEC XBRL company facts: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, used only for private labels.

## Company universe
Companies are current S&P 100 constituents from the snapshot, deduplicated by CIK to handle multiple share classes. Mapped companies: 100.

## Forms included
Original `10-K` and `10-Q` filings only. Amendments are excluded. Both annual and quarterly filings are represented.

## Question templates
Direct questions cover revenue/net sales, net income, operating income, gross profit, diluted EPS, total assets, total liabilities, stockholders' equity, and cash. Derived questions cover operating margin, net margin, liabilities-to-assets ratio, and equity ratio.

## Label construction
Private labels are selected from SEC companyfacts entries that match the target filing accession number and form. Annual duration facts require FY periods with roughly annual durations. Quarterly duration facts require Q1/Q2/Q3 periods with quarter-length durations to avoid YTD ambiguity. Balance-sheet instant facts must match the filing report date. Derived labels are computed from compatible same-period facts.

## Leakage controls
Public files do not include ground-truth answer fields, source fact values, XBRL JSON, formulas, or XBRL tag names. Public context is cleaned text from SEC filing HTML.

## Splits
Target split sizes are train 1,200, validation 150, and test 150. The implemented policy is deterministic hybrid splitting with company-aware ordering and exact-size rebalancing.

## Counts
- Total examples: 1500
- Train: 1200
- Validation: 150
- Test: 150
- Direct examples: 1053
- Derived examples: 447
- 10-K examples: 450
- 10-Q examples: 1050

## Known limitations
- S&P 100 membership comes from a web snapshot rather than a licensed S&P feed.
- Filing context extraction is heuristic and favors financial statement sections plus metric anchors.
- Some valid SEC facts are skipped if period duration or accession linkage is ambiguous.
- Company-level split is hybrid-rebalanced to satisfy exact requested counts.

## Reproducibility
Run `uv run python -m sia_fincheck.build_dataset --target 1500` from the repository root. Raw artifacts are cached under `data/raw/`; processed public/private files are regenerated from those artifacts.
