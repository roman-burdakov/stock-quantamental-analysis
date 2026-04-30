# NVDA Quantamental Engine

**MIT 15.C51 — Project #2: Quantamental Analysis**

A reproducible quantamental pipeline that produces a professional equity research report for NVIDIA Corporation (NVDA) with a Buy/Hold/Sell recommendation backed by structured financial data, narrative-drift analysis, an interpretable ML driver model, and multi-scenario DCF valuation.

**Thesis:** A scalable quantamental research engine can combine structured filing data, narrative-disclosure changes, valuation discipline, and point-in-time controls to produce an auditable investment recommendation. For Nvidia, the core investment question is not whether AI growth has been strong, but whether today's valuation already prices in growth, margin, and durability assumptions that are too aggressive, reasonable, or too conservative.

**Fail-safe design:** The pipeline includes three hard gates — data validation, recommendation eligibility, and audit consistency — that prevent formal Buy/Hold/Sell ratings from being issued when financial data fails validation. This was added after a post-mortem where XBRL parsing returned FY2025 revenue as $27B instead of $130B, producing a catastrophically wrong Sell recommendation.

---

## Data Validation Gate

The pipeline enforces a three-gate architecture to ensure no formal recommendation is issued on unreliable data:

### Gate 1: Data Validation (after XBRL parsing)

After parsing XBRL companyfacts, every core metric (revenue, gross profit, operating income, net income, operating cash flow, capex, FCF, cash, total debt, diluted shares, diluted EPS, R&D) is cross-checked against published 10-K values with a 1% tolerance. The result is a `DataQualityStatus`:

| Status | Meaning |
|---|---|
| `PASS` | All core metrics within tolerance |
| `PASS_WITH_WARNINGS` | Non-critical deviations exist (e.g., minor rounding) |
| `DATA_BLOCKED` | One or more critical metrics fail validation — formal rating blocked |

When `DATA_BLOCKED`, the pipeline continues but all downstream outputs are labeled diagnostic-only.

### Gate 2: Recommendation Eligibility (before report generation)

A formal rating requires all blocking gates to pass: validated data, validated market price, documented DCF assumptions, no unresolved split-basis issues, no lookahead violations, and no audit contradictions. Non-blocking components (ML signal, NLP signal, peer multiples, segment charts) can be `diagnostic_only` without blocking a formal rating — they are simply excluded from score direction and disclosed.

### Gate 3: Audit Consistency (after report generation)

A post-report cross-check detects contradictions between output files:
- `limitations.md` says "no issues" but `data_quality_report.md` has failures
- Report shows Buy/Hold/Sell but `DataQualityStatus` is `DATA_BLOCKED`
- `model_audit.md` claims outperformance but walk-forward results show otherwise
- ML or NLP `diagnostic_only` components presented as supporting rating direction

If contradictions are found, the final package status is `failed` (not submission-ready).

---

## Report Modes

The report generator operates in one of three modes based on pipeline gate results:

| Mode | Trigger | Cover Page Rating | Valuation Label |
|---|---|---|---|
| `formal_rating` | All blocking gates pass | Buy / Hold / Sell | Formal valuation |
| `diagnostic_not_rated` | One or more blocking gates fail | "Not Rated — Data Validation Required" | "Diagnostic only — do not use for recommendation" |
| `failed` | Pipeline error or audit contradictions | "Report Generation Failed" | N/A |

The final package status (written to `outputs/audit_status.json`) is one of:
- `formal_rating_pass` — formal rating issued, audit consistency passes
- `diagnostic_not_rated_pass` — Not Rated issued, blockers disclosed, audit passes
- `failed` — contradictions found or pipeline error, not submission-ready

---

## Rubric Mapping

| Criterion (25% each) | How Addressed |
|---|---|
| **Completeness of financial analysis** | 10 years of XBRL-parsed financials, segment revenue normalization across label changes, 15+ computed ratios with TTM, FCF margin reconciliation, multi-scenario DCF, reverse-DCF grid, peer multiples, walk-forward ML validation against 4 baselines |
| **Novelty of analysis** | Narrative-drift scoring (TF-IDF + 9-theme keyword dictionaries) across consecutive filings, ML feature/target matrix with strict no-lookahead enforcement, reverse-DCF implied-expectations grid, scorecard-based recommendation framework, point-in-time controls throughout |
| **Readability and attractiveness** | Professional Markdown + PDF/HTML report (8–12 pages), 7 decision-useful exhibits with source captions, cover page with thesis and key risks, executive summary, 5 explanatory Jupyter notebooks |
| **Source attribution including LLM prompts** | Every data point traced to filing/API call via provenance log, `source_attribution.md` with accession numbers and URLs, `prompt_log.md` with every LLM interaction, `data_dictionary.md`, skeptical `model_audit.md` self-grading |

---

## Quick Start

### One-Command Run

```bash
python scripts/run_pipeline.py --ticker NVDA --report-date 2026-04-29 --price-date 2026-04-29
```

This fetches all data, parses filings, **validates parsed data against published values**, computes metrics, trains the ML model, runs valuation, **evaluates recommendation eligibility**, generates charts, assembles the report, and **performs the audit consistency check** — producing all required outputs including `outputs/audit_status.json`.

To run only the parsing and validation stage:

```bash
python scripts/run_pipeline.py --ticker NVDA --report-date 2026-04-29 --price-date 2026-04-29 --step parse
```

The `parse` stage includes the data validation gate. If `DataQualityStatus` is `DATA_BLOCKED`, subsequent stages still run but produce diagnostic-only outputs.

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--ticker` | `NVDA` | Ticker symbol |
| `--report-date` | `2026-05-01` | Frozen analysis cutoff — no data with `source_available_date > report_date` enters the report |
| `--price-date` | `2026-05-01` | Market price date for valuation (uses prior trading day if weekend/holiday) |
| `--force-refresh` | off | Re-download all data, ignoring cache |
| `--skip-tests` | off | Skip pytest after pipeline completion |
| `--output-format` | `both` | `pdf`, `html`, or `both` |
| `--step` | `all` | Run a single stage: `ingest`, `parse` (includes validation gate), `metrics`, `segments`, `text`, `nlp`, `ml`, `valuation`, `charts`, `report`, `audit` |

Partial reruns reuse cached upstream outputs in `data/raw/` and `data/processed/`.

---

## Installation

### Python Environment

Requires **Python ≥ 3.13**.

```bash
pip install -r requirements.txt
```

### SEC EDGAR Access

Create a `.env` file in the project root:

```
SEC_USER_AGENT=YourName your.email@example.com
```

SEC EDGAR requires a valid User-Agent header identifying the requester.

### Optional: PDF Generation

PDF output requires [WeasyPrint](https://weasyprint.org/) and its system dependencies:

```bash
# macOS
brew install pango

# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0

# Then install the Python package
pip install weasyprint
```

If PDF generation fails, the pipeline automatically falls back to HTML output and logs the issue in `outputs/limitations.md`.

### Optional Enhancements

Uncomment in `requirements.txt` or install directly:

```bash
pip install xgboost shap sentence-transformers bertopic ruptures
```

---

## File Structure

```
├── README.md                          ← You are here
├── requirements.txt                   ← Core + optional dependencies
├── pyproject.toml                     ← Project metadata
├── .env                               ← SEC_USER_AGENT (not committed)
│
├── src/
│   ├── config.py                      ← All parameters, dates, thresholds, dataclasses
│   ├── edgar_fetch.py                 ← SEC submissions, companyfacts, filing docs, market data
│   ├── xbrl_parser.py                 ← Companyfacts → structured financials + validation
│   ├── data_validation.py             ← Published-value validation gate + DataQualityStatus
│   ├── segment_revenue.py             ← Segment/platform revenue normalization
│   ├── filing_text_parser.py          ← Filing HTML → narrative section text
│   ├── financial_metrics.py           ← Ratios, growth, TTM, FCF margin
│   ├── nlp_features.py               ← TF-IDF similarity, keyword scores, narrative drift
│   ├── ml_models.py                   ← Ridge/ElasticNet, walk-forward, baselines
│   ├── valuation.py                   ← DCF, reverse-DCF grid, peer multiples, scenarios
│   ├── split_adjuster.py             ← Stock split adjustment (10:1 June 2024)
│   ├── charts.py                      ← 7 required exhibits with source captions
│   ├── report_utils.py               ← Jinja2 → Markdown + PDF/HTML (3 report modes)
│   ├── audit_utils.py                ← Attribution, prompt log, data quality, self-audit, consistency
│   └── templates/                     ← Jinja2 report templates
│       ├── report_base.md.j2
│       ├── cover_page.md.j2
│       └── executive_summary.md.j2
│
├── scripts/
│   └── run_pipeline.py                ← One-command pipeline orchestrator
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb        ← SEC fetch + caching walkthrough
│   ├── 02_parsing_and_segments.ipynb  ← XBRL, validation, segment normalization
│   ├── 03_financial_analysis_and_nlp.ipynb  ← Metrics, narrative drift
│   ├── 04_ml_model.ipynb              ← Feature matrix, training, validation
│   └── 05_valuation_and_report.ipynb  ← DCF, reverse-DCF, recommendation
│
├── data/
│   ├── raw/                           ← Cached SEC downloads, market data, provenance log
│   ├── interim/                       ← Parsed filing sections (JSON)
│   └── processed/                     ← Final CSVs (metrics, segments, NLP, ML matrix)
│
├── outputs/
│   ├── nvda_quantamental_report.md    ← Main report (Markdown)
│   ├── nvda_quantamental_report.pdf   ← Main report (PDF, or .html fallback)
│   ├── executive_summary.md           ← 2-page executive summary
│   ├── source_attribution.md          ← Full provenance and source references
│   ├── prompt_log.md                  ← LLM usage log with verification
│   ├── data_dictionary.md             ← Variable definitions and lineage
│   ├── data_quality_report.md         ← Missing tags, validation, coverage
│   ├── model_audit.md                 ← ML audit + skeptical self-grading
│   ├── limitations.md                 ← Data gaps, assumptions, caveats
│   ├── audit_status.json              ← Machine-readable pipeline gate results and package status
│   ├── figures/                       ← All chart PNGs with source captions
│   └── tables/                        ← Supplementary tables
│
└── tests/
    ├── fixtures/                      ← Offline test data (sample filings, known values)
    ├── test_financial_metrics.py      ← Ratio calculations vs manual values
    ├── test_valuation_math.py         ← DCF, terminal value, scenario weighting
    ├── test_no_lookahead.py           ← ML matrix date integrity
    ├── test_report_date_filtering.py  ← No future data in report context
    ├── test_filing_parser.py          ← Extraction on real Nvidia filings
    ├── test_missing_data.py           ← Graceful null/gap handling
    ├── test_ingestion.py              ← Submissions filtering, caching, LOCF
    ├── test_xbrl_parser.py            ← Schema, deduplication, validation
    ├── test_segment_revenue.py        ← Label mapping consistency
    ├── test_nlp_features.py           ← Keyword scoring, TF-IDF validity
    ├── test_ml_models.py              ← Model training and evaluation
    └── test_audit_utils.py            ← Attribution and audit validation
```

---

## Data Sources

| Source | What | Access |
|---|---|---|
| **SEC EDGAR** — `data.sec.gov/submissions/` | Filing metadata (10-K, 10-Q) | Public API, rate-limited 10 req/s |
| **SEC EDGAR** — `data.sec.gov/api/xbrl/companyfacts/` | Structured XBRL financial facts | Public API |
| **SEC EDGAR** — Filing documents | Full HTML for narrative text extraction | Public API |
| **Yahoo Finance** (`yfinance`) | Daily adjusted close prices (NVDA + peers) | Public API |
| **Yahoo Finance** (`yfinance`) | Peer financials (market cap, debt, cash, revenue, EBITDA) | Public API |
| **Manual / Analyst Judgment** | Scenario assumptions (WACC, growth rates, FCF margins), recommendation thresholds, keyword dictionaries | Documented in `config.py` and `source_attribution.md` |

All downloads are cached in `data/raw/`. Subsequent runs skip fetching unless `--force-refresh` is passed.

---

## Notebook Walkthrough

The five notebooks serve as explanatory companions to the pipeline — each imports from `src/` and walks through methodology, intermediate outputs, and interpretation.

| Notebook | Purpose |
|---|---|
| **01 — Data Ingestion** | Demonstrates SEC EDGAR fetching, caching behavior, market data retrieval, and provenance logging |
| **02 — Parsing and Segments** | XBRL parsing with concept fallbacks, **data validation gate** (published-value cross-checks, DataQualityStatus), segment revenue normalization across Nvidia's label changes, filing text extraction |
| **03 — Financial Analysis and NLP** | Ratio computation, TTM aggregation, inflection-point detection, TF-IDF narrative drift, 9-theme keyword scoring with filing snippet examples |
| **04 — ML Model** | Feature/target matrix construction with no-lookahead enforcement, Ridge/ElasticNet training, walk-forward validation, baseline comparison, honest performance disclosure |
| **05 — Valuation and Report** | Multi-scenario DCF with FCF margin reconciliation, reverse-DCF implied-expectations grid, peer multiples, **recommendation eligibility gate**, scorecard-based recommendation, report assembly with mode support (formal_rating / diagnostic_not_rated / failed) |

---

## Runtime Estimates

Approximate times on a standard machine with internet access:

| Stage | First Run | Cached Rerun |
|---|---|---|
| Ingestion (SEC + market data) | 2–5 min | < 5 sec |
| XBRL parsing + validation gate | 10–30 sec | 10–30 sec |
| Financial metrics | 5–10 sec | 5–10 sec |
| Segment normalization | 5–10 sec | 5–10 sec |
| Filing text extraction | 10–30 sec | 10–30 sec |
| NLP features | 10–30 sec | 10–30 sec |
| ML training + validation | 5–15 sec | 5–15 sec |
| Valuation + eligibility gate | 5–10 sec | 5–10 sec |
| Chart generation | 10–20 sec | 10–20 sec |
| Report rendering (Markdown) | 5–10 sec | 5–10 sec |
| Report rendering (PDF) | 10–30 sec | 10–30 sec |
| Audit consistency check | 5–10 sec | 5–10 sec |
| **Total** | **~4–8 min** | **~1–3 min** |

PDF generation requires WeasyPrint and its system dependencies. If unavailable, HTML fallback adds negligible time.

---

## Point-in-Time Controls

Every data row in the pipeline carries a `source_available_date` — the date the data point became publicly available (filing date for SEC filings, trade date for prices, retrieval date for peer snapshots).

Two enforcement modes:

1. **Report generation:** Only data with `source_available_date <= report_date` enters the report context. Enforced at ingestion and in `build_report_context()`.
2. **ML validation (walk-forward):** Strict no-lookahead. Each row in the feature/target matrix satisfies `feature_available_date <= prediction_date` AND `target_available_date > prediction_date`. Verified by `validate_no_lookahead_matrix()` and `tests/test_no_lookahead.py`.

The report includes an audit table with concrete examples of allowed vs. disallowed data usage.

---

## Pipeline Outputs

The pipeline produces the following outputs:

| Output | Description |
|---|---|
| `data/processed/nvda_metrics.csv` | Financial ratios and growth rates |
| `data/processed/nvda_segment_revenue_normalized.csv` | Normalized segment revenue |
| `data/processed/nvda_nlp_features.csv` | Narrative drift and keyword features |
| `data/processed/ml_feature_target_matrix.csv` | ML feature/target matrix |
| `outputs/nvda_quantamental_report.md` | Main report (Markdown) |
| `outputs/nvda_quantamental_report.pdf` | Main report (PDF, or `.html` fallback) |
| `outputs/executive_summary.md` | 2-page executive summary |
| `outputs/source_attribution.md` | Full provenance and source references |
| `outputs/prompt_log.md` | LLM usage log with verification |
| `outputs/data_dictionary.md` | Variable definitions and lineage |
| `outputs/data_quality_report.md` | Missing tags, validation results, coverage |
| `outputs/model_audit.md` | ML audit + skeptical self-grading |
| `outputs/limitations.md` | Data gaps, assumptions, caveats |
| `outputs/audit_status.json` | Machine-readable gate results and final package status |
| `outputs/figures/*` | All chart PNGs with source captions |

`audit_status.json` is the machine-readable summary of all pipeline gates. It contains: `overall_status`, `data_quality_status`, `recommendation_eligibility`, per-component statuses, `checks_run`/`checks_passed`/`checks_failed`, `blocking_issues`, and a timestamp. Use it to programmatically verify the pipeline produced a submission-ready package.

---

## Limitations

- **Annual-only XBRL data for some metrics:** Certain line items are only reported annually, limiting quarterly granularity. ML model may be underpowered with few training samples — disclosed honestly in `model_audit.md`.
- **Segment label changes:** Nvidia has restructured reportable segments and market/platform categories multiple times. Normalization relies on a manually curated mapping; edge cases are documented in `data_quality_report.md`.
- **Peer EV data staleness:** Enterprise value inputs from Yahoo Finance may be stale. Peers with data older than 90 days have EV-based multiples skipped and are logged in `limitations.md`.
- **NLP baseline approach:** Narrative drift uses TF-IDF and keyword dictionaries (not deep learning). Optional sentence-transformer embeddings and BERTopic are available but not enabled by default.
- **Single-ticker focus:** The pipeline is parameterized for ticker switching via `config.py`, but has only been validated end-to-end for NVDA.
- **PDF dependency:** PDF output requires WeasyPrint + system libraries (Pango). HTML fallback is automatic if unavailable.
- **No real-time data:** All analysis is frozen at `report_date` / `price_date`. The pipeline does not stream live data.
- **Valuation assumptions are analyst judgment:** WACC, terminal growth, scenario probabilities, and FCF margin trajectories are documented assumptions, not derived from a formal CAPM or market-implied calculation.

---

## Running Tests

```bash
pytest
```

Unit tests run offline using fixtures in `tests/fixtures/`. Integration tests may require populated `data/raw/` (run the ingestion stage first).

---

## License

Academic project — MIT 15.C51.
