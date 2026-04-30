# Requirements Document

## Introduction

The NVDA Quantamental Engine is a reproducible, submission-ready research system for MIT 15.C51 Project #2 (Quantamental Analysis). It produces a professional 8–12 page equity research report for Nvidia with either (a) a formal Buy/Hold/Sell recommendation when all eligibility gates pass, or (b) a "Not Rated — Data Validation Required" diagnostic report when validation, valuation, or audit gates block a formal rating. The report includes a valuation-supported intrinsic value estimate, defensible financial thesis, documented ML methodology, narrative-drift analysis, scalability discussion, and complete source attribution.

Grading rubric (equally weighted):
1. Completeness of financial analysis (25%)
2. Novelty of analysis (25%)
3. Readability and attractiveness (25%)
4. Source attribution including LLM prompts (25%)

Strategic thesis: "A scalable quantamental research engine can combine structured filing data, narrative-disclosure changes, valuation discipline, and point-in-time controls to produce an auditable investment recommendation. For Nvidia, the core investment question is not whether AI growth has been strong, but whether today's valuation already prices in growth, margin, and durability assumptions that are too aggressive, reasonable, or too conservative."

**Post-mortem context:** The initial pipeline produced a catastrophically wrong Sell recommendation (−77.7% downside) because XBRL parsing returned FY2025 revenue as $26.974B instead of the published $130.497B. The parser confused a quarterly/YTD value with the annual total. This spec revision adds hard data-validation gates, fiscal-year selection correctness requirements, and recommendation eligibility controls to prevent any formal rating from being issued when financial data fails validation.

## Glossary

- **Engine**: The complete system
- **report_date**: Frozen analysis cutoff date in config; no data with `source_available_date > report_date` enters the report or model
- **price_date**: Market price date for valuation; latest adjusted close on or before this date (uses prior trading day if weekend/holiday)
- **source_available_date**: The date a data point became publicly available (filing_date for SEC filings, trade date for prices, retrieval date for peer snapshots)
- **DataQualityStatus**: Pipeline-wide validation state: PASS, PASS_WITH_WARNINGS, DATA_BLOCKED
- **ValidatedMetric**: A parsed metric cross-checked against a published reference value with tolerance, severity, and blocker status
- **RecommendationStatus**: Eligibility gate output: formal_rating, diagnostic_not_rated, or failed
- **ComponentStatus**: Per-component quality state: usable, diagnostic_only, unavailable, blocked
- **SEC_Submissions_Fetcher**: Retrieves filing metadata from `data.sec.gov/submissions/`
- **CompanyFacts_Fetcher**: Retrieves structured XBRL facts from `data.sec.gov/api/xbrl/companyfacts/`
- **Filing_Document_Fetcher**: Downloads full filing HTML for text extraction
- **XBRL_Parser**: Parses companyfacts JSON into structured financial tables
- **Filing_Text_Parser**: Extracts narrative sections using form-specific Item numbers
- **SegmentRevenueNormalizer**: Normalizes Nvidia's changing segment/platform labels into consistent categories
- **NLP_Feature_Extractor**: Computes TF-IDF similarity, keyword scores; optional embeddings/topics
- **Financial_Metrics_Calculator**: Computes ratios, growth, TTM from parsed XBRL
- **ML_Driver_Model**: Interpretable model (Ridge/ElasticNet primary) forecasting a financial driver
- **Valuation_Module**: DCF, reverse-DCF grid, peer multiples, scenarios
- **Report_Generator**: Assembles final report (Markdown + PDF or HTML)
- **Audit_Module**: Attribution, prompt log, data quality, self-audit, consistency checks
- **Peer_Group**: Core semi (AMD, AVGO, INTC, QCOM, MRVL), infrastructure (TSM, ASML), AI capex context (MSFT, AMZN, GOOGL, META)
- **target_price**: Intrinsic value estimate as of report_date (not a formal 12-month sell-side target unless explicitly stated)

## Requirements

### Requirement 1: SEC Data Ingestion

**User Story:** As a quantamental analyst, I want Nvidia's filings and structured financials from SEC EDGAR with caching and availability tracking.

#### Acceptance Criteria

1. THE SEC_Submissions_Fetcher SHALL retrieve filing metadata from `data.sec.gov/submissions/CIK{cik}.json`, returning: accession_number, filing_date, report_period, form_type, sec_url, and `source_available_date` (= filing_date)
2. THE SEC_Submissions_Fetcher SHALL filter to 10-K and 10-Q filings where `source_available_date <= report_date` and fiscal year within configured range
3. THE CompanyFacts_Fetcher SHALL retrieve structured XBRL facts from `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`, preserving source availability metadata per fact
4. THE Filing_Document_Fetcher SHALL download full filing HTML for each accession number, attaching accession, filing_date, and source_available_date
5. ALL SEC requests SHALL include a valid User-Agent header ("ProjectName contact@email.com")
6. ALL SEC requests SHALL enforce rate limiting (max 10 req/s) with exponential backoff on 429/5xx (max 3 retries)
7. ALL downloads SHALL be cached in `data/raw/`; subsequent runs skip fetching unless `--force-refresh` is passed
8. THE Engine SHALL record provenance for every download in `data/raw/provenance_log.jsonl`: timestamp, source_url, accession_number, filing_date, source_available_date, report_period

### Requirement 2: Market and Peer Data Ingestion

**User Story:** As a quantamental analyst, I want market prices and peer financials as of frozen dates with staleness controls.

#### Acceptance Criteria

1. THE Engine SHALL define `report_date` and `price_date` in config; current price = latest adjusted close on or before `price_date` (prior trading day if weekend/holiday)
2. THE Engine SHALL retrieve daily adjusted close for Nvidia and Peer_Group for 10 years; each price row carries `source_available_date` = trade_date
3. No market data with `source_available_date > price_date` SHALL be used in valuation
4. LOCF interpolation allowed ONLY for weekend/holiday gaps (max 3 calendar days); longer gaps logged and excluded
5. THE Engine SHALL retrieve peer financials: market_cap, enterprise_value (or inputs: market_cap + debt − cash), revenue, EBITDA, net_income, FCF, with source_date and source_available_date per peer
6. IF enterprise_value inputs are missing or stale (source_date > 90 days before report_date), EV-based multiples SHALL be skipped for that peer and logged in limitations
7. Peer financial data SHALL NOT be mixed with historical model training without explicit date labeling
8. ALL market data cached in `data/raw/` with source attribution metadata

### Requirement 3: Structured Financial Data Parsing

**User Story:** As a quantamental analyst, I want XBRL financials parsed with availability dates and validated against published values.

#### Acceptance Criteria

1. THE XBRL_Parser SHALL extract: revenue, COGS, gross_profit, operating_income, net_income, diluted_EPS, R&D, SG&A, operating_cash_flow, capex, stock_based_compensation (if available), cash_and_securities, short_term_debt, long_term_debt, total_debt, total_assets, shareholders_equity, current_assets, current_liabilities, diluted_shares, segment_revenue with dimensions
2. Every parsed fact SHALL carry: ticker, fiscal_period, fiscal_year, filing_date, source_available_date, accession_number, metric_name, value, unit, form_type
3. THE Parser SHALL handle multiple XBRL concept names per metric (priority-ordered fallback)
4. THE Parser SHALL deduplicate: prefer 10-K over 10-Q, latest filing over amendments
5. Metrics that cannot be computed from reliable tags SHALL be omitted or marked unavailable, NOT silently estimated
6. THE Engine SHALL validate against Nvidia published 10-K values for latest 3 FYs: revenue, gross_margin, operating_margin, net_income, operating_cash_flow, capex, FCF, diluted_shares, cash, debt — with tolerance thresholds and pass/fail/missing status
7. Manual validation values SHALL appear in source_attribution.md
8. THE Engine SHALL produce `outputs/data_quality_report.md`: missing tags, fallback tags, coverage %, validation results

### Requirement 4: Segment/Platform Revenue Normalization

**User Story:** As a quantamental analyst, I want Nvidia's changing segment labels normalized into consistent categories for trend analysis.

#### Acceptance Criteria

1. THE SegmentRevenueNormalizer SHALL produce `data/processed/nvda_segment_revenue_normalized.csv` with columns: ticker, fiscal_period, fiscal_year, filing_date, source_available_date, source_accession, original_label, normalized_category, value, unit, extraction_method, mapping_notes
2. THE Normalizer SHALL handle changes in Nvidia reportable segments and market/platform categories over time
3. IF XBRL dimensions are insufficient, table extraction from filing text is allowed but must be documented in extraction_method and mapping_notes
4. Report charts using segment data SHALL label whether they use reportable segments or market/platform revenue
5. Missing or changed labels SHALL be logged in data_quality_report.md

### Requirement 5: Filing Text Extraction

**User Story:** As a quantamental analyst, I want narrative sections extracted using form-specific Item numbers with coverage tracking.

#### Acceptance Criteria

1. FOR 10-K: extract Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 7A (Quantitative Disclosures)
2. FOR 10-Q: extract Part II Item 1A (Risk Factors), Part I Item 2 (MD&A)
3. Extraction SHALL use Item-number regex patterns as primary method, section-title matching as fallback
4. Each extracted section SHALL be stored as TextSectionRecord with: accession_number, form_type, section_name, text, char_count, filing_date, source_available_date, parse_status
5. THE Engine SHALL produce a section extraction coverage report (in data_quality_report.md): accession, form_type, section, char_count, parse_status, warning
6. Tests SHALL validate extraction on at least 2 real Nvidia filings (1 10-K, 1 10-Q) using fixture or cached files

### Requirement 6: Financial Metrics Computation

**User Story:** As a quantamental analyst, I want comprehensive ratios with provenance and null handling.

#### Acceptance Criteria

1. Compute per fiscal period: gross_margin, operating_margin, net_margin, ROE, ROA, revenue_growth_YoY, OI_growth_YoY, FCF, FCF_margin, debt_to_equity, current_ratio, R&D_%_revenue, SG&A_%_revenue, diluted_EPS, segment_mix
2. Compute TTM versions from quarterly data
3. Missing inputs → null + log
4. Flag inflection points (YoY change > 2σ from historical mean)
5. Compute historical FCF margin for reconciliation with valuation assumptions
6. Store in `data/processed/nvda_metrics.csv` with: ticker, fiscal_period, filing_date, source_available_date, metric_name, metric_value, source_accession, unit
7. Round-trip property: recomputing from same inputs produces identical results

### Requirement 7: Narrative Drift Analysis

**User Story:** As a quantamental analyst, I want to measure filing language changes to detect business inflections.

#### Acceptance Criteria

1. Compute pairwise TF-IDF cosine similarity between consecutive filings for Risk Factors and MD&A (required)
2. Implement keyword dictionary scoring for 9 themes: AI/accelerated computing, Data Center, export controls/China, supply constraints, competition, customer concentration, gaming cyclicality, margin/pricing pressure, inventory/demand cyclicality
3. Produce Narrative_Drift time series per section
4. Produce emerging/fading topic table
5. Output 3–5 filing snippet examples with exact source references
6. NLP features SHALL carry source_accession and source_available_date
7. For current report: use all filings with source_available_date <= report_date
8. For ML features: obey feature_available_date <= prediction_date
9. Store in `data/processed/nvda_nlp_features.csv`
10. OPTIONAL: sentence-transformer embeddings, BERTopic/LDA, change-point detection

### Requirement 8: ML Financial Driver Model

**User Story:** As a quantamental analyst, I want an interpretable model forecasting revenue growth with honest validation.

#### Acceptance Criteria

1. Forecast primary target: next-quarter or next-year revenue growth using quarterly/TTM data
2. Primary model: Ridge or ElasticNet; tree models optional
3. Walk-forward validation with expanding window (min 3 years training)
4. Compare against 4 baselines: last_period, trailing_4q_avg, three_year_avg, linear_trend
5. THE ML_Driver_Model SHALL output `data/processed/ml_feature_target_matrix.csv` with: feature_period, feature_available_date, prediction_date, target_period, target_available_date, target_name, target_value, feature columns, source_accessions
6. A row is valid ONLY IF all feature_available_dates <= prediction_date AND target_available_date > prediction_date
7. THE Engine SHALL implement `validate_no_lookahead_matrix()` verifying row validity
8. Report MAE, RMSE, directional accuracy for all models and baselines
9. IF model does not outperform baselines, disclose honestly and explain value (feature organization, scenario discipline, auditability)
10. IF annual-only data used, audit SHALL state ML is exploratory and underpowered
11. Output model audit to `outputs/model_audit.md`
12. OPTIONAL: SHAP (if tree model selected), peer-panel model (requires 10-year peer historicals)

### Requirement 9: Valuation and Recommendation

**User Story:** As a quantamental analyst, I want DCF with explicit FCF methodology, reverse-DCF grid, and a defensible recommendation.

#### Acceptance Criteria

1. DCF with explicit assumptions: revenue_cagr, fcf_margin_start, fcf_margin_terminal, fcf_margin_trajectory, tax_rate (if used), terminal_growth, WACC, net_cash, diluted_shares, sbc_treatment
2. Projected FCF = projected_revenue × scenario_FCF_margin (unless more detailed model explicitly implemented)
3. DCF output SHALL include: projected_revenue, fcf_margin, projected_FCF, discount_factors, PV_of_FCF, terminal_value, PV_terminal, enterprise_value, net_cash_bridge, equity_value, per_share_value
4. Include historical FCF margin reconciliation showing how assumed margins compare to actuals
5. Document SBC treatment (included in FCF or adjusted)
6. WACC × terminal_growth sensitivity table (5×5 minimum)
7. Reverse-DCF implied-expectations grid: revenue_CAGR × terminal_FCF_margin, implied share price per cell, highlight cells near current price
8. One-variable solvers (implied CAGR holding margin constant, implied margin holding CAGR constant) allowed as supplemental only; SHALL NOT present single pair as uniquely correct
9. Peer multiples: EV/Revenue, EV/EBITDA, P/E, FCF_yield with limitations; skip EV multiples for peers with missing/stale EV inputs
10. Bear/base/bull scenarios with probability weights → probability-weighted target
11. Recommendation thresholds (applied ONLY after Requirement 17 eligibility gate passes): Buy requires prob-weighted upside >15% AND base-case upside >15% AND bear downside not worse than −25%; Hold when balanced or implied expectations plausible but not attractive; Sell when downside >10% or reverse-DCF assumptions implausibly aggressive. WHEN the eligibility gate has not passed, these thresholds SHALL NOT be evaluated and the rating SHALL be "Not Rated."
12. Recommendation scorecard: valuation_upside, reverse_dcf_plausibility, ml_signal, narrative_drift_signal, risk_concentration, analyst_judgment
13. State what must be true for Buy, Hold, Sell
14. Target price is intrinsic value estimate as of report_date (not formal 12-month target unless explicitly stated)

### Requirement 10: Report Generation

**User Story:** As a quantamental analyst, I want a professional 8–12 page report with guaranteed output format.

#### Acceptance Criteria

1. Always produce `outputs/nvda_quantamental_report.md`
2. Always produce EITHER `outputs/nvda_quantamental_report.pdf` OR `outputs/nvda_quantamental_report.html`; if PDF fails, HTML fallback is acceptable and logged in limitations.md
3. Main report sections: Cover Page, Executive Summary, Company Overview, Historical Financial Analysis, Driver Analysis, ML/Quantamental Insights, Valuation, Risks, Catalysts, Scalability and Automation, Point-in-Time Controls
4. Cover Page: company, ticker, report_date, rating (Buy/Hold/Sell OR "Not Rated — Data Validation Required"), current price, target price (or "N/A — Diagnostic Only" if blocked), upside/downside %, one-sentence thesis, 3–5 bullets, 3 key risks
5. Main report: 6–8 decision-useful exhibits (revenue/segment mix, margin trend, FCF trend, narrative drift, DCF scenarios, reverse-DCF grid, recommendation scorecard)
6. Point-in-Time Controls section with audit table: ≥3 examples of allowed vs disallowed data usage
7. Full source attribution, prompt log, data dictionary, model audit, data quality report → appendix output files with short references in main report
8. Produce `outputs/executive_summary.md` (max 2 pages)
9. Report context SHALL exclude any data with source_available_date > report_date

### Requirement 11: Source Attribution and Audit

**User Story:** As a quantamental analyst, I want every material claim traceable and LLM use fully disclosed.

#### Acceptance Criteria

1. `outputs/source_attribution.md`: every SEC filing (filing_date, report_period, accession, URL), market data sources, peer sources, manual assumptions, analyst judgments, LLM content with verification
2. `outputs/prompt_log.md`: seeded at project start with master prompt; every LLM use appended with: exact prompt, model, purpose, output used, verification source, whether it affected conclusion
3. Statement: "LLM outputs were not treated as sources of truth. Factual claims verified against filings, market data, or explicitly labeled assumptions."
4. `outputs/data_dictionary.md`: every processed variable with source, unit, period, source_available_date, transformation, limitations
5. `outputs/limitations.md`: data gaps, model limitations, assumptions, unresolved blockers
6. `outputs/model_audit.md`: skeptical self-grading (4 criteria × 25%), identifying unsupported claims, weak assumptions, stale data, lookahead risks, valuation inconsistencies, generic prose, missing sources; SHALL NOT inflate grade
7. Every material factual claim, exhibit, valuation input, model input, and analyst assumption must be traceable

### Requirement 12: Testing and Validation

**User Story:** As a quantamental analyst, I want automated tests ensuring correctness, no-lookahead, and report-date filtering.

#### Acceptance Criteria

1. `tests/test_financial_metrics.py`: verify calculations against manually computed values for ≥3 fiscal periods
2. `tests/test_valuation_math.py`: verify DCF PV, terminal value, FCF projection, scenario weighting, sensitivity grid, reverse-DCF grid
3. `tests/test_no_lookahead.py`: verify ML feature/target matrix validity (all feature dates <= prediction_date, target date > prediction_date)
4. `tests/test_report_date_filtering.py`: verify no report output or model input uses data with source_available_date > report_date
5. `tests/test_filing_parser.py`: verify extraction on ≥2 real Nvidia filings using fixtures
6. `tests/test_missing_data.py`: graceful handling of missing tags, prices, sections; EV multiples skipped when inputs missing
7. Unit tests SHALL run offline using `tests/fixtures/`; integration tests may require populated `data/raw/`
8. ALL tests pass on clean checkout with data/raw/ populated

### Requirement 13: Reproducibility and Scalability

**User Story:** As a grader, I want one-command reproducibility with useful flags and a scalability discussion.

#### Acceptance Criteria

1. `scripts/run_pipeline.py` with: `python scripts/run_pipeline.py --ticker NVDA --report-date YYYY-MM-DD --price-date YYYY-MM-DD`
2. Flags: `--force-refresh`, `--skip-tests`, `--output-format pdf|html|both`, `--step all|ingest|parse|metrics|segments|text|nlp|ml|valuation|report|audit`
3. Notebooks as explanatory wrappers (NOT only reproducibility path)
4. `README.md`: objective, rubric mapping, file structure, data sources, one-command run, notebook walkthrough, runtime estimates, limitations
5. `src/config.py` parameterizes ticker, peers, dates, assumptions; switching companies requires only config changes
6. Report Scalability section: automatable vs human-judgment parts, sector limitations, error modes, analyst time saved, quality controls

---

### Requirement 14: Data Integrity and Validation Gate

**User Story:** As a quantamental analyst, I want a hard gate that blocks formal Buy/Hold/Sell recommendations when parsed financial data fails validation against published values, so that catastrophic errors like the FY2025 revenue misparse ($27B vs $130B) can never produce a formal rating.

#### Acceptance Criteria

1. THE Engine SHALL define a DataQualityStatus with three states: `PASS` (all core metrics within tolerance), `PASS_WITH_WARNINGS` (non-critical deviations exist), `DATA_BLOCKED` (one or more critical metrics fail validation)
2. THE Engine SHALL define a ValidatedMetric schema with fields: metric_name, fiscal_year, parsed_value, published_value, diff_pct, tolerance, status (pass/warn/fail/missing), severity (critical/major/minor), blocker (bool)
3. THE following core metrics SHALL be validated before any formal recommendation is issued: revenue, gross_profit, operating_income, net_income, operating_cash_flow, capex, FCF, cash_and_securities, total_debt, diluted_shares, diluted_EPS, R&D
4. WHEN a core annual metric's parsed_value differs from published_value by more than 1% (default tolerance), THEN the metric status SHALL be `fail` and severity SHALL be `critical`
5. WHEN any ValidatedMetric has severity=critical AND blocker=true, THEN DataQualityStatus SHALL be `DATA_BLOCKED`
6. WHEN DataQualityStatus is `DATA_BLOCKED`, THEN the report cover page SHALL show "Not Rated — Data Validation Required" instead of Buy/Hold/Sell
7. WHEN DataQualityStatus is `DATA_BLOCKED`, THEN all DCF and valuation outputs SHALL be labeled "Diagnostic only — do not use for recommendation"
8. WHEN validation failures exist, THEN `outputs/limitations.md` SHALL list each failure with metric_name, fiscal_year, parsed_value, published_value, diff_pct — the file SHALL NOT say "No data quality issues recorded" when failures are present
9. THE Engine SHALL produce `outputs/data_quality_report.md` with a machine-readable validation summary table and a human-readable narrative
10. THE Engine SHALL produce `outputs/audit_status.json` with fields: overall_status, data_quality_status, recommendation_eligibility, component_statuses, blocking_issues[], timestamp
11. WHEN a published reference value is missing for a valuation-critical metric in the base/current fiscal year (revenue, operating_cash_flow, capex, diluted_shares, cash, debt, market_price), THEN the metric SHALL require either a manually sourced validation value (with source in source_attribution.md) or DataQualityStatus SHALL be DATA_BLOCKED
12. THE Engine SHALL compute validation_coverage_pct = (validated_pass + validated_warn) / total_core_metric_year_pairs. WHEN validation_coverage_pct for the latest 3 fiscal years falls below 90%, DataQualityStatus SHALL be DATA_BLOCKED unless the missing references are non-valuation-critical

### Requirement 15: Fiscal-Year Selection Correctness

**User Story:** As a quantamental analyst, I want the XBRL parser to correctly distinguish annual, quarterly, and YTD values so that annual metrics always reflect the full fiscal year.

#### Acceptance Criteria

1. Every FinancialFact SHALL include: fiscal_period_type (annual/quarterly/ytd/instant), period_start, period_end, duration_days, frame (SEC XBRL frame string if available), xbrl_concept, selection_rank, selection_reason
2. WHEN selecting the annual value for a metric, the parser SHALL prefer facts from 10-K filings with annual-duration periods (duration 350–380 days for fiscal-year facts)
3. WHEN a fact has duration < 100 days, it SHALL be classified as quarterly and SHALL NOT be selected as an annual value
4. WHEN a fact has duration between 100 and 340 days, it SHALL be classified as YTD and SHALL NOT be selected as an annual value for any core validation metric (revenue, gross_profit, operating_income, net_income, operating_cash_flow, capex, FCF, cash_and_securities, total_debt, diluted_shares, diluted_EPS, R&D). IF no annual-duration fact exists for a core metric, the metric SHALL be marked unavailable. IF the unavailable metric is valuation-critical, DataQualityStatus SHALL be DATA_BLOCKED. For non-core context-only metrics, a YTD fallback MAY be used only if labeled "diagnostic — YTD fallback" and excluded from valuation inputs.
5. WHEN multiple candidate facts exist for the same metric and fiscal year, the parser SHALL rank by: (a) 10-K annual-duration preferred, (b) latest amendment preferred, (c) frame-tagged facts preferred over unframed, and record selection_rank and selection_reason
6. THE parser SHALL handle fiscal-year amendments deterministically: the latest amendment for a given fiscal year supersedes prior filings
7. Tests SHALL verify that FY2025 NVDA revenue parses as approximately $130.5B (±1%), not $27B — this is a regression test for the original failure

### Requirement 16: Valuation Guardrails

**User Story:** As a quantamental analyst, I want valuation outputs to include sanity checks that flag implausible results before they reach the recommendation.

#### Acceptance Criteria

1. WHEN any valuation-critical input (revenue, FCF, net_cash, diluted_shares, market_price) has failed data validation, THEN formal valuation SHALL be blocked and outputs labeled "Diagnostic only"
2. WHEN the DCF-derived target price differs from the current market price by more than 50% (upside or downside), THEN the report SHALL include a sanity-check bridge explanation identifying which assumptions drive the gap
3. THE reverse-DCF grid SHALL use a reasonable configured assumption range. WHEN the current market price is outside the grid range, the report SHALL explicitly state that the current price lies outside the modeled grid and the solver SHALL compute the implied CAGR and/or terminal FCF margin required to justify the current price. The report SHALL explain whether those implied assumptions are within, above, or far outside historical/plausible ranges
4. WHEN bear-case downside exceeds −40%, THEN the report SHALL flag this as an extreme scenario requiring additional justification
5. THE sensitivity table SHALL use a reasonable configured range; if the current market price falls outside the table, the report SHALL note this rather than extending to unreasonable assumptions

### Requirement 17: Recommendation Eligibility Gate

**User Story:** As a quantamental analyst, I want a formal checklist that must pass before any Buy/Hold/Sell rating is issued, so that incomplete or unreliable inputs cannot produce a formal recommendation.

#### Acceptance Criteria

1. A formal rating (Buy/Hold/Sell) SHALL require ALL blocking gates to pass: (a) DataQualityStatus = PASS or PASS_WITH_WARNINGS, (b) validated market price available, (c) DCF assumptions documented and internally consistent, (d) no unresolved share-count or split-basis issue, (e) no report-date/lookahead violation, (f) no audit contradiction between output files
2. Non-blocking diagnostic components (ML signal, NLP signal, peer multiples, segment chart) SHALL NOT automatically prevent a formal rating. WHEN a non-blocking component has status `diagnostic_only`, it SHALL be excluded from score direction and clearly disclosed, but the formal rating MAY still proceed if all blocking gates pass
3. WHEN any blocking gate fails, THEN the report SHALL show "Not Rated — [reason]" and the recommendation section SHALL explain which gates failed
4. Each scorecard component SHALL show a ComponentStatus: `usable`, `diagnostic_only`, `unavailable`, or `blocked` — never None or blank
5. WHEN ML model underperforms all baselines (MAE worse than every baseline), THEN ML signal status SHALL be `diagnostic_only` and SHALL be excluded from score direction but SHALL NOT block a formal rating
6. WHEN NLP section extraction fails (char_count below minimum threshold for required sections), THEN NLP signal status SHALL be `diagnostic_only` and SHALL be excluded from score direction but SHALL NOT block a formal rating
7. THE RecommendationStatus SHALL be one of: `formal_rating` (all blocking gates pass), `diagnostic_not_rated` (one or more blocking gates fail), `failed` (pipeline error)

### Requirement 18: ML Integrity Gate

**User Story:** As a quantamental analyst, I want ML outputs honestly assessed so that a model with zero predictive power cannot inflate the recommendation.

#### Acceptance Criteria

1. WHEN all model coefficients are zero or near-zero (max |coef| < 0.001), THEN the ML component status SHALL be `diagnostic_only` with reason "degenerate model"
2. WHEN walk-forward MAE is worse than all four baselines, THEN the ML component status SHALL be `diagnostic_only` with reason "underperforms baselines"
3. WHEN annual-only data is used (fewer than 12 training observations), THEN model_audit.md SHALL state "ML is exploratory and underpowered due to limited annual observations"
4. THE model audit SHALL report: number of training observations, feature count, coefficient values, walk-forward MAE vs each baseline, directional accuracy, and an honest assessment of predictive value

### Requirement 19: Segment Quality Gate

**User Story:** As a quantamental analyst, I want segment charts to be meaningful — if normalization fails, the chart should be blocked rather than showing misleading data.

#### Acceptance Criteria

1. WHEN all normalized_category values for a fiscal year are "Other" or "Unclassified", THEN the segment revenue chart for that year SHALL be suppressed or flagged as "segment normalization failed"
2. WHEN segment revenue totals differ from reported total revenue by more than 5%, THEN a reconciliation warning SHALL appear in data_quality_report.md
3. THE segment chart SHALL include a footnote indicating whether it shows reportable segments or market/platform categories and which fiscal years use fallback extraction

### Requirement 20: NLP Quality Gate

**User Story:** As a quantamental analyst, I want NLP features excluded from scoring when the underlying text extraction is too poor to be meaningful.

#### Acceptance Criteria

1. WHEN section extraction char_count is below a minimum threshold (default: 500 characters for MD&A, 300 for Risk Factors), THEN NLP features for that section/filing SHALL be marked `diagnostic_only`
2. WHEN more than 50% of filings have failed or below-threshold text extraction for a required section, THEN the NLP component status SHALL be `diagnostic_only` for the entire analysis
3. NLP diagnostic_only status SHALL be reflected in the recommendation scorecard and limitations.md

### Requirement 21: Peer Quality Gate

**User Story:** As a quantamental analyst, I want peer multiples to be clean and meaningful, with NaN/negative rows filtered and peer tiers clearly separated.

#### Acceptance Criteria

1. Each peer row SHALL carry a PeerRowStatus: `usable`, `missing_ev`, `negative_ev`, `negative_ebitda`, `negative_earnings`, `stale_market_data`, `context_only`, or `excluded_from_primary_chart`
2. WHEN a peer row has NaN or negative enterprise_value, THEN that row SHALL be excluded from EV-based multiple calculations (EV/Revenue, EV/EBITDA) and logged with status
3. WHEN a peer row has negative net_income, THEN that row SHALL be excluded from primary P/E chart and logged; it MAY appear in an excluded/context table
4. Semi peers (AMD, AVGO, INTC, QCOM, MRVL) SHALL be separated from context peers (MSFT, AMZN, GOOGL, META) in all peer comparison tables
5. Infrastructure peers (TSM, ASML) SHALL be labeled as such in peer tables
6. WHEN fewer than 3 semi peers have valid EV multiples, THEN peer valuation SHALL be labeled "limited peer sample" in the report
7. WHEN peer market data source_date is older than peer_staleness_threshold_days, THEN that peer SHALL be excluded from primary valuation comparison and logged

### Requirement 22: Audit Consistency Gate

**User Story:** As a quantamental analyst, I want an automated audit that catches contradictions between output files so that the report never claims "no issues" when issues exist.

#### Acceptance Criteria

1. THE Audit SHALL fail IF `outputs/limitations.md` states "No data quality issues recorded" or equivalent while `outputs/data_quality_report.md` contains any validation failures
2. THE Audit SHALL fail IF the report cover page shows Buy/Hold/Sell while DataQualityStatus is DATA_BLOCKED
3. THE Audit SHALL fail IF `outputs/model_audit.md` claims model outperforms baselines while walk-forward results show otherwise
4. THE Engine SHALL produce `outputs/audit_status.json` with: overall_status (pass/fail), checks_run, checks_passed, checks_failed, failure_details[], timestamp
5. WHEN audit fails, THEN the pipeline SHALL log all failures and the report SHALL include an "Audit Warnings" section listing contradictions found

### Requirement 23: Full AI Prompt Log

**User Story:** As a grader evaluating source attribution (25% of rubric), I want a complete, structured log of every AI/LLM interaction used during the project so that I can assess how AI tools were used, what was generated versus human-authored, and how outputs were verified.

#### Acceptance Criteria

1. `outputs/prompt_log.md` SHALL contain a structured entry for every significant LLM interaction used during development and report generation, including: the exact prompt or a faithful summary, the model name and version, the purpose of the interaction, what output was used, and how the output was verified against primary sources
2. Each prompt log entry SHALL classify the interaction type: `code_generation`, `report_drafting`, `data_analysis`, `debugging`, `architecture_design`, or `verification`
3. The prompt log SHALL include the final repair/generation prompt in full or faithful summary, not just a one-line description
4. The prompt log SHALL disclose which report sections contain LLM-drafted prose, and for each, state the verification method (e.g., "verified against 10-K filing accession 0001045810-25-000023" or "analyst assumption — not independently verifiable")
5. The prompt log SHALL honestly disclose any gaps in logging coverage (e.g., "early development prompts were not logged with the same granularity")
6. The prompt log SHALL state the total number of significant LLM interactions and the approximate fraction that are fully logged versus summarized
7. WHEN an LLM interaction directly influenced a material claim in the report (e.g., a valuation assumption, a risk assessment, a recommendation rationale), THEN the prompt log entry SHALL cross-reference the specific report section and the primary source used for verification

### Requirement 24: Exact Data Retrieval Details

**User Story:** As a grader evaluating completeness (25% of rubric), I want exact documentation of how every data point was retrieved — including API endpoints, query parameters, retrieval timestamps, and caching behavior — so that the analysis is fully reproducible and auditable.

#### Acceptance Criteria

1. `outputs/source_attribution.md` SHALL include, for every SEC EDGAR data retrieval: the exact API endpoint URL, the CIK used, the retrieval timestamp, the HTTP status code, and the local cache path
2. `outputs/source_attribution.md` SHALL include, for every market data retrieval: the data provider (e.g., yfinance), the ticker(s) queried, the date range requested, the retrieval timestamp, and the number of rows returned
3. `outputs/source_attribution.md` SHALL include, for every peer financial retrieval: the data provider, the ticker, the specific fields retrieved, the source_date of the data, the retrieval timestamp, and the staleness assessment (days old relative to report_date)
4. `data/raw/provenance_log.jsonl` SHALL contain a machine-readable entry for every external data retrieval with fields: timestamp, step, source_url, http_status, cache_path, rows_returned, metadata (ticker, accession, date_range as applicable)
5. THE source attribution SHALL distinguish between data retrieved live versus loaded from cache, and for cached data, state when the cache was originally populated
6. THE source attribution SHALL include a "Data Retrieval Summary" table showing: data source, number of API calls, total rows retrieved, cache hit rate, and any retrieval failures with error details
7. WHEN a data retrieval fails or returns partial results, THEN the failure SHALL be logged in provenance_log.jsonl and disclosed in source_attribution.md with the error message and the fallback action taken

### Requirement 25: Point-in-Time Constituent Universe

**User Story:** As a grader evaluating novelty and completeness, I want the report to document the exact peer/constituent universe used at each point in the analysis timeline, including any changes to the peer set over the 10-year analysis window, so that survivorship bias and universe changes are transparent.

#### Acceptance Criteria

1. THE report SHALL include a "Peer Universe" section or table documenting: each peer ticker, the peer tier (core semiconductor, infrastructure, AI capex context), the justification for inclusion, and the date range for which the peer is included in the analysis
2. THE source attribution SHALL document whether the peer universe is fixed (same set for all 10 years) or time-varying, and if fixed, SHALL disclose the survivorship bias implications (e.g., "AMD was a much smaller company in FY2016 than today; using current peer status for historical comparison introduces look-ahead bias in peer selection")
3. WHEN a peer company underwent a material corporate event during the analysis window (e.g., merger, spin-off, delisting, ticker change), THEN the report SHALL note this event and its impact on comparability
4. THE peer universe documentation SHALL include the S&P 500 or relevant index membership status of each peer at the start and end of the analysis window, or explicitly state that index membership was not tracked
5. THE report SHALL include a "Constituent Universe Limitations" note explaining: (a) the peer set was selected based on current-day relevance, not historical membership, (b) this introduces potential survivorship bias, and (c) a fully point-in-time peer universe would require historical index constituent data that was not available in this analysis
6. FOR each peer used in valuation multiples, THE source attribution SHALL state: the peer ticker, the financial data date, the market data date, the staleness in days, and whether the peer was included or excluded from the primary comparison (with reason if excluded)
