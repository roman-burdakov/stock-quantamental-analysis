# Implementation Plan: NVDA Quantamental Engine

## Overview

Reproducible quantamental pipeline producing an A+ equity research report for Nvidia. All tasks reset to unchecked after post-mortem: the initial pipeline produced a catastrophically wrong Sell recommendation (−77.7% downside) because XBRL parsing returned FY2025 revenue as $26.974B instead of the published $130.497B. This plan adds hard validation gates, fiscal-year selection fixes, and recommendation eligibility controls.

When encountering ambiguity: make a documented assumption, continue, record in `outputs/limitations.md` and `outputs/model_audit.md`.

## Tasks

### Milestone 0: Capture Current Failure as Regression Fixtures

- [x] 0.1 Snapshot current broken outputs as regression baselines
  - Save current `outputs/nvda_quantamental_report.md` → `tests/fixtures/regression/broken_report_v1.md`
  - Save current `outputs/data_quality_report.md` → `tests/fixtures/regression/broken_dqr_v1.md`
  - Save current `outputs/limitations.md` → `tests/fixtures/regression/broken_limitations_v1.md`
  - Save current `data/processed/nvda_metrics.csv` → `tests/fixtures/regression/broken_metrics_v1.csv`
  - Document the failure: FY2025 revenue parsed as $26.974B vs published $130.497B
  - _Purpose: Ensure we can always reproduce and test against the original failure_

- [x] 0.2 Create published-value reference fixture
  - Create `tests/fixtures/nvda_published_values.json` with known 10-K values for FY2023, FY2024, FY2025
  - Include: revenue, gross_profit, operating_income, net_income, operating_cash_flow, capex, FCF, cash, total_debt, diluted_shares, diluted_EPS, R&D
  - Source each value to specific 10-K filing accession number and page
  - _Reqs: 14.2, 14.3, 15.7_

- [x] 0.3 Create XBRL regression test fixture
  - Extract the specific companyfacts JSON entries that caused the FY2025 revenue misparse
  - Save as `tests/fixtures/regression/fy2025_revenue_candidates.json`
  - Include quarterly, YTD, and annual candidate facts with their period metadata
  - _Purpose: Targeted regression test for fiscal-year selection_

### Milestone 1: Spec Alignment

- [x] 1.1 Rewrite requirements.md, design.md, tasks.md
  - Add data validation gate requirements (Req 14)
  - Add fiscal-year correctness requirements (Req 15)
  - Add valuation guardrails (Req 16)
  - Add recommendation eligibility gate (Req 17)
  - Add ML/NLP/segment/peer quality gates (Reqs 18-21)
  - Add audit consistency gate (Req 22)
  - Reset all tasks to unchecked
  - _This task — currently being executed_

### Milestone 2: Fiscal-Year Selection Fix

- [x] 2.1 Add period metadata to FinancialFact dataclass
  - Add fields: fiscal_period_type, period_start, period_end, duration_days, frame, xbrl_concept, selection_rank, selection_reason
  - Update `src/config.py` with enhanced FinancialFact
  - _Reqs: 15.1_

- [x] 2.2 Implement period classification in XBRL parser
  - Add `_classify_period_type()`: annual (350-380d), quarterly (<100d), ytd (100-340d), instant
  - Parse period_start and period_end from companyfacts JSON startDate/endDate
  - Compute duration_days
  - _Reqs: 15.2, 15.3, 15.4_

- [x] 2.3 Implement annual fact selection algorithm
  - Add `_select_annual_fact()` with ranking: 10-K annual-duration → frame-tagged → latest amendment
  - Reject quarterly and YTD facts for annual selection
  - Record selection_rank and selection_reason on every selected fact
  - Handle amendments deterministically (latest filing_date wins)
  - _Reqs: 15.2, 15.3, 15.4, 15.5, 15.6_

- [x] 2.4 Write fiscal-year selection regression tests
  - Test that FY2025 revenue parses as ~$130.5B (±1%), not $27B
  - Test that quarterly facts (<100d) are never selected as annual
  - Test that YTD facts (100-340d) are rejected for annual selection
  - Test amendment handling: latest amendment supersedes
  - Test frame-tagged preference
  - Use fixtures from 0.3
  - _Reqs: 15.7_

### Milestone 3: Published-Value Validation with Hard Blocking

- [x] 3.1 Implement ValidatedMetric and DataQualityStatus data models
  - Add to `src/config.py`: ValidatedMetric, DataQualityStatus, ComponentStatus, ComponentStatusEnum
  - Define core_validation_metrics list in EngineConfig
  - Set default tolerance to 1%
  - _Reqs: 14.1, 14.2_

- [x] 3.2 Implement `src/data_validation.py`
  - DataValidationGate class with validate_parsed_data()
  - Cross-check each core metric against published values
  - Compute diff_pct, assign severity (critical for core annual metrics)
  - Compute DataQualityStatus: PASS / PASS_WITH_WARNINGS / DATA_BLOCKED
  - check_fiscal_year_selection() to verify no quarterly/YTD used as annual
  - generate_validation_summary() for audit_status.json
  - _Reqs: 14.1-14.5, 14.9, 14.10_

- [x] 3.3 Update xbrl_parser.py to use DataValidationGate
  - validate_against_published() returns list[ValidatedMetric]
  - compute_data_quality_status() returns DataQualityStatus
  - generate_data_quality_report() includes validation table and status
  - _Reqs: 14.9, 3.6, 3.8_

- [x] 3.4 Write validation gate tests
  - Test PASS when all metrics within 1% tolerance
  - Test DATA_BLOCKED when revenue differs by >1%
  - Test PASS_WITH_WARNINGS for non-critical deviations
  - Test that FY2025 revenue=$27B vs published=$130.5B → DATA_BLOCKED
  - Test missing published value for valuation-critical base-year metric → DATA_BLOCKED unless manually sourced
  - Test missing capex reference → DATA_BLOCKED if FCF constructed from OCF minus capex
  - Test validation_coverage_pct below 90% for latest 3 FYs → DATA_BLOCKED
  - Test manually sourced validation reference unblocks only if source_attribution.md includes the source
  - _Reqs: 14.4, 14.5, 14.11, 14.12_

### Milestone 4: Split Adjustment

- [x] 4.1 Add split_history to EngineConfig
  - Define NVIDIA 10:1 split (June 10, 2024) in config
  - Store both raw_value and adjusted_value on FinancialFact (never overwrite raw)
  - Add adjustment_factor, adjustment_basis, validation_basis fields
  - Verify pre-split and post-split EPS are on consistent basis
  - _Reqs: 3.1, 3.2_

- [x] 4.2 Write split adjustment tests
  - Test that raw facts are preserved (raw_value never modified)
  - Test that adjusted facts are reproducible from raw + factor
  - Test that current price and diluted shares use compatible basis
  - Test that 10x share-count mismatch triggers DATA_BLOCKED
  - Test diluted_shares consistency across split boundary
  - _Reqs: 12.1_

### Milestone 5: Valuation Guardrails

- [x] 5.1 Implement valuation input validation
  - check_valuation_inputs() in ValuationModule: blocked if any valuation-critical input failed
  - Return ComponentStatus for valuation
  - _Reqs: 16.1_

- [x] 5.2 Implement sanity-check bridge explanations
  - generate_sanity_checks() in ValuationModule
  - Flag when DCF target differs from current price by >50%
  - Flag when reverse-DCF grid doesn't contain current price ±10%
  - Flag when bear downside exceeds −40%
  - Check sensitivity table spans current price
  - _Reqs: 16.2, 16.3, 16.4, 16.5_

- [x] 5.3 Write valuation guardrail tests
  - Test >50% divergence triggers bridge explanation
  - Test grid warning when current price outside range
  - Test bear downside >−40% flag
  - Test sensitivity table range check
  - _Reqs: 16.1-16.5_

### Milestone 6: Reverse-DCF Repair

- [x] 6.1 Fix reverse-DCF grid semantics
  - Use reasonable configured assumption ranges for CAGR and FCF margin
  - Highlight cells nearest to current price
  - WHEN current price is outside grid range, explicitly state that and use solver to compute implied CAGR/margin
  - Report whether implied assumptions are within, above, or far outside historical/plausible ranges
  - Do NOT extend grid to unreasonable assumptions just to include current price
  - _Reqs: 9.7, 16.3_

- [x] 6.2 Write reverse-DCF grid tests
  - Test grid uses configured reasonable ranges
  - Test explicit warning when current price outside grid
  - Test solver computes implied CAGR and implied margin for current price
  - Test report text explains plausibility of implied assumptions
  - Test one-variable solvers converge
  - _Reqs: 12.2_

### Milestone 7: Recommendation Eligibility Gate

- [x] 7.1 Implement RecommendationStatus and eligibility gate
  - Add RecommendationStatus dataclass to config.py
  - Implement eligibility check: all required components must be usable or explicitly excluded
  - Required: data validation PASS, market price validated, DCF assumptions documented, peer data clean/excluded, ML valid/excluded, NLP valid/excluded
  - _Reqs: 17.1, 17.2, 17.6_

- [x] 7.2 Update Recommendation dataclass with eligibility fields
  - Add eligibility_status, not_rated_reason, sanity_check_notes
  - Scorecard components show ComponentStatus (usable/diagnostic_only/unavailable/blocked), never None
  - _Reqs: 17.3_

- [x] 7.3 Update generate_recommendation() to respect eligibility
  - If RecommendationStatus is not formal_rating → rating = "Not Rated"
  - Include not_rated_reason explaining which components failed
  - Scorecard shows all component statuses
  - _Reqs: 17.1, 17.2, 17.3_

- [x] 7.4 Write recommendation eligibility tests
  - Test formal_rating when all blocking gates pass (even if ML is diagnostic_only)
  - Test diagnostic_not_rated when DATA_BLOCKED
  - Test that ML diagnostic_only is excluded from score direction but does NOT block formal rating when data validation and valuation gates pass
  - Test that NLP diagnostic_only is excluded from score direction but does NOT block formal rating
  - Test that scorecard never contains None status
  - Test that rating thresholds (Req 9.11) are only evaluated after eligibility gate passes
  - _Reqs: 17.1-17.7_

### Milestone 8: ML Audit Integration

- [x] 8.1 Implement ML quality assessment
  - assess_ml_quality() in MLDriverModel
  - Check: all coefficients near-zero → diagnostic_only ("degenerate model")
  - Check: MAE worse than all baselines → diagnostic_only ("underperforms baselines")
  - Check: annual-only data → add caveat to model_audit.md
  - Return ComponentStatus
  - _Reqs: 18.1, 18.2, 18.3, 18.4_

- [x] 8.2 Update model_audit.md generation
  - Include: training observations count, feature count, coefficient values
  - Include: walk-forward MAE vs each baseline
  - Include: honest assessment of predictive value
  - Include: annual-only caveat if applicable
  - _Reqs: 18.4, 8.9, 8.10_

- [x] 8.3 Write ML quality gate tests
  - Test degenerate model detection (all coefs near zero)
  - Test baseline underperformance detection
  - Test annual-only caveat generation
  - _Reqs: 18.1-18.4_

### Milestone 9: Segment Normalization Repair

- [x] 9.1 Implement segment quality check
  - check_segment_quality() in SegmentRevenueNormalizer → ChartStatus
  - Block chart if all categories are "Other"/"Unclassified"
  - Warn if segment totals differ from reported revenue by >5%
  - _Reqs: 19.1, 19.2_

- [x] 9.2 Add segment chart footnotes
  - Label whether chart shows reportable segments or market/platform categories
  - Note which fiscal years use fallback extraction
  - _Reqs: 19.3, 4.4_

- [x] 9.3 Write segment quality tests
  - Test chart blocked when all categories are "Other"
  - Test reconciliation warning when totals differ >5%
  - Test footnote generation
  - _Reqs: 19.1-19.3_

### Milestone 10: Filing Text and NLP Repair

- [x] 10.1 Implement text extraction quality check
  - check_extraction_quality() in FilingTextParser → ComponentStatus
  - diagnostic_only if >50% filings below char threshold
  - Thresholds: 500 chars for MD&A, 300 for Risk Factors
  - _Reqs: 20.1, 20.2_

- [x] 10.2 Implement NLP quality assessment
  - assess_nlp_quality() in NLPFeatureExtractor → ComponentStatus
  - diagnostic_only if extraction quality is poor
  - Reflect in scorecard and limitations.md
  - _Reqs: 20.1, 20.2, 20.3_

- [x] 10.3 Write NLP quality gate tests
  - Test diagnostic_only when char_count below threshold
  - Test diagnostic_only when >50% filings have poor extraction
  - Test that diagnostic status propagates to scorecard
  - _Reqs: 20.1-20.3_

### Milestone 11: Peer Multiples Cleanup

- [x] 11.1 Implement peer filtering and separation
  - filter_peer_multiples() in ValuationModule
  - Filter NaN/negative EV rows with logging
  - Separate semi peers from infrastructure peers from context peers
  - Label peer tiers in all tables
  - _Reqs: 21.1, 21.2, 21.3_

- [x] 11.2 Add limited peer sample warning
  - When fewer than 3 semi peers have valid EV multiples → label "limited peer sample"
  - _Reqs: 21.4_

- [x] 11.3 Write peer quality tests
  - Test NaN/negative EV row exclusion from EV-based charts
  - Test negative P/E exclusion from primary P/E chart
  - Test negative EBITDA exclusion from EV/EBITDA chart
  - Test stale peer data exclusion
  - Test peer tier separation (semi vs infrastructure vs context)
  - Test limited sample warning
  - Test PeerRowStatus assignment for each exclusion reason
  - _Reqs: 21.1-21.7_

### Milestone 12: Limitations and Audit Consistency

- [x] 12.1 Fix limitations.md generation
  - generate_limitations() MUST include validation failures when they exist
  - SHALL NOT output "No data quality issues recorded" when failures are present
  - Include each failure: metric_name, fiscal_year, parsed_value, published_value, diff_pct
  - _Reqs: 14.8, 22.1_

- [x] 12.2 Implement audit consistency checker
  - check_audit_consistency() in AuditModule
  - Check 1: limitations.md vs data_quality_report.md contradiction
  - Check 2: report rating vs DataQualityStatus contradiction
  - Check 3: model_audit claims vs walk-forward results
  - Return AuditStatus
  - _Reqs: 22.1, 22.2, 22.3_

- [x] 12.3 Implement audit_status.json generation
  - generate_audit_status_json() in AuditModule
  - Fields: overall_status, data_quality_status, recommendation_eligibility, component_statuses, checks_run/passed/failed, blocking_issues, failure_details, timestamp
  - _Reqs: 14.10, 22.4_

- [x] 12.4 Write audit consistency tests
  - Test fail when limitations says "no issues" but DQR has failures
  - Test fail when report shows Buy/Hold/Sell but DATA_BLOCKED
  - Test fail when model_audit claims outperformance but results show otherwise
  - Test pass when all files are consistent
  - Test audit_status.json schema and content
  - _Reqs: 22.1-22.5_

### Milestone 13: Report Generation Repair

- [x] 13.1 Add report mode support to templates
  - Update Jinja2 templates for three modes: formal_rating, diagnostic_not_rated, failed
  - Cover page: "Not Rated — Data Validation Required" when diagnostic_not_rated
  - Valuation sections: "Diagnostic only — do not use for recommendation" label
  - Scorecard: show ComponentStatus for each component
  - _Reqs: 14.6, 14.7, 17.2_

- [x] 13.2 Update report_utils.py for report modes
  - build_report_context() accepts RecommendationStatus
  - generate_markdown_report() renders based on ReportMode
  - generate_executive_summary() reflects report mode
  - Add "Audit Warnings" section when audit fails
  - _Reqs: 14.6, 14.7, 22.5_

- [x] 13.3 Write report mode tests
  - Test diagnostic_not_rated mode renders correct cover page
  - Test valuation diagnostic label appears
  - Test scorecard shows all component statuses
  - Test audit warnings section appears when audit fails
  - _Reqs: 14.6, 14.7, 17.2_

### Milestone 14: Chart Rules

- [x] 14.1 Implement chart gating
  - Segment chart: suppressed if ChartStatus.renderable is False, show fallback_message
  - DCF/reverse-DCF charts: include diagnostic label when valuation is diagnostic-only
  - Scorecard chart: show component statuses including blocked/diagnostic
  - Reverse-DCF grid chart: annotate when current price is outside grid
  - _Reqs: 19.1, 16.3, 17.3_

- [x] 14.2 Write chart gating tests
  - Test segment chart suppression
  - Test diagnostic label on DCF charts
  - Test scorecard renders all statuses
  - _Reqs: 19.1, 16.3_

### Milestone 15: End-to-End Acceptance Tests

- [x] 15.1 Write regression test: FY2025 revenue misparse
  - Load companyfacts fixture, run parser, verify FY2025 revenue ≈ $130.5B
  - Verify that $27B value is classified as quarterly/YTD and rejected
  - _Reqs: 15.7_

- [x] 15.2 Write regression test: DATA_BLOCKED prevents formal rating
  - Inject bad revenue value, run validation gate
  - Verify DataQualityStatus = DATA_BLOCKED
  - Verify recommendation = "Not Rated"
  - Verify valuation labeled "Diagnostic only"
  - _Reqs: 14.5, 14.6, 14.7_

- [x] 15.3 Write regression test: limitations.md consistency
  - Run pipeline with validation failures
  - Verify limitations.md lists failures (not "no issues")
  - Verify audit consistency check passes
  - _Reqs: 14.8, 22.1_

- [x] 15.4 Write end-to-end pipeline test
  - Run full pipeline with cached data
  - Verify all required outputs exist
  - Verify audit_status.json is valid
  - Verify report mode matches data quality status
  - _Reqs: 13.1, 14.10_

- [x] 15.5 Write test: clean data produces formal rating
  - Use fixtures with correct published values
  - Verify DataQualityStatus = PASS
  - Verify recommendation is Buy/Hold/Sell (not "Not Rated")
  - Verify scorecard has no blocked components
  - _Reqs: 17.1, 17.6_

### Milestone 16: Documentation and Final Package

- [x] 16.1 Update README.md
  - Document data validation gate and its purpose
  - Document report modes (formal_rating, diagnostic_not_rated, failed)
  - Document audit_status.json
  - Update one-command run instructions
  - _Reqs: 13.4_

- [x] 16.2 Update notebooks 01-05
  - Add validation gate walkthrough to notebook 02
  - Add recommendation eligibility discussion to notebook 05
  - Ensure notebooks reflect new pipeline stages
  - _Reqs: 13.3_

- [x] 16.3 Verify all required outputs
  - `data/processed/nvda_metrics.csv`
  - `data/processed/nvda_segment_revenue_normalized.csv`
  - `data/processed/nvda_nlp_features.csv`
  - `data/processed/ml_feature_target_matrix.csv`
  - `outputs/nvda_quantamental_report.md`
  - `outputs/nvda_quantamental_report.pdf` OR `outputs/nvda_quantamental_report.html`
  - `outputs/executive_summary.md`
  - `outputs/source_attribution.md`
  - `outputs/prompt_log.md`
  - `outputs/data_dictionary.md`
  - `outputs/data_quality_report.md`
  - `outputs/limitations.md`
  - `outputs/model_audit.md`
  - `outputs/audit_status.json` ← NEW
  - `outputs/figures/*`
  - _Reqs: 10.1, 10.2, 14.10_

- [x] 16.4 Run full pipeline and verify final package status
  - Execute: `python scripts/run_pipeline.py --ticker NVDA --report-date 2026-05-01 --price-date 2026-05-01`
  - Verify audit_status.json shows one of: `formal_rating_pass`, `diagnostic_not_rated_pass`, or `failed`
  - No known contradictions may remain in a pass state
  - Tests must fail if:
    - report shows Buy/Hold/Sell while DATA_BLOCKED
    - limitations says no issues while validation failures exist
    - valuation uses fallback/YTD/quarterly annual values for core metrics
    - ML diagnostic_only is presented as supporting the rating direction
    - NLP diagnostic_only is presented as supporting the rating direction
    - peer chart includes invalid rows without exclusion labels
    - report mode and audit_status.json disagree
  - Verify all tests pass
  - _Reqs: 13.1, 22.4_

### Milestone 17: Optional Enhancements (Stretch)

- [x] 17.1* Sentence-transformer embedding distances
- [x] 17.2* BERTopic/LDA topic modeling
- [x] 17.3* Change-point detection with ruptures
- [x] 17.4* Tree model (RF/GBT) with SHAP
- [x] 17.5* Semiconductor peer-panel ML (requires 10-year peer historicals)
- [x] 17.6* Additional charts beyond required 6-8
- [x] 17.7* Highly polished PDF styling
- [x] 17.8* Multi-ticker support validation (run on AMD to test scalability)

### Milestone 18: Full AI Prompt Log (Req 23)

- [x] 18.1 Restructure `outputs/prompt_log.md` with structured entries
  - Add a structured entry for every significant LLM interaction: exact prompt or faithful summary, model name/version, purpose, output used, verification method
  - Classify each entry by type: `code_generation`, `report_drafting`, `data_analysis`, `debugging`, `architecture_design`, `verification`
  - Include the final repair/generation prompt in full or faithful summary
  - _Reqs: 23.1, 23.2, 23.3_

- [x] 18.2 Add LLM-drafted section disclosure
  - For each report section that contains LLM-drafted prose, state the verification method (e.g., "verified against 10-K filing accession X" or "analyst assumption")
  - Cross-reference prompt log entries to specific report sections when the interaction influenced a material claim
  - _Reqs: 23.4, 23.7_

- [x] 18.3 Add prompt log completeness disclosure
  - State total number of significant LLM interactions and approximate fraction fully logged versus summarized
  - Honestly disclose gaps in logging coverage (e.g., early development prompts not logged with same granularity)
  - _Reqs: 23.5, 23.6_

- [x] 18.4 Write prompt log acceptance tests
  - Test that prompt_log.md contains at least 5 structured entries
  - Test that each entry has model name, purpose, and verification fields
  - Test that the final repair prompt is included
  - Test that incompleteness is disclosed
  - _Reqs: 23.1–23.7_

### Milestone 19: Exact Data Retrieval Details (Req 24)

- [x] 19.1 Enhance provenance_log.jsonl with retrieval details
  - Add fields to every provenance entry: http_status, cache_path, rows_returned, cache_hit (bool), retrieval_duration_ms
  - For SEC EDGAR calls: include exact API endpoint URL and CIK
  - For market data calls: include ticker(s), date range, provider name
  - For peer financial calls: include ticker, fields retrieved, source_date, staleness_days
  - _Reqs: 24.1, 24.2, 24.3, 24.4_

- [x] 19.2 Add "Data Retrieval Summary" to source_attribution.md
  - Generate a summary table: data source, number of API calls, total rows retrieved, cache hit rate, retrieval failures
  - Distinguish live retrieval versus cache load for each data source
  - For cached data, state when the cache was originally populated
  - _Reqs: 24.5, 24.6_

- [x] 19.3 Add retrieval failure logging
  - When a data retrieval fails or returns partial results, log the failure in provenance_log.jsonl with error message and fallback action
  - Disclose retrieval failures in source_attribution.md
  - _Reqs: 24.7_

- [x] 19.4 Write data retrieval detail tests
  - Test that provenance_log.jsonl entries contain required fields (timestamp, step, source_url, http_status, cache_path)
  - Test that source_attribution.md contains a Data Retrieval Summary table
  - Test that cache hits are distinguished from live retrievals
  - _Reqs: 24.1–24.7_

### Milestone 20: Point-in-Time Constituent Universe (Req 25)

- [x] 20.1 Add "Peer Universe" section to report
  - Document each peer ticker, tier, justification, and date range of inclusion
  - State whether the peer universe is fixed or time-varying
  - Disclose survivorship bias: peer set selected based on current-day relevance, not historical membership
  - _Reqs: 25.1, 25.2_

- [x] 20.2 Add peer corporate event notes
  - For each peer that underwent a material corporate event during the 10-year window (merger, spin-off, ticker change), note the event and its impact on comparability
  - Document index membership status (S&P 500 or relevant index) at start and end of analysis window, or state that index membership was not tracked
  - _Reqs: 25.3, 25.4_

- [x] 20.3 Add "Constituent Universe Limitations" note
  - Explain that the peer set was selected based on current-day relevance, not historical membership
  - Disclose that this introduces potential survivorship bias
  - State that a fully point-in-time peer universe would require historical index constituent data not available in this analysis
  - _Reqs: 25.5_

- [x] 20.4 Enhance peer source attribution
  - For each peer used in valuation multiples, document: ticker, financial data date, market data date, staleness in days, inclusion/exclusion status with reason
  - _Reqs: 25.6_

- [x] 20.5 Write constituent universe tests
  - Test that the report contains a Peer Universe section or table
  - Test that survivorship bias disclosure is present
  - Test that source_attribution.md includes per-peer staleness and inclusion status
  - _Reqs: 25.1–25.6_

## Notes

- ALL tasks in Milestones 0-16 are mandatory
- Milestones 18-20 address grader feedback: full AI prompt log, exact data retrieval details, and point-in-time constituent universe
- Milestone 17 sub-tasks are optional stretch goals (marked *)
- Unit tests run offline using `tests/fixtures/`
- Integration tests may require populated `data/raw/`
- When encountering ambiguity: document assumption, continue, record in limitations.md and model_audit.md
- Pipeline produces working report at Milestone 13 completion
- source_available_date is the universal availability gate across all data
- The three pipeline gates (data validation, recommendation eligibility, audit consistency) are the most critical additions — they prevent the catastrophic failure mode that produced the wrong Sell recommendation
