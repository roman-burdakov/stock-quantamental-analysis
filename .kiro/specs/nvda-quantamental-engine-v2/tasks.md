# Iteration 2 Tasks — Decision-Driving ML/NLP Layer (Revised)

Each task lists the requirement(s) it satisfies, an MVP tag, and a concrete acceptance check. Tasks are ordered by dependency. The implementation must commit `MLConfig` to git **before** running walk-forward on the latest data (Req 12).

## MVP Tier Tags (per `mvp_path.md`)

- **[MVP-E]** = MVP-Essential — must complete; if any [MVP-E] task fails final validation, the v2 stage is skipped and v1 ships unchanged
- **[MVP-X]** = MVP-Extended — target if MVP-E succeeds; failure here degrades to MVP-E
- **[OPT]** = Optional / Out-of-MVP — defer to v3 if time-constrained

The default `--ml-v2-mode` is `essential` until MVP-E is validated end-to-end. See `mvp_path.md` for the decision tree and fallback behavior.

---

## Milestone 0: MLConfig Pre-Registration (DO FIRST) — [MVP-E]

- [x] **0.1** [MVP-E] Create `MLConfig` dataclass in `src/config.py`
  - Fields: `feature_groups: dict[str, list[str]]`, `model_hyperparameters: dict`, `walk_forward: dict[str, dict]`, `tier_thresholds: dict`, `adjustment_weights: dict`, `bootstrap_n: int = 1000`, `random_seed: int = 42`
  - `feature_groups` keys: `"fundamentals"`, `"market"`, `"nlp"`, `"full_no_analyst"` — these are the only ML feature groups. Analyst data is **never** an ML feature. The analyst overlay payload (used by `compare_to_analyst_overlay`) is stored separately in `MLConfig.analyst_overlay_fields` (a list of field names extracted from `AnalystSnapshot` for the report comparator), NOT in `feature_groups`.
  - `model_hyperparameters["elasticnet"]`: `{alpha: 0.1, l1_ratio: 0.5, max_iter: 10000}`
  - `walk_forward["rev_growth_quarterly_YoY"]`: `{min_train: 12, embargo: 1, target_horizon: 1}`
  - `walk_forward["rev_growth_annual"]`: `{min_train: 16, embargo: 4, target_horizon: 4}`
  - `walk_forward["excess_return_direction"]`: `{min_train: 16, embargo: 4, target_horizon: 4}`
  - `tier_thresholds`: contributing/high-confidence numeric cutoffs (Req 7)
  - `adjustment_weights`: `{diagnostic: 0.0, contributing: 0.20, high_confidence: 0.35}`
  - **Acceptance:** Tests instantiate `MLConfig` with default values; values match Req 7 and Req 8.1 exactly
  - **Reqs:** 12.1, 12.2

- [x] **0.2** Add `MLConfig` git commit gate to `run_pipeline.py`
  - At ml_v2 stage entry, log the git SHA of `src/config.py` and the commit timestamp
  - Write to `outputs/ml_config_provenance.json`
  - **Acceptance:** Output file contains SHA and timestamp; populated even if pipeline fails downstream
  - **Reqs:** 12.5

---

## Milestone 1: Data Foundation Refresh

- [x] **1.1** Add `^SOX` and `^GSPC` to `EdgarFetcher.fetch_market_prices()`
  - Update default ticker list in `EngineConfig` to include `["^SOX", "^GSPC"]`
  - Verify yfinance returns prices for these tickers from 2014-01-01 onward
  - **Acceptance:** `data/raw/market_prices.csv` contains rows with `ticker ∈ {^SOX, ^GSPC}`; row count for these tickers ≥ 2,500 each
  - **Reqs:** 4.1

- [x] **1.2** [MVP-E] Implement `src/peer_freshness.py` with **two gates**: `MarketPriceFreshnessGate` (for ML) and `PeerFinancialsFreshnessGate` (for valuation)
  - `MarketPriceFreshnessGate.evaluate_and_refresh()`: trigger market_prices refresh if any required ticker (NVDA, ^SOX, ^GSPC, peers) is more than 5 trading days behind report_date; block ML stage entirely if NVDA is stale
  - `PeerFinancialsFreshnessGate.evaluate_and_refresh()`: trigger peer_financials refresh if max age > 30 days; per-peer exclusion still > 30 days; block valuation peer multiples if ≥3/5 core peers excluded
  - Write reports to `outputs/market_price_freshness_report.json` and `outputs/peer_financials_freshness_report.json`
  - **Acceptance:** Unit tests cover both gates with synthetic stale and fresh data; market gate blocks NVDA-stale case; peer-financials gate independent of price freshness
  - **Reqs:** 5.1–5.10

- [x] **1.3** [MVP-X] Implement `src/analyst_estimates.py` with `AnalystEstimateFetcher` — **decision-time overlay only, NOT an ML feature**
  - `fetch_snapshot(ticker, retrieval_date)` returns `AnalystSnapshot`
  - `render_overlay_summary(snapshot, ml_prediction)` returns dict with comparator narrative (analyst growth, ml growth, delta_pp, direction, magnitude)
  - **No `to_features()` method.** Analyst data is never converted to ML features and never enters the training matrix.
  - Cache to `data/raw/analyst_estimates_<YYYYMMDD>.json`
  - Graceful failure: `fetch_succeeded=False` if yfinance returns empty/error; pipeline continues without overlay
  - **Acceptance:** Unit test asserts `AnalystEstimateFetcher` does not expose any `to_features` method; snapshot is consumed only by `render_overlay_summary` for the report comparator
  - **Reqs:** 2.1, 2.2, 2.3, 2.4, 2.7

- [x] **1.4** Tests for Milestone 1
  - `tests/test_peer_freshness.py`: stale detection, refresh trigger, exclusion logic, blocked-status logic
  - `tests/test_analyst_estimates.py`: schema parsing, missing-field handling, cache file format
  - **Acceptance:** All tests pass; `pytest tests/test_peer_freshness.py tests/test_analyst_estimates.py` exits 0
  - **Reqs:** 2.6, 5.6

---

## Milestone 2: NLP Coverage Recovery

- [x] **2.1** Identify and re-fetch empty filing stubs
  - Scan `data/interim/sections_*.json` for files < 1KB (currently 30 of 46)
  - For each stub, check `data/raw/filings/<accession>.html` exists and is well-formed (>100KB and contains `<body>`)
  - Re-fetch HTML via `EdgarFetcher.fetch_filing_document()` if missing or malformed
  - **Acceptance:** After re-fetch, ≤5 of 46 cached HTMLs are still problematic (logged as `fetch_failed_persistent`)
  - **Reqs:** 3.1, 3.2

- [x] **2.2** Improve `src/filing_text_parser.py` for 10-Q
  - Add 10-Q-specific section labels: `"Part I, Item 2"` (MD&A) and `"Part II, Item 1A"` (Risk Factors)
  - Add fallback method `_extract_mda_proxy(html)`: clean HTML, return first 10,000 chars of body text
  - Track quality tier: `full_extraction` | `partial_extraction` | `mda_proxy_fallback` | `failed`
  - Write quality tier to each `sections_*.json` as a top-level field
  - **Acceptance:** After re-running on the 46 cached filings, ≥70% have tier ≥ `mda_proxy_fallback`
  - **Reqs:** 3.2, 3.3

- [x] **2.3** Add sentiment feature to `src/nlp_features.py`
  - Use `textblob.TextBlob(text).sentiment.polarity` (range −1 to +1)
  - For each filing's MD&A: compute polarity; for prior filing: compute polarity; delta = current − prior
  - Add columns `sentiment_polarity`, `sentiment_delta` to NLP features output
  - **Acceptance:** `nvda_nlp_features.csv` gains the two columns; values in [−1, 1] for non-null rows
  - **Reqs:** 3.4

- [x] **2.4** Tests for Milestone 2
  - `tests/test_nlp_coverage.py`: assert ≥70% extraction tier coverage on cached filings; assert sentiment in valid range
  - `tests/test_filing_text_parser.py`: per-form-type extraction; fallback proxy logic
  - **Reqs:** 3.3, 3.4

---

## Milestone 3: Market Feature Builder

- [ ] **3.1** Implement `src/market_features.py` with `MarketFeatureBuilder`
  - `compute_features(prices, as_of_dates, excluded_peers)` returns DataFrame
  - All features computed only from prices with `date ≤ as_of_date`
  - Columns: trailing returns (3m, 12m), excess returns vs SOX/SPX, volatility, beta, peer-spread returns
  - **Acceptance:** Sample run on 2024-Q4 cutoff produces ~10 columns with no NaNs (except for excluded peers)
  - **Reqs:** 4.2, 4.3, 4.5

- [ ] **3.2** No-lookahead invariant test
  - `tests/test_market_features.py::test_no_lookahead_invariant`
  - Pseudocode:
    ```
    feats_a = builder.compute_features(prices_full, [as_of])
    prices_truncated = prices_full[prices_full.date <= as_of]
    feats_b = builder.compute_features(prices_truncated, [as_of])
    assert feats_a.equals(feats_b)
    ```
  - **Acceptance:** Test passes for 5 different as_of_dates spanning 2018–2025
  - **Reqs:** 4.3

---

## Milestone 4: Quarterly Panel Builder

- [ ] **4.1** Implement `src/quarterly_panel.py::QuarterlyPanelBuilder`
  - `_select_quarterly_skeleton`: filter `metrics` to rows with `fiscal_period` matching `FY\d{4}-Q[1-4]` or `FY\d{4}` (Q4 alias); pivot to one row per quarter
  - `_derive_quarterly_from_ytd`: implement YTD difference rule (Req 1.4)
  - `_attach_fundamental_features`: join YTD-derived + instant + direct quarterly metrics
  - `_attach_nlp_features`: left-join by `feature_available_date` ↔ `filing_date`; **no LOCF**
  - `_attach_market_features`: left-join by `feature_available_date` ↔ `as_of_date`
  - `_attach_targets`: build primary, secondary, tertiary targets per Req 6
  - **Acceptance:** Output panel has ≥35 rows with non-null primary target; ≥30 rows for secondary; column count 20–30
  - **Reqs:** 1.1–1.6, 6.1–6.3

- [ ] **4.2** Mixed-frequency YTD validation
  - Add `_validate_quarterly_sums_to_annual()`: for each FY, sum derived Q1+Q2+Q3+Q4 capex/OCF; compare to annual XBRL value; ≥1% mismatch sets `feature_imputed_<col>=True`
  - Log validation summary to `data/processed/quarterly_panel_validation.json`
  - **Acceptance:** Validation runs without error; ≥80% of FY×metric pairs have <1% sum-mismatch
  - **Reqs:** 1.4, 1.6

- [ ] **4.3** No-lookahead validation for panel
  - For every row: `feature_available_date < target_available_date`
  - Market features were computed with `as_of_date = feature_available_date` only
  - NLP features were extracted from filings with `filing_date ≤ feature_available_date`
  - **Acceptance:** Existing `validate_no_lookahead_matrix()` (extended) returns True for entire panel
  - **Reqs:** 1.2, 6.4

- [ ] **4.4** Tests for Milestone 4
  - `tests/test_quarterly_panel.py`: row count, non-null targets, YTD-derivation correctness, no-lookahead
  - **Acceptance:** All tests pass
  - **Reqs:** 1.7, 6.4

---

## Milestone 5: ML Decision Engine

- [ ] **5.1** Implement walk-forward in `src/ml_decision.py::MLDecisionEngine.walk_forward`
  - Horizon-aware embargo: per Req 6.4 and Milestone 0.1 config
  - For each fold: train on rows with `target_available_date ≤ split_date`; test on row at split_date
  - Skip test row if its `feature_available_date ≤ max(train.target_available_date)` (lookahead protection)
  - Return `WalkForwardResult` with N_OOS, R², MAE, directional accuracy, OOS predictions DataFrame
  - **Acceptance:** Annual-target WF on real panel produces ≥10 OOS predictions; quarterly-YoY WF produces ≥25 OOS predictions
  - **Reqs:** 6.4, 6.5, 6.6

- [ ] **5.2** Implement naive baselines
  - `naive_baseline_directional_accuracy(panel, target, baseline_type)`
  - Baselines: persistence, seasonal-naive (same quarter last year), 3y-trailing-mean, always-positive, coin-flip
  - **Analyst consensus is NOT a naive baseline.** Historical analyst-consensus time series is unavailable from public sources (yfinance only provides current snapshot). Analyst consensus is reported as a current-period overlay comparator per Req 2 and ablation group E (Req 10.3); it is never used as a walk-forward naive baseline.
  - **Acceptance:** Each baseline produces a numeric directional accuracy and N; smoke test on synthetic data
  - **Reqs:** 6.7

- [ ] **5.3** [MVP-E] Implement McNemar's exact significance test
  - `mcnemar_exact(model_correct: np.ndarray, baseline_correct: np.ndarray) -> dict`
  - Returns `{n_concordant, n_discordant_a, n_discordant_c, p_value}`
  - Uses `scipy.stats.binomtest(a, a+c, p=0.5, alternative="greater")` on discordant pairs (paired test)
  - **Acceptance:** Unit tests cover (a) model strictly better than baseline (a > c, p < 0.5), (b) tie (a = c, p = 1.0), (c) baseline better (a < c, p = 1.0); reject any commit that uses aggregate-binomial-on-counts
  - **Reqs:** 7.1, 7.2, 7.3, 7.6

- [ ] **5.4** Implement confidence-tier assignment
  - `assign_confidence_tier(wf_result, ml_config)` per Req 7
  - **Acceptance:** Unit test cases cover all three tiers and edge thresholds (e.g., n_oos=24 → diagnostic; n_oos=25 + acc=55% + p=0.19 → contributing)
  - **Reqs:** 7.1, 7.2, 7.3, 7.4

- [ ] **5.5** Implement ablation
  - `compute_ablation(panel, target="rev_growth_annual")` runs walk_forward for 4 feature subsets (A, B, C, D from Req 10.1)
  - Returns DataFrame with columns: `group`, `target`, `n_oos`, `directional_accuracy`, `mae`, `mae_improvement_vs_naive_pct`, `r2`, `mcnemar_p_vs_naive`
  - **Acceptance:** Output has 4 rows; saved to `data/processed/ml_ablation_results.csv`
  - **Reqs:** 10.1, 10.2, 10.4, 10.5

- [ ] **5.6** [MVP-E] Implement empirical residual band
  - `bootstrap_residual_band(wf_result, dcf_callable, dcf_assumed_growth, n_bootstrap=1000)` returns (5th, 50th, 95th) percentile target prices
  - Save bootstrap target prices to `data/processed/ml_residual_band_bootstrap.csv`
  - Use `random_seed=42` from MLConfig for reproducibility
  - **Acceptance:** Two consecutive runs produce identical band; band low ≤ median ≤ band high; output is labeled "empirical residual band" in all artifacts (never "confidence interval" or "CI")
  - **Reqs:** 9.1, 9.1a, 9.2, 9.3, 9.4

- [ ] **5.7** [MVP-X] Implement analyst overlay comparator (NOT a learned-feature prediction)
  - `MLDecisionEngine.compare_to_analyst_overlay(ml_prediction, analyst_snapshot)`:
    - Inputs: ML walk-forward prediction (from full panel, no analyst features) and `AnalystSnapshot`
    - Computes `analyst_revenue_growth_next_yr` from snapshot
    - Returns `OverlayComparator(ml_growth, analyst_growth, delta_pp, direction, agreement_label)`
  - **Critical:** No ElasticNet training on `panel + analyst features`. The ML model uses only historically-available features. Analyst data is a side-by-side comparator for the report narrative.
  - **Acceptance:** Unit test asserts that with `analyst_snapshot=None`, `compare_to_analyst_overlay` returns `None` and the ML prediction passes through unchanged. Asserts that no model is fit using analyst values.
  - **Reqs:** 2.4, 2.5, 2.6

- [ ] **5.8** Implement `compute_ml_adjustment` and `evaluate_success_criteria`
  - `compute_ml_adjustment` per Req 8.1
  - `evaluate_success_criteria` checks Req 11.1–11.4
  - Save adjustment object to `data/processed/ml_adjustment.json`
  - **Acceptance:** Unit test with mocked WF (acc=65%, p=0.05, n=30) → tier=high_confidence, weight=0.35; with mocked WF (acc=51%, p=0.4, n=30) → tier=diagnostic, weight=0
  - **Reqs:** 8.1, 8.2, 11.1–11.4

- [ ] **5.9** Tests for Milestone 5
  - `tests/test_ml_decision.py`: walk-forward correctness, embargo enforcement, no-lookahead, tier logic, bootstrap reproducibility, ablation row count, success-criteria evaluation
  - **Acceptance:** All tests pass; pytest exits 0
  - **Reqs:** 6.4, 7, 8, 9, 10, 11

---

## Milestone 6: Pipeline + Valuation Integration

- [ ] **6.1** Add `ml_v2` stage to `scripts/run_pipeline.py`
  - Insert between existing `ml` and `valuation` stages
  - Register as new value of `--step` flag
  - Orchestrate: peer freshness → analyst fetch → market features → quarterly panel → ML decision engine
  - Save `MLV2Result` to state for valuation stage
  - **Acceptance:** `python scripts/run_pipeline.py --step ml_v2` runs end-to-end without errors when run after a successful `ml` stage
  - **Reqs:** All

- [ ] **6.2** Update `src/valuation.py::build_final_recommendation`
  - Add optional kwarg `ml_adjustment: MLAdjustment | None = None`
  - When provided: use `ml_adjustment.adjusted_target` as target_price; add scorecard row "ML/NLP Signal" with weight per Req 8.3
  - When None: existing v1 behavior, full backward compat
  - **Acceptance:** Existing tests for `build_final_recommendation` pass without modification
  - **Reqs:** 8.4, 8.5, 13.3

- [ ] **6.3** Update `src/audit_utils.py`
  - Add new audit checks per design "Audit Integration" section
  - Update `audit_status.json` schema with all `ml_v2_*` fields
  - **Acceptance:** `audit_status.json` after a full run contains all listed ml_v2 fields with correct types
  - **Reqs:** 5.6, 8, 11

---

## Milestone 7: Report & Charts

- [ ] **7.1** Add ablation chart to `src/charts.py`
  - `chart_ml_ablation(ablation_df, out_path)`: grouped bar chart of directional accuracy by feature group
  - Annotate marginal-lift values
  - **Acceptance:** PNG produced at `outputs/figures/ml_ablation.png`; legible at 300 DPI
  - **Reqs:** 10.5, 13.6

- [ ] **7.2** Add target-price distribution chart
  - `chart_ml_target_price_distribution(bootstrap_csv, dcf_only_target, adjusted_target, out_path)`: histogram with vertical lines for DCF-only and ML-adjusted targets
  - **Acceptance:** PNG produced at `outputs/figures/ml_target_price_distribution.png`
  - **Reqs:** 9.6, 13.6

- [ ] **7.3** Add walk-forward time-series chart
  - `chart_ml_walk_forward(predictions_csv, out_path)`: predicted vs actual over time with shaded prediction interval
  - **Acceptance:** PNG produced at `outputs/figures/ml_walk_forward_timeseries.png`
  - **Reqs:** 13.5

- [ ] **7.4** Update `src/templates/report_base.md.j2`
  - New section "Quantitative Signal Integration" between ML Audit and Valuation
  - Include walk-forward results table, ablation table, significance test results, success criteria check, target-price-distribution exhibit
  - Update "Limitations" section to include analyst-data historical limitation and peer staleness disclosures
  - **Acceptance:** Rendered report contains the new section with all populated content
  - **Reqs:** 13.2, 13.4, 13.5, 13.6

- [ ] **7.5** Update `src/templates/executive_summary.md.j2`
  - First page shows: "DCF-only target: $X | ML-adjusted target: $Y (or 'no adjustment' if fallback) | Empirical residual band (5th–95th percentile): $A–$B | Confidence tier: T | n_residuals: N"
  - **Acceptance:** Rendered executive summary shows all four numbers; values match audit_status.json
  - **Reqs:** 13.1

---

## Milestone 8: Validation & Honest Reporting

- [ ] **8.1** Run full pipeline end-to-end
  - `python scripts/run_pipeline.py --step all --report-date 2026-05-14 --price-date 2026-05-14`
  - Verify all output files exist (per design "File Outputs" table)
  - **Acceptance:** Pipeline exits 0; `audit_status.json.overall_status == "pass"` (or `pass_with_warnings`)
  - **Reqs:** all

- [ ] **8.2** Run existing v1 test suite — no regressions
  - `pytest tests/` (excluding new v2 test files initially, then including them)
  - **Acceptance:** All v1 tests still pass; new v2 tests pass
  - **Reqs:** 12 (no changes to v1 behavior)

- [ ] **8.3** [MVP-E] Verify hierarchical success-criteria outcome is reported honestly per Req 11
  - **Primary criterion** (Req 11.1, mvp_path.md is the source of truth on labels):
    - If the empirical residual band **narrows** by ≥10% AND McNemar p<0.10 AND MAE improvement ≥10%: outcome = `confidence-improving`. Report headlines: "ML is decision-driving on the primary axis (confidence-improving)."
    - If the empirical residual band **widens** by ≥10%: outcome = `risk-revealing`. Report says: "ML reveals additional forecast uncertainty above the DCF scenario sensitivity range. This is decision-useful but **does not** meet the pre-registered primary success criterion."
    - If band-width change is within ±10%: outcome = `neutral`. Report says: "ML and DCF concur within statistical noise."
  - **Secondary criterion** (Req 11.2): if annual N_OOS ≥ 25 AND tier ≥ contributing AND target-price |Δ| ≥ 3% — report adds "Target price adjusted by N% based on annual ML signal"
  - **Tertiary findings** (Req 11.3): ablation marginal lifts reported with Bonferroni caveat
  - If primary is `risk-revealing` or `neutral`: the report MUST NOT use the phrase "decision-driving" anywhere; it MAY use "decision-useful" only for `risk-revealing`
  - **Acceptance:** Report text matches `audit_status.json.ml_v2_band_outcome` exactly. The string "decision-driving" appears in the report if and only if `ml_v2_band_outcome == "confidence_improving"`. The string "decision-useful" appears if and only if `ml_v2_band_outcome == "risk_revealing"`. Neither appears for `neutral`. The consistency test (Task 8.5) enforces these conditional invariants.
  - **Reqs:** 11.1–11.5

- [ ] **8.4** [MVP-E] Manual review of report draft
  - Verify executive summary numbers match underlying data
  - Verify confidence-tier explanation is human-readable
  - Verify ablation table is interpretable
  - Verify the band outcome label (confidence-improving / risk-revealing / neutral) is used consistently
  - **Acceptance:** Report passes manual review with no factual contradictions
  - **Reqs:** 13

- [x] **8.5** [MVP-E] Cross-file consistency test (NEW — guards against terminology drift)
  - Create `tests/test_v2_spec_consistency.py` and `scripts/check_v2_consistency.sh`
  - The shell script greps the spec directory and the implemented source code for stale terms that should never appear in v2 outputs:
    - Banned in implementation code and report templates: `binomial_p_vs_naive`, `bootstrap_ci_low`, `bootstrap_ci_high`, `bootstrap_target_price_ci`, `decision_with_analyst`, `to_features` (in any class derived from analyst), `90% interval`, `90% confidence interval`, `confidence interval` (when applied to bootstrap output), `confidence-band` (use `empirical residual band` instead), `analyst_consensus_baseline` / `analyst_baseline` (analyst data is overlay only, never a walk-forward baseline)
    - Allowed only in negative assertions, comparative prose, or historical commentary in spec docs
  - The pytest test `test_v2_spec_consistency.py` enforces:
    - No stale terms in `src/ml_decision.py`, `src/analyst_estimates.py`, `src/peer_freshness.py`, `src/quarterly_panel.py`, `src/market_features.py`
    - No stale terms in `src/templates/*.j2` rendered output (run a smoke render and grep)
    - No stale terms in `outputs/audit_status.json` keys
    - All current valid terms are present where required: `mcnemar_p_vs_naive` field exists, `residual_band_low`/`high` fields exist, `analyst_overlay_fields` (not `decision_with_analyst`) is the config key
  - **Band-outcome conditional invariants** (NEW per Lo audit #2):
    - The string "decision-driving" appears in the rendered report if and only if `audit_status.json.ml_v2_band_outcome == "confidence_improving"`
    - The string "decision-useful" appears if and only if `ml_v2_band_outcome == "risk_revealing"`
    - Neither appears for `ml_v2_band_outcome == "neutral"`
    - The phrase "ML is decision-driving" is grep-banned unless `confidence_improving` in audit JSON
  - **Acceptance:** Both the shell script and pytest run exit 0 against the implemented v2 code. Failure output names the offending file, line number, and stale term.
  - **Why this matters:** This test guards against the exact terminology drift the Lo-style audit identified between requirements/design/tasks. Without it, future edits could silently reintroduce the old flawed naming.
  - **Reqs:** 11 (band-outcome invariants), 12 (pre-registration discipline), and as a meta-gate on all v2 implementation correctness

---

## Cross-Cutting: Test Coverage Targets

- New test files: `test_peer_freshness.py`, `test_analyst_estimates.py`, `test_nlp_coverage.py`, `test_market_features.py`, `test_quarterly_panel.py`, `test_ml_decision.py`
- Combined new test count: ≥40 tests
- All new modules have ≥80% line coverage
- All public methods have at least one happy-path and one edge-case test

---

## Estimated Effort

| Milestone | Hours |
|---|---|
| 0: MLConfig | 2 |
| 1: Data foundation | 6 |
| 2: NLP recovery | 8 |
| 3: Market features | 4 |
| 4: Quarterly panel | 8 |
| 5: ML decision engine | 12 |
| 6: Pipeline integration | 4 |
| 7: Report & charts | 6 |
| 8: Validation | 4 |
| **Total** | **~54 hours** |
