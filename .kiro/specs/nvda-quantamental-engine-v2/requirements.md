# Iteration 2 Requirements — Decision-Driving ML/NLP Layer (Revised)

## Problem Statement

The v1 pipeline produced a credible Hold but ML/NLP were diagnostic-only. Professor's specific guidance: rebuild the quantitative layer around a **denser point-in-time NVIDIA panel** with four feature groups — **quarterly fundamentals, analyst-estimate revisions, filing-language features, and market/peer-relative variables** — and show whether the ML/NLP layer changes the **rating, target price, or confidence level**.

**Goal:** ML/NLP signals demonstrably move at least one of {rating, target price, confidence band} relative to a DCF-only baseline, with statistical credibility commensurate to sample size.

**Anti-goal:** Polish over substance. A statistically weak ML adjustment dressed up as "decision-driving" is worse than v1's honest gating.

---

## Req 1: Quarterly Fundamentals Panel with Mixed-Frequency Handling

**As a** quantamental analyst
**I want** a quarterly-frequency feature matrix derived from XBRL with explicit handling of YTD-cumulative and balance-sheet concepts
**So that** every feature is correctly point-in-time at quarterly cadence

### Acceptance Criteria

- 1.1 Panel covers FY2015-Q1 through FY2026-Q1 (~44 quarters)
- 1.2 For each quarterly row, `feature_available_date` = filing date of the 10-Q (or 10-K for Q4); features use only data available on or before that date
- 1.3 Income-statement features (gross_margin, operating_margin, R&D_intensity) are derived directly from quarterly XBRL facts
- 1.4 YTD-cumulative concepts (capex, operating_cash_flow) are converted to quarterly: Q_t = YTD_t − YTD_{t-1} within the same fiscal year, with Q1 = YTD_Q1
- 1.5 Balance-sheet concepts (total_debt, total_cash) use end-of-period instant values; LOCF allowed for ≤1 missing quarter, flagged with `feature_imputed_<column>=True`
- 1.6 Each feature column has companion metadata: `<column>_source` ∈ {`xbrl_quarterly`, `xbrl_ytd_derived`, `xbrl_instant`, `locf_imputed`}
- 1.7 Minimum 35 rows with non-null primary target required to enable contributing-tier ML; ≥30 minimum for diagnostic tier
- 1.8 Save to `data/processed/ml_quarterly_panel.csv` with provenance columns (accession numbers, source dates, imputation flags)

---

## Req 2: Analyst-Estimate Decision-Time Overlay (NOT a Learned Feature)

**As a** quantamental analyst
**I want** analyst consensus estimates reported as a decision-time overlay alongside the ML prediction
**So that** the professor's feedback on this feature group is addressed without overstating statistical validity

### Critical Reframing

The previous version of this requirement stated that analyst features would be added to the "decision panel" and used by the ElasticNet model at decision time. **This is mathematically degenerate**: with only the current snapshot available, a regularized regression cannot learn valid coefficients for analyst features. We therefore reclassify analyst data as an **overlay/sensitivity input**, not as a learned ML feature. The ML model is trained and predicts using fundamentals + market + NLP only.

### Acceptance Criteria

- 2.1 Fetch from `yfinance.Ticker("NVDA")`: `earnings_estimate`, `revenue_estimate`, `eps_trend`, `eps_revisions`, `recommendations_summary`
- 2.2 Compute analyst-derived growth signal:
  - `analyst_revenue_growth_curr_yr` (consensus revenue / prior-year revenue − 1)
  - `analyst_revenue_growth_next_yr`
  - `analyst_eps_revision_30d`, `analyst_eps_revision_60d`, `analyst_eps_revision_90d`
  - `analyst_n_analysts`, `analyst_recommendation_mean`, `analyst_recommendation_delta_30d`
- 2.3 Cache snapshot at `data/raw/analyst_estimates_<retrieval_date>.json`
- 2.4 **Analyst data is never injected into the ML training panel.** The ML model uses only historically-available features (fundamentals + market + NLP).
- 2.5 At decision time, the report displays a **side-by-side comparison**:
  - ML-predicted next-FY revenue growth (statistically validated via walk-forward)
  - Analyst-consensus next-FY revenue growth (current snapshot only)
  - Difference and direction of disagreement
- 2.6 The published ML adjustment to target price uses **only the ML prediction**, not analyst-blended values. Analyst data informs the report's narrative ("ML and analysts agree" or "ML diverges from analyst consensus by Xpp; possible signal or noise") but does not directly modify the target price.
- 2.7 If yfinance fetch fails, analyst overlay is omitted from the report with a non-blocking warning. ML adjustment proceeds unchanged.
- 2.8 The report's "Quantitative Signal Integration" section explicitly states: "Walk-forward statistical validation applies to the ML model only. Analyst-consensus values are reported as a current-snapshot comparator with no historical validation possible from public data sources."

---

## Req 3: Filing-Language Features (NLP) with Coverage Recovery

**As a** quantamental analyst
**I want** NLP features extracted from ≥70% of available quarterly periods after re-fetch and improved parsing
**So that** narrative signals contribute non-trivially to the ML model

### Acceptance Criteria

- 3.1 Identify all section JSONs that are empty stubs (~515 bytes) — currently 30 of 46 cached files
- 3.2 For each empty stub:
  - Verify HTML exists in `data/raw/filings/`; re-fetch if missing or truncated
  - Re-parse with form-specific rules (10-Q vs 10-K)
  - If structured extraction still fails, apply fallback: extract first 10,000 chars of cleaned text as `mda_proxy`
- 3.3 Track per-filing extraction quality: `full_extraction` | `partial_extraction` | `mda_proxy_fallback` | `failed`
- 3.4 Compute per-filing features (skip `failed`):
  - `narrative_drift_tfidf` (cosine similarity vs prior filing's MD&A)
  - `narrative_drift_embedding` (sentence-transformers similarity, optional — skip if package unavailable)
  - `sentiment_polarity` (TextBlob)
  - `sentiment_delta` (current − prior)
  - `mda_length_delta`
  - 9 keyword theme scores (existing v1 dictionaries)
- 3.5 For quarterly periods with no filing or `failed` extraction, NLP features are NaN. **No LOCF for NLP** (LOCF would mute drift signal); imputation flag instead.
- 3.6 Coverage gate: NLP feature group is "active" if ≥60% of quarterly rows have any extraction tier ≥ `mda_proxy_fallback`; otherwise NLP features are excluded from the primary ML model but still included in the ablation study

---

## Req 4: Market and Peer-Relative Features (Price-Derived Only)

**As a** quantamental analyst
**I want** market context features computable from already-cached price data without requiring historical peer fundamentals
**So that** the feature group is feasible without unbounded data acquisition

### Acceptance Criteria

- 4.1 Add `^SOX` (Philadelphia Semiconductor Index) and `^GSPC` (S&P 500) to `fetch_market_prices()` and cache in `data/raw/market_prices.csv`
- 4.2 For each quarterly observation (anchored on `feature_available_date`), compute:
  - `nvda_return_3m`, `nvda_return_12m` (trailing returns ending at filing date)
  - `nvda_excess_return_3m_vs_sox`, `nvda_excess_return_12m_vs_sox`
  - `nvda_excess_return_3m_vs_spx`, `nvda_excess_return_12m_vs_spx`
  - `nvda_volatility_60d` (annualized stdev of daily returns)
  - `nvda_beta_252d_vs_sox` (rolling regression slope)
  - For each peer in {AMD, AVGO, INTC, QCOM, MRVL}: `nvda_minus_<peer>_return_12m` (return spread)
- 4.3 All features use only prices with `date ≤ feature_available_date`
- 4.4 **No historical peer fundamental ratios** (P/E rank, EV/Sales rank): excluded due to absence of point-in-time peer fundamentals data. Documented in limitations.md.
- 4.5 Missing market data → NaN (no imputation); document gaps

---

## Req 5: Two Independent Freshness Gates

**As a** quantamental analyst
**I want** market-price and peer-financial freshness checked separately
**So that** each freshness concern is correctly scoped to the features that depend on it

### Critical Distinction

ML market/peer features are **price-derived** (return spreads, vol, beta). They depend on `market_prices.csv` freshness. Valuation comparable multiples are **financial-statement-derived** (P/E, EV/Sales). They depend on `peer_financials.csv` freshness. The previous version of this requirement conflated the two. They are split here.

### Acceptance Criteria — Market-Price Freshness Gate (for ML)

- 5.1 At ML stage entry, check `market_prices.csv`: max date for each ticker (NVDA, ^SOX, ^GSPC, peers) must be ≤ 5 trading days before report_date
- 5.2 If any required ticker is stale (>5 trading days behind), trigger refresh via `EdgarFetcher.fetch_market_prices()`
- 5.3 After refresh, per-ticker check: tickers still missing the most recent 5 trading days are excluded from market features
- 5.4 If NVDA itself is stale → block ML stage entirely (data validation gate equivalent)
- 5.5 Save report to `outputs/market_price_freshness_report.json`

### Acceptance Criteria — Peer-Financials Freshness Gate (for Valuation Multiples Only)

- 5.6 At valuation stage, check `peer_financials.csv`: per-row `source_available_date` must be ≤ 30 days before report_date
- 5.7 If max age > 30 days, trigger refresh
- 5.8 Per-peer check: peers still stale > 30 days are excluded from peer-multiple comparisons (existing v1 behavior, formalized)
- 5.9 If ≥3 of 5 core peers excluded, the peer-multiples exhibit is suppressed (existing v1 quality gate)
- 5.10 Save report to `outputs/peer_financials_freshness_report.json`

### audit_status.json Fields

- `market_price_freshness_status`: "fresh" | "refreshed" | "partial" | "blocked"
- `peer_financials_freshness_status`: "fresh" | "refreshed" | "partial" | "stale_blocked"
- `excluded_tickers_market_price_stale`: [list]
- `excluded_peers_financials_stale`: [list]

---

## Req 6: Multi-Target ML Framework with Horizon-Aware Validation

**As a** quantamental analyst
**I want** ML targets that match decision relevance and walk-forward validation that respects each target's horizon
**So that** OOS metrics are credible

### Acceptance Criteria

- 6.1 **Primary target:** `revenue_growth_quarterly_YoY` at quarter t+1 (i.e., predict next quarter's revenue divided by same-quarter-prior-year revenue minus 1). Removes seasonality. Horizon = 1 quarter.
- 6.2 **Secondary target:** `revenue_growth_annual_FY` for the next full fiscal year (used for DCF integration). Horizon = 4 quarters.
- 6.3 **Tertiary target:** `excess_return_12m_vs_spx_direction` — binary, 1 if NVDA's forward 12-month return exceeds the S&P 500's forward 12-month return, computed strictly from prices ≥ feature_available_date. Horizon = 4 quarters. **Replaces the v1 lookahead-prone "above median" definition.**
- 6.4 Walk-forward validation parameters per target:
  - Primary (1Q horizon): min 12Q training, 1Q embargo, 1Q test
  - Secondary (4Q horizon): min 16Q training, 4Q embargo, 1Q test
  - Tertiary (4Q horizon): min 16Q training, 4Q embargo, 1Q test
- 6.5 Models, in order of preference: ElasticNet (primary), Ridge (regularization comparison), LassoLars (sparser baseline). **GradientBoosting is excluded** due to overfitting risk at N≈30.
- 6.6 **Pre-registered model selection:** ElasticNet is the model used for the published ML adjustment. Ridge and LassoLars are reported as sensitivity analyses. No post-hoc "best model" selection across the three regressors.
- 6.7 Naive baselines per target (reported alongside model):
  - Primary: persistence (`y_{t+1} = y_t`); seasonal-naive (`y_{t+1} = y_{t-3}` last cycle)
  - Secondary: 3-year trailing mean; persistence (same growth as last FY)
  - (Analyst consensus is NOT a baseline — historical revisions unavailable; see Req 2 and Req 10 group E for the overlay treatment)
  - Tertiary: 50-50 (coin flip); always-positive (post-2015 base rate)

---

## Req 7: Statistical Significance Gates (McNemar Paired Test)

**As a** quantamental analyst
**I want** ML signal promotion gated by McNemar's exact test on paired model-vs-baseline outcomes
**So that** statistical significance reflects the correct paired-data structure

### Test Specification

For directional accuracy comparison (model M vs baseline B over N OOS predictions), build the 2×2 contingency table:

|  | Baseline correct | Baseline wrong |
|---|---|---|
| Model correct | b | a |
| Model wrong | c | d |

The discordant pairs are `a` (model right, baseline wrong) and `c` (model wrong, baseline right). McNemar's exact test computes the binomial p-value on `min(a,c)` out of `a+c` under H0 that model and baseline are equivalent. This is the correct test for paired binary outcomes; the previous "aggregate binomial vs baseline correctness count" was a weaker approximation.

### Implementation

```python
from scipy.stats import binomtest

def mcnemar_exact(a: int, c: int) -> float:
    """McNemar's exact test on discordant pairs (a, c).
       Returns one-sided p-value: H1 is model > baseline.
    """
    n_disc = a + c
    if n_disc == 0:
        return 1.0  # Identical outcomes, no evidence
    if a <= c:
        return 1.0  # Model not better than baseline
    return binomtest(a, n_disc, p=0.5, alternative="greater").pvalue
```

### Acceptance Criteria

- 7.1 **Diagnostic tier** (default): N_OOS < 25 OR directional accuracy ≤ 50% OR McNemar p > 0.20 vs best naive baseline OR MAE_improvement ≤ 0%
- 7.2 **Contributing tier**: directional accuracy ≥ 55% AND N_OOS ≥ 25 AND McNemar p < 0.20 AND R²_OOS ≥ 0 (regression) or AUC ≥ 0.55 (classification) AND **MAE improvement over best naive ≥ 10%** (regression targets) or **log-loss improvement ≥ 10%** (classification targets)
- 7.3 **High-confidence tier**: directional accuracy ≥ 60% AND N_OOS ≥ 30 AND McNemar p < 0.10 AND R²_OOS ≥ 0.05 or AUC ≥ 0.60 AND **MAE improvement over best naive ≥ 20%**
- 7.4 Per-target tier assigned independently
- 7.5 The McNemar test, naive-baseline pairing, and tier assignment are recorded in `data/processed/ml_significance_tests.csv` with columns: `target`, `baseline_type`, `n_oos`, `n_concordant`, `n_discordant_a`, `n_discordant_c`, `mcnemar_p`, `mae_model`, `mae_naive_best`, `mae_improvement_pct`, `tier_assigned`
- 7.6 Ties (where both model and baseline produce the same prediction value within a numerical tolerance) are excluded from `a` and `c` and reported separately as `n_ties` in audit output

### Why the Economic-Error Gate Matters

Directional accuracy alone is misleading for high-growth NVDA: revenue growth has been positive in ~85% of recent quarters. A naive "always positive" baseline would achieve ~85% directional accuracy automatically. The model could appear directionally correct while being economically useless. The MAE-improvement gate ensures the ML model adds **economic magnitude information**, not just directional classification. Both must improve for tier promotion.

---

## Req 8: ML → Decision Integration with Horizon-Mapped DCF Adjustment

**As a** quantamental analyst
**I want** the ML signal to adjust **near-term** DCF growth (Y1–Y2) and fade to base trajectory by Y3+
**So that** a one-year forecast does not mechanically distort the multi-year CAGR

### Critical Design Change

The previous version used `adjusted_growth = dcf_base_assumed_growth + weight × ml_growth_delta` and re-ran the entire DCF with the adjusted growth applied to all years. This is **financially incoherent**: a one-year forecast error should not modify the long-term CAGR or terminal growth. Standard equity-research practice is to adjust near-term years and fade back to base.

### Horizon-Mapped Adjustment Formula

```
ml_growth_delta = ml_predicted_annual_growth − dcf_base_y1_growth   # both are 1Y horizons
weight = {diagnostic: 0.0, contributing: 0.20, high_confidence: 0.35}[tier]

# Apply weight to Y1 fully, fade by 50% at Y2, zero from Y3 onward
adjusted_growth_y1 = dcf_base_y1_growth + weight × ml_growth_delta
adjusted_growth_y2 = dcf_base_y2_growth + 0.5 × weight × ml_growth_delta
adjusted_growth_y3_plus = dcf_base_y3_plus_growth   # unchanged
terminal_growth = dcf_base_terminal_growth          # unchanged

adjusted_target_price = recompute_DCF(
    revenue_growth_trajectory=[adjusted_growth_y1, adjusted_growth_y2, adjusted_growth_y3_plus, ...],
    same WACC, margins, terminal_growth, scenario probabilities
)
```

### Acceptance Criteria

- 8.1 ML adjustment modifies only Y1 (full weight) and Y2 (half weight); Y3+ and terminal growth are unchanged
- 8.2 Weight values are a single source of truth in `EngineConfig.MLConfig.adjustment_weights`
- 8.3 The DCF function accepts a per-year revenue-growth trajectory (not a single CAGR); existing v1 DCF will be extended to support this
- 8.4 Scorecard: ML/NLP row with score = sign(ml_growth_delta) × magnitude_bucket where bucket = ±2 if |delta|>10pp, ±1 if 3–10pp, 0 if <3pp; weight applied = same `weight` value, max contribution 0.35 × 2 = 0.7

### Annual-Target Underpowering Fallback (NEW)

- 8.5 If walk-forward N_OOS for the annual target is **< 25**: ML **does not adjust the target price**. Instead:
  - The primary quarterly-YoY target's residuals are used to compute the empirical residual band on the existing DCF target (Req 9)
  - The report states: "ML walk-forward on annual target produced only N_OOS={N} predictions, below the contributing-tier threshold. Target price equals DCF-only target. Empirical residual band is reported using the higher-N quarterly target."
- 8.6 If the annual target's confidence tier is `diagnostic` for any reason (significance, R², N_OOS): same fallback as 8.5

### Reporting

- 8.7 Every report copy of the target price labels both:
  - "DCF-only target: $X (Y1 growth = G1%, Y2 = G2%, ..., terminal = T%)"
  - "ML-adjusted target: $Y (Y1 growth = G1+ΔG1%, Y2 = G2+ΔG2%/2, ..., terminal = T% unchanged)"
  - "ML weight: W%, confidence tier: T, N_OOS: N"
- 8.8 If `tier = diagnostic` (or fallback per 8.5): "ML signal does not meet contributing-tier thresholds; final target price equals DCF-only target."
- 8.9 No silent rating change: if a tier promotion would flip Buy↔Hold↔Sell, the report dedicates a paragraph to the disagreement and an explicit reconciliation

---

## Req 9: Empirical Residual Band on Target Price

**As a** quantamental analyst
**I want** a target-price band derived from walk-forward residuals, labeled honestly as an *empirical residual band* (not a calibrated statistical confidence interval)
**So that** the visible uncertainty does not overclaim formal coverage guarantees

### Naming and Honest Labeling

The previous version of this requirement used "90% confidence interval" language. With ~20–30 OOS residuals and possible regime shifts in NVIDIA's business, the bootstrap output does not provide formal coverage guarantees. We rename the output to **"empirical residual band (5th–95th percentile of bootstrapped residuals)"** with explicit caveat in every appearance.

### Acceptance Criteria

- 9.1 Collect OOS residuals for the **primary target** (quarterly-YoY revenue growth — has higher N_OOS than annual)

- 9.1a **Quarterly→annual residual scaling rule** (new): A quarterly-YoY revenue-growth residual is **not** directly equal to an annual-revenue-trajectory residual. Apply explicit scaling before perturbing the DCF. Choose ONE of the following methods (default = Method A):
  - **Method A — Rolling 4-quarter aggregation (preferred):** Aggregate consecutive quarterly residuals into rolling 4-quarter windows. Each window's mean residual approximates an annual-scale forecast error. Use these aggregated residuals (typically N_aggregated ≈ N_quarterly − 3) as the bootstrap pool for DCF perturbation. This converts quarterly-scale residuals into annual-scale residuals via aggregation, with no arbitrary scaling factor.
  - **Method B — Conservative scaling:** Multiply each quarterly residual by a fixed factor of 0.5 to reflect the partial-year nature of the signal, and label the band "quarterly-signal residual band, conservatively scaled." Document the factor in `model_audit.md`.
  - **Method C — Decoupled reporting:** Do not perturb the DCF target price with quarterly residuals at all. Report the quarterly residual band as "fundamental-forecast uncertainty (quarterly model)" alongside the DCF-only target as a separate exhibit. The DCF target carries no ML-implied band when only the quarterly target has sufficient N_OOS.
- 9.1b The chosen method (A/B/C) is recorded as `MLConfig.residual_scaling_method` and committed to git pre-walk-forward (per Req 12)
- 9.1c The report explicitly states which method was used and why: e.g., "The empirical residual band uses 4-quarter rolling aggregation of quarterly walk-forward residuals (Method A); n_aggregated_residuals = 27."

- 9.2 Bootstrap 1,000 samples of the (scaled) residual distribution
- 9.3 For each bootstrap sample, perturb `dcf_base_y1_growth` by a draw, re-run DCF (with horizon mapping per Req 8), collect target price
- 9.4 Report 5th, 50th, 95th percentile target prices: e.g., "$165 [empirical residual band: $138–$198, n_residuals=27, scaling=Method A]"
- 9.5 Compare DCF-only sensitivity range (existing v1 scenario-weighted) vs ML-augmented residual band; show whether the ML signal **tightens or widens** the band (in absolute dollar width)
- 9.6 Empirical residual band figure included in the report (Exhibit: "Target Price Distribution: DCF-only Sensitivity vs ML Empirical Residual Band")
- 9.7 Every appearance in the report includes the caveat: "*This is an empirical residual band derived from {scaling method description}. It is not a formally calibrated prediction interval; coverage depends on residual stationarity and the validity of the quarterly→annual scaling rule. Reported here for diagnostic uncertainty visualization.*"
- 9.8 If N_OOS < 20 for the primary target: do not produce the band; report only DCF-only sensitivity range and document why ML residual band was not computed

---

## Req 10: Ablation & Marginal Contribution Reporting

**As a** quantamental analyst
**I want** an ablation table demonstrating each feature group's marginal contribution
**So that** the professor sees direct evidence of decision-driving content

### Acceptance Criteria

- 10.1 Five ablation configurations (all run on the **secondary target**, annual revenue growth):
  - **A. Fundamentals-only** (Req 1 features)
  - **B. A + Market** (+ Req 4 features)
  - **C. A + NLP** (+ Req 3 features)
  - **D. A + B + C** (no analyst estimates — runs on full historical panel)
  - **E. Analyst overlay comparator** — current snapshot only, NOT a walk-forward feature group, NOT trained, NOT in any ML matrix. Reported alongside ablation A–D as a side-by-side comparator showing how analyst consensus compares to the ML predictions. The professor's "analyst-estimate revisions" feedback is addressed via this overlay; statistical validation does not apply.
- 10.2 For A–D: walk-forward R², MAE, MAE-improvement-vs-naive, directional accuracy, N_OOS, McNemar p-value vs best naive
- 10.3 For E: report the analyst-consensus snapshot side-by-side with the ML prediction (no walk-forward, no statistical metrics — the comparator narrative is descriptive)
- 10.4 Marginal-lift table: A→B, A→C, B→D — measures contribution of each group
- 10.5 Save to `data/processed/ml_ablation_results.csv`; include figure `outputs/figures/ml_ablation.png`
- 10.6 Report narrative explicitly addresses: "Adding {Market | NLP} features changes directional accuracy by X.X percentage points (McNemar paired-test p-value Y.YY) and MAE by Z.Z%. {This is | This is not} statistically meaningful at the α=0.10 level (Bonferroni-corrected for 3 ablation comparisons: α/3 = 0.033)."

---

## Req 11: Decision-Driving Success Criteria (Pre-Registered, Hierarchical)

**As a** quantamental analyst
**I want** a hierarchical success definition with one primary criterion and ranked secondaries
**So that** the iteration's success is evaluated without "any one of N" multiple-testing inflation

### Hierarchy (Pre-Registered)

The previous "any one of four" framing was a multiple-testing risk. With small N_OOS, the more success paths allowed, the higher the chance of luck-driven success. The corrected framing has a single primary criterion with explicit ordering of secondaries.

### Primary Success Criterion (REQUIRED for "decision-driving")

- 11.1 **Empirical residual band impact from the pre-registered primary target** (quarterly-YoY revenue growth):
  - The empirical residual band (computed per Req 9 with declared scaling method) **narrows** the DCF-only sensitivity range by ≥10% in absolute dollar width (i.e., ML-augmented band width < 0.90 × DCF-only sensitivity range width), AND
  - The walk-forward primary-target McNemar p-value is < 0.10 vs the best naive baseline, AND
  - The MAE of the primary-target ML model improves by ≥10% over the best naive baseline (per Req 7's economic-error gate)
- If primary criterion is met → v2 is **"decision-driving (confidence-improving)"**; the report headlines this outcome
- If primary criterion is NOT met → v2 is **NOT decision-driving on the primary axis**; v2 may still be reported but is not headlined as primary success

### Band-Outcome Classification (Pre-Registered, Single Source of Truth: mvp_path.md)

The MVP path document defines three mutually exclusive outcomes. Only the first counts as decision-driving success:

| Outcome | Band-width condition (vs DCF-only sensitivity range) | McNemar + MAE gates | Counts as decision-driving success? |
|---|---|---|---|
| **Confidence-improving** | Band narrows by ≥10% | Required | **Yes** (Req 11.1 primary success met) |
| **Risk-revealing** | Band widens by ≥10% | Not gated; reported regardless | **No**; decision-useful but not success |
| **Neutral** | Within ±10% of DCF-only width | Not relevant | **No** |

The report classifier emits exactly one of these three labels in `audit_status.json.ml_v2_band_outcome`. The headline language is gated:
- `confidence-improving` → "ML is decision-driving on the primary axis"
- `risk-revealing` → "ML reveals additional forecast uncertainty (decision-useful but does not meet primary success criterion)"
- `neutral` → "ML and DCF concur within statistical noise"

### Secondary Success Criterion (REPORTED separately)

- 11.2 **Target-price impact from the annual target** (only if annual N_OOS ≥ 25 AND tier ≥ contributing):
  - |ML-adjusted target − DCF-only target| / DCF-only target ≥ 3%
  - This is reported as an additional finding when met, but does not substitute for the primary criterion

### Tertiary Success Criterion (DESCRIPTIVE only)

- 11.3 **Ablation lift** (descriptive, not gate):
  - At least one feature-group addition (B, C) improves walk-forward directional accuracy by ≥3 percentage points
  - Reported as descriptive evidence of feature contribution; does NOT count as decision-driving success on its own due to multiple-comparison concerns
  - Ablation results are reported with a Bonferroni-corrected p-value caveat: "α-corrected for 3 ablation comparisons = α/3"

### Rating Impact (Out of Scope as Success Criterion)

- 11.4 If the ML adjustment causes a Buy↔Hold↔Sell rating change, this is reported as an additional finding **with explicit reconciliation discussion**, not counted as success automatically (rating changes from a small-N ML signal warrant explanation, not celebration)

### Honest "Not Decision-Driving" Outcome

- 11.5 If primary criterion (11.1) is not met:
  - The report states: "ML and DCF analyses concur within statistical noise on the pre-registered primary criterion. ML's contribution this cycle is confirmation rather than redirection."
  - Secondary and tertiary findings are still reported in the appendix
  - The report does NOT claim "decision-driving" success based on secondary or tertiary criteria alone
  - This is a valid result and is presented without inflation

### Summary Table in Report

| Criterion | Tier | Threshold | Result | Status |
|---|---|---|---|---|
| Empirical residual band narrows by ≥10% + McNemar p<0.10 + MAE improvement ≥10% | Primary (confidence-improving only) | All three | … | ✓/✗ |
| Empirical residual band widens by ≥10% (risk-revealing) | Reported, not success | One condition | … | reported |
| Target-price |Δ| ≥ 3% (annual target contributing tier) | Secondary | Both | … | ✓/✗/N/A |
| Ablation lift ≥3pp directional accuracy | Tertiary | Descriptive | … | reported |
| Rating change | Reported | Discuss | … | reported |

---

## Req 12: Pre-Registered Implementation (No Post-Hoc Tuning)

**As a** quantamental analyst
**I want** model hyperparameters, feature lists, and decision thresholds frozen before walk-forward execution
**So that** OOS metrics are uncontaminated by selection bias

### Acceptance Criteria

- 12.1 `EngineConfig` contains a frozen `MLConfig` dataclass with: `feature_columns_per_group`, `model_hyperparameters`, `walk_forward_params`, `tier_thresholds`, `adjustment_weights`, `residual_scaling_method` (per Req 9.1a), `target_definitions`, `random_seed`
- 12.2 The MLConfig is committed to git **before** the walk-forward is run on the latest data
- 12.3 No tuning loop in the pipeline that adjusts these based on OOS metrics
- 12.4 Hyperparameter search (if any) is run on a held-out development period (e.g., FY2015–FY2018 only) with the rest kept for walk-forward; the development period is excluded from the reported walk-forward metrics

### Pre-Registration Provenance Manifest (Strengthened)

- 12.5 At ml_v2 stage entry, write `outputs/ml_config_provenance.json` with the following fields:
  - `git_commit_sha`: HEAD commit SHA at run time
  - `git_dirty`: bool, true if working tree has uncommitted changes
  - `git_dirty_files`: list of dirty file paths if `git_dirty=True`
  - `mlconfig_hash`: SHA256 of the canonical-JSON-serialized `MLConfig` dataclass
  - `panel_csv_hash`: SHA256 of `data/processed/ml_quarterly_panel.csv`
  - `feature_columns_used`: list of feature column names per group
  - `target_definitions_version`: semver string for target construction logic (incremented on any change to target definitions)
  - `raw_data_manifest_hash`: SHA256 of a manifest listing all input files (companyfacts, market_prices, peer_financials, sections JSONs) with their sizes and modification times
  - `mlconfig_committed_pre_walk_forward`: bool, true if `git_dirty=False` AND the MLConfig was last modified in a commit ≥1 commit before HEAD
  - `pipeline_version`: semver of the pipeline code itself
  - `python_version`, `key_package_versions`: scikit-learn, pandas, numpy, scipy versions
  - `random_seed`: from MLConfig
  - `timestamp`: ISO-8601 UTC
- 12.6 The audit-consistency check verifies that `mlconfig_committed_pre_walk_forward` is `True` for any v2 walk-forward result that promotes a target above diagnostic tier; otherwise, the tier is forcibly downgraded to diagnostic with reason `pre_registration_violation`
- 12.7 The model_audit.md report cites the full provenance JSON content (not just a SHA) in an appendix subsection "Pre-Registration Manifest"

---

## Req 13: Updated Report Integration

**As a** quantamental analyst
**I want** the report to clearly show ML/NLP impact (or non-impact) on the recommendation
**So that** the grading rubric criteria are fully addressed

### Acceptance Criteria

- 13.1 Executive summary first page: "DCF-only target: $X | ML-adjusted target: $Y (or 'no adjustment — fallback' if Req 8.5 triggered) | Empirical residual band (5th–95th percentile, scaling=Method A): $A–$B | n_residuals: N | Confidence tier: T | Outcome: confidence-improving | risk-revealing | neutral (per Req 11)"
- 13.2 New main-body section "Quantitative Signal Integration" (1.5–2 pages):
  - Walk-forward results table per target
  - Significance test results (McNemar paired-test p-values vs naive baselines)
  - Ablation table with marginal-lift commentary
  - Confidence-tier explanation and adjustment math
  - Bootstrap target-price distribution figure
  - Explicit success-criteria check (Req 11) outcome
- 13.3 Recommendation scorecard includes "ML/NLP Signal" row with weight from Req 8.3
- 13.4 If success criterion is not met (Req 11), the report dedicates a paragraph to the honest outcome
- 13.5 New appendix subsection "Quantitative Layer Methodology" describing walk-forward protocol, feature taxonomy, and significance tests
- 13.6 Two new exhibits: ablation chart (Exhibit 8) and target-price distribution chart (Exhibit 9)

---

## Req 14: NLP Fallback-Extraction Quality Gate

**As a** quantamental analyst
**I want** a quality gate beyond raw coverage on NLP extraction
**So that** noisy fallback text (TOC, boilerplate) does not pollute NLP features

### Why

The fallback `mda_proxy` extraction (first 10,000 chars of cleaned text) can include table-of-contents, signature blocks, exhibit lists, or legal boilerplate when filings have unusual structure. Coverage above 60% does not guarantee meaningful content. A quality gate is required.

### Acceptance Criteria

- 14.1 For every filing extracted via `mda_proxy_fallback`, compute keyword density: count occurrences of MD&A indicator terms in the extracted text, normalized by text length:
  - Required terms (≥1 must appear): `revenue`, `results of operations`, `net income`, `gross margin`, `compared to`, `quarter`, `fiscal year`
  - If 0 of these appear in the proxy text, mark the extraction as `proxy_low_quality` automatically
- 14.2 **Manual QA sample (strengthened)**: At pipeline run end, sample **10** `mda_proxy_fallback` extractions (or all of them if fewer than 10 exist). For each:
  - Emit the first 2,000 characters to `outputs/nlp_proxy_samples.txt`
  - Pipeline writes a stub CSV `outputs/nlp_proxy_qa.csv` with columns `accession`, `quality_label`, `qa_reviewer`, `qa_date`
  - The `quality_label` column is left blank for human reviewer to fill: `relevant_mda` | `partial_relevant` | `boilerplate` | `toc_or_signature` | `unrelated`
- 14.3 **Manual QA acceptance threshold (HARD GATE)**: NLP feature group can be "active" **only if both** of these are true:
  - `outputs/nlp_proxy_qa.csv` is fully filled out (no blank `quality_label` rows)
  - **≥8 of 10** sampled fallback extractions are labeled `relevant_mda` or `partial_relevant`
  
  If either condition fails, NLP feature group is forced to `diagnostic_only` regardless of any other metric. This is a non-overridable hard gate per the Lo audit.

- 14.4 **Audit enforcement**: The pipeline's audit-consistency check writes `audit_status.json.nlp_proxy_qa_complete: bool` (true only if 14.3 conditions met). The audit gate **MUST** be true for `audit_status.json.nlp_feature_group_status == "active"`. The consistency test (Task 8.5) verifies this implication holds: if `nlp_feature_group_status == "active"` but `nlp_proxy_qa_complete == false`, the audit fails with reason `nlp_qa_incomplete`.

- 14.5 **Pre-pipeline manual QA workflow**: To avoid blocking the pipeline run, the manual QA workflow is:
  1. First pipeline run produces unfilled `outputs/nlp_proxy_qa.csv`, NLP is diagnostic-only
  2. Reviewer manually fills in the 10 `quality_label` values
  3. Subsequent pipeline runs read the filled CSV; if 14.3 conditions are met, NLP becomes active in that run
  4. The audit consistency check enforces: NLP is never reported as active without a filled QA CSV containing ≥8/10 relevant labels

- 14.6 Coverage gate (Req 3.6) is computed using only `full_extraction`, `partial_extraction`, and `mda_proxy_fallback` filings that pass both 14.1 and 14.3
- 14.7 If automated 14.1 flags >30% of proxy fallbacks as `proxy_low_quality`, NLP feature group is downgraded to `diagnostic_only` regardless of overall coverage (independent of manual QA)

---

## Req 15: MVP-Essential Implementation Path

**As a** quantamental analyst
**I want** a clearly-defined minimum viable implementation that delivers the professor's core ask
**So that** v1's strong submission is never degraded by partial v2 implementation

### MVP-Essential Scope (must complete or revert to v1)

The following deliverables form the grade-safe MVP. If any cannot be completed cleanly, the v2 stage is **disabled** and the v1 report ships unchanged with a note that ML enhancements were attempted but not validated.

1. **Quarterly fundamentals panel** (Req 1) — fundamentals + market features only (no NLP, no analyst, no peer-financials staleness)
2. **Market-price freshness gate** (Req 5.1–5.5) — for ML features only
3. **Primary target walk-forward** (Req 6.1, 6.4, 6.7) — quarterly-YoY revenue growth with McNemar test
4. **Pre-registered MLConfig** (Req 12)
5. **Empirical residual band on confidence** (Req 9) — applied to existing v1 DCF target; no target-price adjustment
6. **Report section showing DCF-only target with empirical residual band** (Req 13.1, 13.2 narrowed)

The MVP delivers the professor's "show whether ML/NLP changes the confidence level" criterion via the residual band, even if target-price adjustment is not validated. This is the minimum decision-driving impact.

### MVP-Extended Scope (target if MVP-essential succeeds)

7. NLP coverage recovery (Reqs 3, 14)
8. Annual-target walk-forward (Req 6.2)
9. Horizon-mapped DCF target-price adjustment (Req 8) — only if annual target reaches contributing tier
10. Analyst overlay (Req 2) — decision-time comparator only
11. Ablation table (Req 10)

### Out-of-MVP Scope (defer to v3 if needed)

12. Tertiary target (excess return direction, Req 6.3)
13. Peer-financials freshness gate (Req 5.6–5.10) — already partially exists in v1
14. Bootstrap distribution figure as a primary exhibit (move to appendix if MVP-extended fails)

### Acceptance Criteria

- 15.1 Each task in `tasks.md` is tagged `[MVP-essential]`, `[MVP-extended]`, or `[optional]`
- 15.2 The pipeline has a `--ml-v2-mode` flag with values `essential | extended | full`; default is `essential` until MVP-extended is validated end-to-end
- 15.3 If any MVP-essential task fails final validation, `run_pipeline.py` skips the entire ml_v2 stage and emits `outputs/ml_v2_skipped_reason.json`; v1 report ships
- 15.4 The decision to ship v2 features is gated by a final validation script `scripts/validate_v2.py` that exits 0 only if all MVP-essential acceptance criteria pass

---

## Req 16: Model-Stability Gate (Residual Autocorrelation, Regime-Shift)

**As a** quantamental analyst
**I want** acceptance gates on residual stationarity and regime stability
**So that** statistical-significance claims hold under nonstationarity, which is a known concern for adaptive markets

### Why

NVDA's revenue trajectory has had distinct regimes (gaming-led pre-2022, AI/data-center-led 2023+). A walk-forward model that "passes" McNemar might be picking up regime-aligned trends that won't generalize. We need explicit stability gates.

### Acceptance Criteria

- 16.1 **Residual autocorrelation test (Ljung-Box):** Compute Ljung-Box test on walk-forward residuals at lags 1, 2, 4. If p-value < 0.05 at any lag (rejecting white-noise hypothesis): downgrade tier by one level (high_confidence → contributing → diagnostic). Record `ljung_box_p_lag1`, `ljung_box_p_lag2`, `ljung_box_p_lag4`, `ljung_box_downgrade_applied` in audit output.
- 16.2 **Pre/post regime split:** Split the walk-forward OOS predictions into two halves chronologically (or pre-FY2023 vs FY2023+ if the panel spans this transition). Compute MAE in each half. If the more recent half's MAE is ≥30% worse than the earlier half's MAE: downgrade tier by one level and record `regime_mae_ratio_recent_to_earlier`, `regime_split_downgrade_applied`.
- 16.3 **Rolling residual MAE:** Compute MAE over the most recent 8 OOS predictions vs MAE over the earliest 8 OOS predictions. If recent ≥50% worse than earliest: downgrade to diagnostic regardless of other metrics. Record `recent_window_mae`, `early_window_mae`.
- 16.4 All three checks are run automatically inside `MLDecisionEngine.assign_confidence_tier()` after the initial tier is computed; downgrades are applied in this order: 16.1 → 16.2 → 16.3
- 16.5 The model_audit.md reports: "Initial tier: T1. After Ljung-Box check: T2. After regime split: T3. After rolling-window check: T4 (final)." Each downgrade documents the failing metric.
- 16.6 The report's "Quantitative Signal Integration" section discloses any tier downgrade with one sentence explaining the cause (e.g., "Tier downgraded from contributing to diagnostic because Ljung-Box p<0.05 at lag 1, indicating residual autocorrelation that violates walk-forward independence assumptions.")

### Why This Is a Hard Gate

Without this, a model can pass McNemar by virtue of a regime trend (e.g., "growth has been positive every quarter post-COVID, so the model that always predicts positive wins"). This is exactly the failure mode the professor warned against: ML appearing rigorous while not actually generalizing. Promoting Req 16 from prose-level risk to acceptance criterion ensures the gate is automatically enforced.

---

## Non-Requirements (Explicit)

- Deep learning / transformers (interpretability prioritized)
- Real-time streaming data
- Multi-ticker support (NVDA-only, per professor's framing)
- Historical analyst-revision time series (yfinance limitation, documented)
- Historical peer fundamental ratios (data acquisition cost too high; price-derived peer features are used instead)
- Paid data sources (FactSet, Bloomberg, I/B/E/S History)

---

## Cross-Reference: Professor's Feedback Mapping

| Professor's Point | Addressed By |
|---|---|
| "Quarterly fundamentals" | Req 1 |
| "Analyst-estimate revisions" | Req 2 |
| "Filing-language features" | Req 3 |
| "Market/peer-relative variables" | Req 4 |
| "Peer-market data are stale" | Req 5 |
| "Show whether ML/NLP changes rating" | Req 8.6, Req 13.3 |
| "Show whether ML/NLP changes target price" | Req 8.4, Req 13.1 |
| "Show whether ML/NLP changes confidence level" | Req 9, Req 13.1 |
| "Decision-driving (not diagnostic)" | Req 11 (pre-registered success criteria) |
