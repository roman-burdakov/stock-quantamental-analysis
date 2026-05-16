# Implementation Analysis — V2 Critical Issues

This document drills into the five most difficult implementation issues in the v2 spec. Each section: the problem, why naive approaches fail, the chosen approach, and remaining residual risks.

---

## 1. Historical Analyst Data Is Not Available

### Problem

The professor explicitly named "analyst-estimate revisions" as one of four feature groups. yfinance is the only free, public source. yfinance returns a **current snapshot** of analyst estimates and recommendations — not a time series. There is no `yfinance.Ticker("NVDA").get_recommendations(as_of="2020-06-30")` that returns 2020 consensus estimates.

### Why the Naive Approaches Fail

**Naive approach 1: "Just include analyst features in the walk-forward panel."**
- Today's snapshot would be applied to 2018, 2019, ..., 2025 historical rows
- Every historical training row would carry **2026-04** analyst data
- This is lookahead by construction; OOS metrics would be inflated and meaningless

**Naive approach 2: "Scrape Wayback Machine for historical analyst pages."**
- Wayback coverage of yfinance pages is sparse and unreliable
- Schema has changed multiple times
- Even if recovered, no consistent date alignment with our quarterly cutoffs
- High effort, low confidence in data quality

**Naive approach 3: "Simulate analyst revisions from price changes."**
- Defeats the purpose; would just be a re-encoded price feature
- Adds no incremental information

### Chosen Approach: Decision-Time Overlay (NOT a Learned Feature)

The reviewer's correction: even with the proposed "single-row analyst-included prediction," a regression model **cannot learn coefficients for features that have only one observation**. Adding analyst features to a model fit on the latest single decision row is mathematically degenerate. The corrected approach treats analyst data as a **side-by-side comparator**, not as inputs to the ML predictor:

**Walk-forward panel** (used for credibility metrics and the published ML prediction):
- Fundamentals + Market + NLP features only
- 44 quarterly rows
- All features are point-in-time historical
- ElasticNet trained on this panel produces the published ML revenue-growth prediction
- Walk-forward OOS metrics (McNemar p, N_OOS, R²) are the published statistical credibility

**Analyst overlay** (reported alongside ML prediction):
- Current snapshot from yfinance (revenue/EPS consensus, recommendation mean, revisions)
- Used to compute a **comparator** to the ML prediction
- Reported as: "ML predicts X% revenue growth; analyst consensus is Y%; difference is Z pp"
- **Never injected into the ML model's training or prediction**
- **Never modifies the target price arithmetically**
- Informs narrative: "ML and analysts concur" or "ML diverges from analyst consensus, possible signal or noise"

### Honest Disclosure Pattern

Report and `model_audit.md` say:

> The walk-forward credibility metrics (directional accuracy, R², MAE-improvement-vs-naive, McNemar paired-test p-value) are computed on a feature set that excludes analyst estimates, because historical analyst-revision time series are not available from public sources. The published ML adjustment to the target price uses **only the ML model's prediction** — the model is trained on fundamentals + market + NLP only and never sees analyst data. Analyst consensus is reported as a side-by-side comparator at decision time and informs narrative discussion (agreement / divergence) but does not arithmetically modify the target price. This is documented at every appearance of analyst data in the report.

### Residual Risks

- A reader could confuse the ML-adjusted target with an "analyst-blended" target. The disclosure must be unambiguous — placed adjacent to every analyst-related number in the report. The report explicitly states: ML adjustment uses ML predictions only; analyst values appear only as a comparator narrative.
- If yfinance changes its schema mid-iteration, the analyst fetch breaks. Mitigation: fetch failure is non-blocking; pipeline produces non-analyst prediction with a warning.

### Why This Is Defensible

The professor's framing is "rebuild the quantitative layer around a denser point-in-time NVIDIA panel" — this is achieved with fundamentals + market + NLP. The analyst layer is an **additional** signal at decision time. A more expensive paid-data approach (I/B/E/S History via Refinitiv) would give historical revisions but is out of scope for an academic project. The chosen approach is the strongest defensible position given the data constraint, and the limitation is documented rather than hidden.

---

## 2. Mixed-Frequency XBRL: YTD-Cumulative Concepts

### Problem

NVIDIA's 10-Q filings report cash-flow concepts (operating cash flow, capex, depreciation) as **year-to-date cumulative** values. Income-statement concepts (revenue, gross profit, R&D) are typically reported as **per-period** quarterly values. Balance-sheet concepts (debt, cash) are **instant** values at period end.

To build a quarterly feature panel, we need each concept at quarterly cadence. Naive use of XBRL facts will mix incompatible frequencies.

### Why Naive Approaches Fail

**Naive approach 1: "Use whatever XBRL gives us per quarter."**
- Q3 capex from a 10-Q would be 9-month YTD, not 3-month
- FCF margin would be 9-month-cash-flow / 3-month-revenue: nonsensical ratio
- Walk-forward training learns garbage; predictions are unreliable

**Naive approach 2: "Skip the YTD concepts; use only per-period quarterly facts."**
- We lose capex, OCF, FCF — critical inputs
- Fundamentals feature group becomes too thin

**Naive approach 3: "Compute capex and OCF only annually."**
- Capex/OCF features become annual-only; defeats the purpose of quarterly cadence
- ~5 useful values across 44 quarters → no improvement over v1

### Chosen Approach: YTD-Difference Derivation with Validation

For each YTD-cumulative concept C and fiscal quarter t in fiscal year FY_y:

```
if t == Q1:
    quarterly_C[FY_y, Q1] = ytd_C[FY_y, Q1]
elif t in {Q2, Q3}:
    quarterly_C[FY_y, t] = ytd_C[FY_y, t] - ytd_C[FY_y, t-1]
elif t == Q4:
    annual_C[FY_y] = annual XBRL value from 10-K
    quarterly_C[FY_y, Q4] = annual_C[FY_y] - ytd_C[FY_y, Q3]
```

### Validation Rule

Sum the four derived quarterly values; compare to the 10-K annual value. If the absolute relative error exceeds 1%, flag the entire FY's derived quarterly values for that concept as `feature_imputed_<col>=True`.

### Edge Cases

| Edge Case | Handling |
|---|---|
| 10-K is amended after 10-Q filings; YTD values restated | Use the latest non-superseded XBRL fact; record amendment lineage |
| 10-Q YTD value missing for a quarter | LOCF the previous quarter's derived value with imputation flag |
| 10-K annual value is later restated | Re-derive Q4 with new annual value; bump panel hash |
| Fiscal year end change | NVIDIA has not changed fiscal calendar in recent history; fall back to per-period if encountered |

### Residual Risks

- If XBRL companyfacts has internal inconsistency between annual and quarterly facts (which v1 already exposed at annual cadence), derivation produces garbage. Mitigation: validation rule + imputation flag.
- Some concepts that look YTD are actually per-period; misclassification produces wrong derivation. Mitigation: per-concept handling table reviewed during implementation; unit-test each concept against published quarterly values from filings.

### Audit Hook

`audit_status.json` includes `quarterly_panel_validation` with per-concept, per-FY mismatch rates. If >20% of FY×concept pairs exceed the 1% tolerance, the gate marks the quarterly panel as `partial_quality` and the report discloses this prominently.

---

## 3. Walk-Forward With an Annual Target (Horizon-Aware Embargo)

### Problem

The secondary target is "next-FY annual revenue growth," with a 4-quarter horizon. A naive walk-forward with 1-quarter embargo would leak information: if we train on quarter Q[t]'s features → Q[t+4]'s annual target, then test on Q[t+1]'s features, the **test row's target spans Q[t+1] through Q[t+4]**, which overlaps with the training row's target window.

This is the same problem as auto-correlated targets in financial time series. It is subtle and easy to get wrong.

### Why Naive Approaches Fail

**Naive approach 1: "1-quarter embargo for all targets."**
- Annual target horizon is 4 quarters; embargo of 1 quarter leaves 3 quarters of overlap
- OOS metrics are inflated by leakage
- Confidence-tier promotion based on inflated metrics → unwarranted

**Naive approach 2: "Hold out a single FY for testing."**
- Only 10 FYs in panel; ~9 observations OOS even with one-FY-out
- Statistical power too low for meaningful tier promotion
- Plus: which FY do we hold out? Selection becomes itself a leakage

**Naive approach 3: "Use only annual rows for the annual target."**
- Reduces panel to 10 rows; cannot meet 25 OOS minimum
- Inferior to v1's 4-observation diagnostic-only outcome

### Chosen Approach: Horizon-Aware Embargo + Quarterly Rows With Overlapping Targets

For the annual target:
- Each quarterly row has a target = next-FY annual revenue growth (for the FY that begins after this quarter ends, OR the FY this quarter is in if before its 10-K is filed)
- Walk-forward fold for split index s:
  ```
  train = rows where target_available_date <= split_filing_date - 4*90 days
  test  = rows where filing_date == split_filing_date
  ```
- The 4-quarter (~365-day) gap between train's max target_available_date and test's filing_date prevents overlap

### Concrete Example

Panel rows sorted by `feature_available_date`:

```
Row index | feature_period | feature_available_date | target_period | target_available_date
1         | FY2018-Q1      | 2018-05-15            | FY2019        | 2019-02-25
2         | FY2018-Q2      | 2018-08-15            | FY2019        | 2019-02-25
3         | FY2018-Q3      | 2018-11-15            | FY2019        | 2019-02-25
4         | FY2018         | 2019-02-25            | FY2020        | 2020-02-21
5         | FY2019-Q1      | 2019-05-20            | FY2020        | 2020-02-21
...
```

To predict target for row 5 (FY2020 annual growth, available 2020-02-21):
- Train must include only rows whose target was known by 2019-05-20 minus 4-quarter embargo (2018-05-20)
- Eligible training rows: only rows with `target_available_date <= 2018-05-20` (which is none, since FY2018 was first available in early 2019)
- Walk-forward starts producing predictions only after sufficient elapsed time

This is restrictive but correct. With 44 quarters and a 4-quarter embargo, useful OOS predictions begin around Q[16+4] = Q20. We get 44 − 20 = ~24 OOS predictions for the annual target, which is at the lower end of meaningful statistical power.

### Quarterly Target Has Looser Constraints

The primary target (`revenue_growth_quarterly_YoY`, 1-quarter horizon) only needs 1-quarter embargo, giving ~30 OOS predictions. This target is the workhorse for confidence-tier evaluation.

### Mitigation for Annual Target Underpowering

Because the annual target has fewer OOS predictions:
- Significance test threshold is **the same** (McNemar paired-test p < 0.20 for contributing, < 0.10 for high-confidence)
- The published ML adjustment uses the annual target prediction, but the **confidence tier** uses whichever target produces the stronger statistical case (typically the quarterly YoY)
- Reported caveats explicit about which target's WF metrics gate which decision

### Residual Risks

- The annual target's true horizon depends on filing timing (10-K filed ~25 days after FY end). Embargo of 4 calendar quarters (= 365 days) is a conservative approximation.
- For FYs where 10-K is filed later than usual, embargo may need adjustment. Rare; flagged in QA only.

---

## 4. Bootstrap Empirical Residual Band on the Target Price

### Problem

The professor wants visible confidence-level changes. R² and MAE are not reader-friendly. We need a target-price band (e.g., "$165 [empirical residual band, 5th–95th percentile: $138–$198, n=27]") that visually communicates uncertainty.

The challenge: how to translate ML walk-forward residuals into target-price uncertainty in a way that is statistically defensible and computationally tractable.

### Why Naive Approaches Fail

**Naive approach 1: "Just use the ML model's residual standard error in the linear regression sense."**
- Assumes Gaussian residuals; nonsense for revenue-growth predictions
- Doesn't account for non-linear DCF translation (target price is not linear in growth)
- Underestimates fat tails

**Naive approach 2: "Use Monte Carlo on every DCF input."**
- Requires distributions for WACC, terminal growth, FCF margins — all subjective
- Conflates ML uncertainty with valuation uncertainty; the user wants the former specifically

**Naive approach 3: "Use a parametric prediction interval from ElasticNet."**
- ElasticNet doesn't give well-calibrated prediction intervals
- Conformal prediction is an option but adds complexity

### Chosen Approach: Bootstrap Residuals → DCF Re-Run

```python
def bootstrap_residual_band(wf_result, dcf_callable, dcf_assumed_growth, n=1000, seed=42):
    residuals = wf_result.oos_predictions["y_true"] - wf_result.oos_predictions["y_pred"]
    rng = np.random.default_rng(seed)
    target_prices = np.empty(n)
    for i in range(n):
        eps = rng.choice(residuals.values)  # sample with replacement
        perturbed_growth = dcf_assumed_growth + eps
        target_prices[i] = dcf_callable(perturbed_growth)
    return np.percentile(target_prices, [5, 50, 95])
```

This produces an empirical residual band that:
- Reflects empirical residual distribution (no Gaussian assumption)
- Translates non-linearly through the DCF (correct because DCF is non-linear in growth)
- Is reproducible with fixed seed
- Has clear interpretation: "the historical out-of-sample errors of our model, applied to today's prediction, give this range"

### Reader-Facing Interpretation

> "The DCF base-case target price is $173. After applying the ML signal's adjustment, the central estimate is $182. The empirical residual band (5th–95th percentile, derived from bootstrapping the model's historical out-of-sample residuals through the DCF, scaling=Method A: 4Q rolling aggregation) is $148–$215, n_residuals=27. This is wider than the DCF-only scenario sensitivity range of $156–$190, reflecting the additional uncertainty from imperfect ML predictions. Per Req 11, this is a *risk-revealing* outcome (band widened by ≥10%), not a *confidence-improving* outcome."

If the ML interval is **narrower** than the DCF-only interval, that's a positive signal: the ML signal is reducing total uncertainty, not adding noise.

### Residual Risks

- Bootstrap assumes residuals are exchangeable across time. If there's a regime shift (e.g., post-COVID), this assumption is violated. Mitigation: use only the most recent N years of WF residuals for the bootstrap.
- The DCF callable must be deterministic and cheap (≤10ms per call); 1,000 calls × 10ms = 10s is acceptable. If DCF runs slow, reduce n_bootstrap to 500.
- Empirical residual band is conditional on all DCF inputs **other than growth**; doesn't reflect uncertainty in WACC, margin assumptions, etc. This is by design — we're isolating ML-uncertainty contribution.

### Comparison to DCF-Only Interval

DCF-only interval is computed using v1's existing scenario-weighted approach (bear/base/bull probabilities applied to scenario-specific growth rates). Both intervals are reported side-by-side in Exhibit 9.

---

## 5. Multiple Comparisons and Selection Bias

### Problem

The walk-forward analysis runs 3 targets × 3 models = 9 model-target combinations. Reporting "the best one" is a form of leakage: the more combinations evaluated, the more likely one looks impressive by chance.

With N≈30 OOS predictions per combination, the expected directional accuracy spread under H0 (no signal) is ±9 percentage points. If you run 9 combinations, the maximum among them has expected value much higher than 50%, even with no signal.

### Why Naive Approaches Fail

**Naive approach 1: "Pick the best model by walk-forward and report it."**
- Selection bias inflates apparent performance
- p-value of "best of 9" is much weaker than p-value of "this specific pre-registered model"

**Naive approach 2: "Bonferroni at α/9."**
- α = 0.10 → individual threshold 0.011
- With N=30 and effect size ≈ 5pp accuracy lift, achieving p<0.011 requires accuracy gap of ~12pp — impractically large
- Test becomes too conservative; nothing passes

**Naive approach 3: "Just don't worry about it; report the best."**
- Misleading; introduces exactly the false-confidence risk that the professor warned against

### Chosen Approach: Pre-Registration of Primary Model + Adjustment Target

Before running walk-forward on the latest data:
1. Commit `MLConfig` to git with: **primary model = ElasticNet** (model class used for all targets), **primary target (per Req 6.1) = quarterly-YoY revenue growth** (the highest-N target, used for the empirical residual band that drives MVP-Essential confidence assessment), and **secondary target (per Req 6.2) = annual revenue growth** (used for DCF target-price adjustment if it reaches contributing tier per Req 8.5)
2. The **published target-price adjustment** uses **only** ElasticNet predictions on the **secondary (annual) target**. The annual target is the only target eligible to modify target price because DCF year-1 growth is the natural mapping target (Req 8.1 horizon mapping).
3. The **published empirical residual band** uses **only** ElasticNet predictions on the **primary (quarterly-YoY) target** because it has higher N_OOS, which gives a more stable bootstrap distribution.
4. Ridge and LassoLars predictions, and the tertiary return-direction target, are reported as **sensitivity analyses** — they do not gate the published adjustment or band.
5. The directional-accuracy, MAE-improvement, and McNemar paired-test for the published prediction are pre-registered: no "best of" selection.

### Naming Reconciliation (per Lo Audit)

| Concept | Name | Used for |
|---|---|---|
| Primary target (Req 6.1) | `rev_growth_quarterly_YoY` | Empirical residual band (Req 9); MVP-Essential confidence-band outcome (Req 11.1) |
| Secondary target (Req 6.2) | `rev_growth_annual_FY` | DCF target-price adjustment (Req 8); only triggers if annual N_OOS ≥ 25 (Req 8.5 fallback) |
| Tertiary target (Req 6.3) | `excess_return_12m_vs_spx_direction` | Sensitivity analysis only; not a gate |
| Primary model | ElasticNet | Used for all three targets; pre-registered |

Document in `model_audit.md`:

> The published ML adjustment uses ElasticNet predictions on the **secondary** next-FY annual revenue growth target (when annual N_OOS ≥ 25 and tier ≥ contributing per Req 8.5). The published empirical residual band uses ElasticNet predictions on the **primary** quarterly-YoY revenue growth target. This pairing was pre-registered before walk-forward execution (see `outputs/ml_config_provenance.json` for the git SHA and commit timestamp). Ridge, LassoLars, and the tertiary return-direction target are reported as appendix sensitivities; their results do not feed the published adjustment or band.

### Why This Choice

- ElasticNet is the most appropriate model class for N≈30 with mixed-strength features (regularization handles colinearity; both L1 and L2 components)
- The primary (quarterly-YoY) target has higher N_OOS, making it the right choice for the residual band
- The secondary (annual) target is more directly relevant to DCF Y1 growth, making it the right choice for target-price adjustment — but only when its N_OOS is sufficient
- Pre-registration is the standard discipline in fields like clinical trials and economics for exactly this problem

### Sensitivity Reporting (Not Gating)

| Sensitivity | Purpose |
|---|---|
| Ridge on secondary annual target | Does L1 sparsity matter for adjustment? |
| LassoLars on secondary annual target | Does pure L1 with low-dim features work better? |
| ElasticNet on tertiary return direction | Does the model see direction in returns? |
| ElasticNet on primary target with seasonal-naive baseline | Robustness of the residual band |

Reported as: "Ridge produces a similar adjustment of $Y, lending robustness to the ElasticNet result." or "LassoLars produces a divergent adjustment of $Z; we proceed with the pre-registered ElasticNet result and note the disagreement in the limitations."

### Residual Risks

- If the pre-registered ElasticNet underperforms a sensitivity model, there's a temptation to swap. The git-committed MLConfig prevents this. Audit check: assert MLConfig SHA at run time matches the commit SHA before walk-forward.
- The pre-registration is only as good as the discipline of the implementer. Mitigation: written commit message explicitly states "MLConfig pre-registered for v2 walk-forward; no post-hoc changes."

---

## 6. Horizon-Mapped DCF Adjustment (Critical Financial-Mechanics Fix)

### Problem

The previous spec applied `adjusted_growth = base_growth + weight × ml_delta` and re-ran the entire DCF with this single growth applied to every projection year. **This is financially wrong.** The ML output predicts next-FY revenue growth (1-year horizon). The DCF uses a multi-year revenue trajectory: e.g., Y1=22%, Y2=18%, Y3=14%, ..., Y10=6%, terminal=3%. A one-year forecast error should not cascade into the long-term CAGR or terminal growth.

### Why Naive Approaches Fail

**Naive approach 1: Apply ML delta to all years.**
- A 5pp ML upside on Y1 becomes a 5pp uplift to terminal growth too
- Terminal value is hyper-sensitive to terminal growth (small changes → large valuation swings)
- A short-horizon prediction effectively rewrites the long-term thesis
- Statistically unsound (model has no validation at 10-year horizon)

**Naive approach 2: Apply only to Y1, leave Y2+ unchanged.**
- Discontinuous trajectory: Y1 = 27%, Y2 = 18% creates an unrealistic step-down
- Real businesses have momentum; if Y1 surprises high, Y2 likely also tilts up
- Underweights credible information

### Chosen Approach: Linear Fade

```
ml_delta = ml_predicted_y1_growth − dcf_base_y1_growth   # both 1Y horizons
weight = tier_weight (0.0, 0.20, or 0.35)

Y1 adjustment: + weight × ml_delta            (full)
Y2 adjustment: + 0.5 × weight × ml_delta      (half)
Y3+ adjustment: 0                             (none)
Terminal: 0                                    (none)
```

This pattern reflects:
- **Y1 is what ML directly predicts** — full credit when statistically validated
- **Y2 has spillover from Y1** — partial credit (operational momentum)
- **Y3+ revert to analyst-judgment trajectory** — long-term thesis is not rewritten by a 1-year forecast
- **Terminal growth is sacrosanct** — represents long-run competitive equilibrium, not in scope for short-term ML

### Concrete Example

DCF base trajectory: `[22%, 18%, 14%, 11%, 9%, 7%, 6%, 5%, 4%, 3%]`, terminal = `3%`

ML predicts Y1 growth of 28%. ML tier = `contributing` (weight 0.20).

ml_delta = 28% − 22% = +6pp

Adjusted trajectory:
- Y1: 22% + 0.20 × 6% = **23.2%**
- Y2: 18% + 0.5 × 0.20 × 6% = **18.6%**
- Y3 through Y10: unchanged
- Terminal: 3% unchanged

This is conservative and defensible. Even if ML is materially off (a few percentage points high or low on Y1), the long-term valuation anchor — terminal growth and Y3+ trajectory — is untouched.

### Reporting in the Report

> The DCF base case projects revenue growth of 22%, 18%, 14%, ... declining to a terminal rate of 3%. The ML model predicts Y1 growth of 28%, **6pp above** the DCF base case. With the model's contributing-tier weight of 20%, this raises the projected Y1 growth to 23.2% and Y2 growth to 18.6%; Y3 onwards and terminal growth are unchanged. The resulting target price is $182, versus $173 DCF-only — a **$9 (5%) increase**.

### Residual Risks

- The 50% Y2 fade and 0% Y3+ fade are choices, not derivations. A reader could argue for different fade patterns. Mitigation: document the choice in `model_audit.md`; sensitivity analysis showing the impact of different fade patterns (e.g., 0%/0% vs 50%/25% vs 100%/50%) in the appendix.
- If the DCF function is not currently year-by-year (uses a single CAGR), the v1 valuation module must be extended. Task 6.2 (update `valuation.py::build_final_recommendation`) covers this.
- Magnitude bucket (±2 / ±1 / 0) on the scorecard is a separate axis from the trajectory adjustment; ensure they are not double-counted.

### Connection to Reviewer's Critique

The reviewer wrote: *"A one-year forecast error should not mechanically replace or adjust a 10-year CAGR unless the mapping is defined."* The mapping is now defined: full-Y1, half-Y2, zero-Y3+, zero-terminal. This is documented in Req 8.1 and `valuation.py` is extended to accept a per-year trajectory.

---

## Summary of Mitigations

| Issue | Naive Failure Mode | Chosen Mitigation |
|---|---|---|
| Historical analyst data | Apply current snapshot to historical rows (lookahead) OR fit one-row model | Decision-time **overlay** (comparator) only; ML model never sees analyst features |
| Mixed-frequency XBRL | Mix YTD and per-period in same column | YTD-difference derivation with validation against 10-K |
| Walk-forward on annual target | 1Q embargo with 4Q-horizon target (leakage) | Horizon-aware embargo: 4Q gap; fallback if N_OOS < 25 |
| Confidence bands | Parametric SE assuming Gaussian; or "calibrated 90% CI" overclaim | Bootstrap residuals through DCF; renamed "empirical residual band" with caveat |
| Multiple comparisons | "Best of 9" selection bias | Pre-registered ElasticNet × annual target |
| DCF horizon mismatch | Apply 1Y ML forecast to 10Y CAGR | Horizon-mapped: full Y1, half Y2, zero Y3+, zero terminal |
| Significance test | Aggregate binomial ignoring pairing | McNemar's exact test on discordant pairs |
| Peer freshness conflation | Single gate for both ML features and valuation comps | Two independent gates: market-price (ML) and peer-financials (comps) |
| NLP coverage vs quality | Coverage above 60% claimed sufficient | Fallback proxy text quality-checked via keyword density (Req 14) |
| Scope risk | All-or-nothing v2 implementation | MVP-tiered path: Essential → Extended → Optional, with v1 fallback |

Each of these is a place where a sloppy implementation produces results that *look* better than the spec promises while being statistically meaningless. The chosen mitigations are conservative and defensible, at the cost of being more restrictive than the maximum possible reportable result. That tradeoff aligns with the professor's stated value: "honesty is a strength."

---

## What Could Still Go Wrong

Even with these mitigations:

1. **The ML signal genuinely doesn't add value.** Possible. NVDA revenue growth is highly auto-correlated and dominated by data-center demand cycles. With N≈30 and noisy targets, the model may not pass the contributing-tier threshold. **In that case, Req 11.5 governs**: report honestly that ML and DCF concur, no material adjustment, and the iteration's value is the methodology infrastructure (which can be re-applied at the next quarter's report or to a different ticker).

2. **Walk-forward looks impressive but is one-off.** Even with proper embargo, a single-decade panel may have favorable autocorrelation structure. Mitigation: report the autocorrelation of residuals; if Ljung-Box test rejects independence, downgrade tier.

3. **The empirical residual band may be misleadingly narrow.** If residuals are heteroscedastic (smaller errors during stable periods, larger during transitions), bootstrap underestimates current uncertainty if we're in a transition regime. Mitigation: report rolling-residual analysis; flag if recent residuals are systematically larger.

4. **Implementation bugs make features wrong.** The most likely failure mode. Mitigation: rigorous unit tests at every module boundary; the v1 post-mortem (XBRL FY2025 misparse) is a reminder that feature pipelines fail silently.

These residual risks are documented in `model_audit.md` after the v2 run and inform the honest-disclosure narrative.
