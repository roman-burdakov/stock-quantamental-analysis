# MVP Path — Grade-Safe v2 Implementation

## Why This Document Exists

V1 produced a clean, defensible 31.75/35 report. V2 expands to five new modules, ~32 tasks, ~54 hours of work. The reviewer correctly flagged that this scope is large enough to **damage the v1 submission** if partially implemented. This document defines the minimum viable v2 that delivers professor-asked decision-driving impact while guaranteeing v1 is never degraded.

---

## Three Tiers

### Tier 1: MVP-Essential (must complete OR revert to v1)

The smallest set of changes that delivers "ML changes the confidence level" — the lowest-bar version of the professor's ask.

| # | Deliverable | Reqs | Tasks | Approx Hours |
|---|---|---|---|---|
| 1 | Quarterly fundamentals panel (no NLP, no analyst) | Req 1 | 4.1, 4.2, 4.3, 4.4 | 8 |
| 2 | Market-price freshness gate for ML | Req 5.1–5.5 | 1.2 (modified) | 2 |
| 3 | Market features (price-derived only) | Req 4 | 1.1, 3.1, 3.2 | 4 |
| 4 | MLConfig pre-registration + git commit gate | Req 12 | 0.1, 0.2 | 2 |
| 5 | Primary-target walk-forward with McNemar | Req 6.1, 6.4, 6.7, Req 7 | 5.1, 5.2, 5.3, 5.4 | 6 |
| 6 | Empirical residual band on existing DCF target | Req 9 | 5.6 | 2 |
| 7 | Report section: residual band + walk-forward narrative | Req 13.1 (narrowed), Req 13.2 (narrowed) | 7.4 (subset), 7.5 (subset) | 4 |
| 8 | MVP validation script | Req 15.4 | new | 2 |
| **Total MVP-Essential** | | | | **~30 hours** |

**Output of MVP-Essential:** The report shows DCF-only target price PLUS an empirical residual band derived from ML walk-forward residuals on the primary quarterly target. The text states whether ML residuals **change** the existing DCF sensitivity range. This is the minimum demonstration that ML changes "confidence level" — the professor's third option.

### Three Band-Change Outcomes (per Req 11)

The MVP-Essential narrative distinguishes three outcomes explicitly. Only the first counts as **decision-driving success**:

| Outcome label | Condition | Narrative |
|---|---|---|
| **Confidence-improving** | ML band ≥10% **narrower** than DCF-only sensitivity range AND McNemar p<0.10 AND MAE improvement ≥10% | "ML signal tightens uncertainty around the DCF target by X%, lending modest additional confidence to the recommendation." Counts as primary success per Req 11.1. |
| **Risk-revealing** | ML band ≥10% **wider** than DCF-only sensitivity range | "ML residuals reveal additional uncertainty not visible in DCF scenario sensitivity. The widened band suggests forecast volatility above analyst-judgment range. This is decision-useful but does NOT count as decision-driving success per the pre-registered hierarchy." |
| **Neutral** | ML band within ±10% of DCF-only sensitivity range | "ML signal neither tightens nor materially widens the DCF target range. ML and DCF concur within statistical noise." |

A wider band can be informative — it tells the reader the model has discovered uncertainty the analyst-judgment scenarios missed. But it is **not "confidence improvement,"** and the report should not claim it is. The previous draft of this document conflated the two; this is corrected.

### Tier 2: MVP-Extended (target if Tier 1 succeeds)

| # | Deliverable | Reqs | Tasks | Hours |
|---|---|---|---|---|
| 9 | NLP coverage recovery + quality gate | Req 3, Req 14 | 2.1, 2.2, 2.3, 2.4 (+ new quality task) | 10 |
| 10 | Annual-target walk-forward (4Q embargo) | Req 6.2 | 5.1 (annual variant) | 4 |
| 11 | Horizon-mapped DCF target-price adjustment | Req 8 | 5.7, 5.8, 6.2 | 6 |
| 12 | Analyst overlay (decision-time comparator only) | Req 2 | 1.3 (modified, no `to_features`) | 4 |
| 13 | Ablation table | Req 10 | 5.5 | 4 |
| **Total MVP-Extended** | | | | **~28 hours** |

**Output of MVP-Extended:** Full target-price adjustment, ablation table proving which feature group contributes, analyst comparator narrative. Decision-driving in the strongest sense.

### Tier 3: Out-of-MVP (defer to v3)

| # | Deliverable | Reason for Deferral |
|---|---|---|
| 14 | Tertiary target (excess return direction) | Adds complexity for marginal value |
| 15 | Peer-financials freshness gate (formal v2 module) | v1 already has informal peer-staleness; not a critical gap |
| 16 | Bootstrap distribution figure as primary exhibit | If MVP-extended fails, move to appendix |
| 17 | Multiple-comparison sensitivity with Ridge/Lasso | Pre-registered ElasticNet is the published path; sensitivities are appendix-only |

---

## Decision Tree at Each Stage

```
                    [Run pipeline]
                         │
                         ▼
            [MVP-Essential validation]
              ├─ pass ───► run MVP-Extended
              └─ fail ───► skip ml_v2 stage
                            ship v1 report unchanged
                            log reason to ml_v2_skipped_reason.json

         [MVP-Extended validation]
              ├─ pass ───► run Out-of-MVP (if time)
              │            ship full v2 report
              └─ fail ───► ship MVP-Essential v2 report
                            (residual band only, no target adjust)
                            log degradation reason
```

---

## CLI Flag and Script Behavior

### `run_pipeline.py --ml-v2-mode {essential|extended|full}`

- `essential` (default until MVP-essential validated): runs Tier 1 only
- `extended`: runs Tier 1 + Tier 2; if Tier 1 fails, falls back to v1 (no ml_v2 stage)
- `full`: runs all three tiers; same fallback behavior

### `scripts/validate_v2.py`

Standalone validation script. Exits 0 only if ALL of:

1. `data/processed/ml_quarterly_panel.csv` exists with ≥30 rows of non-null primary target
2. `outputs/ml_walk_forward_predictions.csv` exists with ≥20 OOS rows
3. `outputs/market_price_freshness_report.json` shows status ≠ "blocked_nvda_stale"
4. `outputs/ml_config_provenance.json` exists with valid SHA
5. `data/processed/ml_residual_band.json` exists with low/median/high values
6. The walk-forward McNemar p-value is reportable (numeric, not NaN)

If any check fails, the script writes a diagnostic to `outputs/v2_validation_failures.json` and exits non-zero. The pipeline's report-generation stage reads this and decides whether to include the v2 section.

### Hard Safety Rule

The v1 report generation is **never gated on v2 success**. The pipeline architecture is:

```
v1 stages (always run) ──► v1 report assembly (always available)
                ▲
                │
v2 stages (try) ──► validate_v2 ──► if pass: enrich report
                                  └─► if fail: log reason, ship v1 report
```

If v2 fails for any reason — XBRL parsing error, yfinance outage, statistical significance not reached, model fits crash — the v1 report still ships. Audit consistency check still passes (v1 audit has no v2 dependency).

---

## What MVP-Essential Looks Like in the Report

Sample text (Executive Summary):

> **Recommendation: Hold.** Target price: $173 (DCF-only, scenario-weighted).
>
> The ML walk-forward model on quarterly revenue growth (44-quarter panel, primary target = next-quarter YoY revenue growth, ElasticNet pre-registered) achieves directional accuracy of **62%** over **30** out-of-sample predictions, with McNemar p-value of **0.07** versus the seasonal-naive baseline. The model's empirical residual band on the DCF target price is **$148–$195**, compared to the DCF-only scenario-weighted sensitivity range of **$143–$202**. The ML signal **tightens** the uncertainty range by **5%**, lending modest additional confidence to the Hold recommendation around $173.
>
> *The empirical residual band is computed by bootstrapping out-of-sample model residuals through the DCF; it is not a formally calibrated prediction interval. ML target-price adjustment was not applied because the annual-revenue-growth target produced fewer than 25 OOS predictions, below the contributing-tier threshold.*

This narrative does the following:
- Demonstrates ML changes the **confidence level** (tighter or wider band)
- Provides walk-forward statistical credibility (McNemar p, N_OOS)
- Honestly admits when target-price adjustment is not applied
- Gives the professor a clear, decision-relevant ML output without overclaiming

---

## What Could Push v2 Beyond MVP-Essential

If after Milestone 4 (quarterly panel) the data quality is high — N_OOS comfortably above 30 for both quarterly and annual targets, fundamentals derivation cleanly validated — proceed to MVP-Extended with confidence.

If the panel is thinner than expected (e.g., capex YTD-derivation has too many imputation flags), keep MVP-Essential and document the constraint honestly.

The cost of being conservative: a B+ → A- novelty score uplift instead of A+ uplift.
The cost of being aggressive and broken: a B novelty score and undermined credibility.

The reviewer's framing is correct: *"do not let a broken v2 degrade the v1 submission."*

---

## Pre-Implementation Checklist

Before writing any code:

- [ ] Confirm with user which tier is the target (Essential, Extended, or Full)
- [ ] Confirm the deadline for v2 deliverable
- [ ] Confirm the willingness to ship v1 unchanged if MVP-Essential fails
- [ ] Commit the MVP-Essential MLConfig to git as the baseline
- [ ] Set the CLI flag default to `essential` until MVP-Essential is validated end-to-end
- [ ] Add `outputs/ml_v2_skipped_reason.json` as a graceful-failure artifact

This document, the requirements.md, and the design.md together comprise the grade-safe implementation plan.