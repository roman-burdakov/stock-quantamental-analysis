# Iteration 2 Design — Decision-Driving ML/NLP Layer (Revised)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│   Existing v1 Pipeline (unchanged except where noted)             │
│   edgar_fetch → xbrl_parser → financial_metrics → valuation       │
│   filing_text_parser (enhanced) → nlp_features (enhanced)         │
│   data_validation → report_utils → audit_utils                    │
└─────────────────────┬────────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────────┐
│   New v2 Modules                                                  │
│                                                                   │
│   analyst_estimates.py ─── yfinance analyst data fetcher          │
│   peer_freshness.py ─────── peer staleness gate + auto-refresh    │
│   market_features.py ────── price-derived market/peer features    │
│   quarterly_panel.py ────── mixed-frequency XBRL → quarterly panel│
│   ml_decision.py ────────── multi-target ML with significance     │
│                              tests, ablation, residual bands     │
└──────────────────────────────────────────────────────────────────┘
```

Five new modules. The v1 pipeline is unchanged except for:
- `filing_text_parser.py`: enhanced 10-Q parsing + fallback proxy extraction
- `nlp_features.py`: adds sentiment polarity feature
- `edgar_fetch.py`: adds `^SOX` and `^GSPC` tickers; adds analyst estimate fetch hook
- `valuation.py`: `build_final_recommendation()` accepts optional `ml_adjustment` param
- `audit_utils.py`: new audit checks for ml_v2 outputs
- `run_pipeline.py`: new `ml_v2` stage between `ml` and `valuation`

---

## Module 1: `src/analyst_estimates.py` (Decision-Time Overlay, NOT a Learned Feature)

### Critical Reframing from Initial Design

The initial design implied a model trained with analyst features for the latest decision row. **This is mathematically degenerate** — an ElasticNet (or any regression) cannot fit valid coefficients for features with only one historical observation. The reviewer correctly flagged this. The corrected design treats analyst data purely as a **decision-time overlay**:

- ML model is trained on fundamentals + market + NLP only (historical data available)
- Analyst snapshot is fetched at decision time and reported alongside the ML prediction as a **comparator**
- The published ML adjustment uses only the ML prediction; analyst values inform narrative but do not modify target price arithmetically

### Class

```python
@dataclass
class AnalystSnapshot:
    retrieval_date: date
    revenue_growth_curr_yr: float | None  # consensus revenue growth, current FY
    revenue_growth_next_yr: float | None  # consensus revenue growth, next FY
    eps_estimate_curr_yr: float | None
    eps_estimate_next_yr: float | None
    eps_revision_30d: float | None
    eps_revision_60d: float | None
    eps_revision_90d: float | None
    n_analysts: int | None
    recommendation_mean: float | None  # 1=Strong Buy, 5=Sell
    recommendation_delta_30d: float | None
    fetch_succeeded: bool
    fetch_error: str | None

class AnalystEstimateFetcher:
    def __init__(self, config: EngineConfig) -> None: ...

    def fetch_snapshot(self, ticker: str, retrieval_date: date) -> AnalystSnapshot:
        """Fetch from yfinance.Ticker; cache to data/raw/."""

    # NOTE: No `to_features()` method. Analyst data is not converted to ML features.
    # It is consumed directly as a decision-time overlay/comparator.

    def render_overlay_summary(
        self,
        snapshot: AnalystSnapshot,
        ml_prediction: float,
    ) -> dict:
        """Returns dict with:
           - analyst_revenue_growth: from snapshot
           - ml_revenue_growth: passed in
           - delta_pp: ml - analyst
           - direction: 'ml_above' | 'ml_below' | 'concur'
           - magnitude: 'large' (>5pp) | 'moderate' (2-5pp) | 'small' (<2pp)
        """
```

### How the Report Uses This

In the "Quantitative Signal Integration" section:

> The ML walk-forward model predicts NVIDIA next-FY revenue growth of **X%**. Analyst consensus (yfinance snapshot, retrieved YYYY-MM-DD) is **Y%**. The ML prediction is **{above|below}** consensus by **|X−Y|pp**. Walk-forward statistical validation (McNemar p={p}) applies only to the ML prediction; the analyst comparator is a current snapshot with no historical-revision time series available from public data sources.

If ML and analyst diverge materially, this is flagged as a discussion point rather than a target-price modifier.

### Implementation Notes

- yfinance schema: `Ticker("NVDA").earnings_estimate` returns DataFrame indexed by `0q, +1q, 0y, +1y` with columns including `avg, low, high, numberOfAnalysts, growth`
- `revenue_growth_curr_yr` derived from `Ticker.revenue_estimate` row `0y`: `(avg / prior_year_revenue) − 1`
- Cache: `data/raw/analyst_estimates_<YYYYMMDD>.json`
- Failures non-blocking: pipeline continues without overlay

---

## Module 2: `src/peer_freshness.py` — Two Independent Gates

### Critical Reframing

The initial design used a single `PeerFreshnessGate` that checked `peer_financials.csv` and removed peers from "peer-relative features." This conflated two unrelated concerns:

1. **ML market/peer-return features** are price-derived (return spreads, volatility). They depend on `market_prices.csv` freshness only.
2. **Valuation comparable multiples** (P/E, EV/Sales) are financial-statement-derived. They depend on `peer_financials.csv` freshness only.

We split these into two independent gates.

### Class Structure

```python
@dataclass
class MarketPriceFreshnessReport:
    threshold_trading_days: int = 5
    tickers_checked: list[str]
    tickers_refreshed: list[str]
    tickers_excluded_stale: list[str]
    nvda_stale: bool
    status: str  # "fresh" | "refreshed" | "partial" | "blocked_nvda_stale"

@dataclass
class PeerFinancialsFreshnessReport:
    threshold_days: int = 30
    peers_checked: list[str]
    peers_refreshed: list[str]
    peers_excluded_stale: list[str]
    status: str  # "fresh" | "refreshed" | "partial" | "stale_blocked"

class MarketPriceFreshnessGate:
    """Gates ML market/peer-return features."""

    def evaluate_and_refresh(
        self,
        market_prices: pd.DataFrame,
        report_date: date,
        edgar_fetcher: EdgarFetcher,
        required_tickers: list[str],
    ) -> MarketPriceFreshnessReport: ...

class PeerFinancialsFreshnessGate:
    """Gates valuation comparable multiples (existing v1 behavior, formalized)."""

    def evaluate_and_refresh(
        self,
        peer_financials: pd.DataFrame,
        report_date: date,
        edgar_fetcher: EdgarFetcher,
    ) -> PeerFinancialsFreshnessReport: ...
```

### When Each Runs

- `MarketPriceFreshnessGate`: at the start of the **ml_v2** stage, before market features are computed
- `PeerFinancialsFreshnessGate`: at the start of the **valuation** stage, before peer multiples are computed (existing v1 location)

### audit_status.json Fields

- `market_price_freshness_status`, `excluded_tickers_market_price_stale`, `nvda_market_data_stale`
- `peer_financials_freshness_status`, `excluded_peers_financials_stale`

---

## Module 3: `src/market_features.py`

### Responsibility

Price-derived market and peer-relative features at quarterly cadence.

### Class

```python
class MarketFeatureBuilder:
    def __init__(self, config: EngineConfig) -> None: ...

    def compute_features(
        self,
        prices: pd.DataFrame,           # daily OHLCV for NVDA + peers + ^SOX + ^GSPC
        as_of_dates: pd.Series,         # one date per quarterly observation
        excluded_peers: list[str],      # from PeerFreshnessGate
    ) -> pd.DataFrame:
        """Returns DataFrame indexed by as_of_date with feature columns."""

    def _trailing_return(
        self,
        prices: pd.Series,
        as_of: date,
        window_days: int,
    ) -> float | None:
        """Total return from (as_of - window_days) to as_of, using last available price ≤ as_of."""

    def _rolling_beta(
        self,
        nvda_returns: pd.Series,
        bench_returns: pd.Series,
        as_of: date,
        window_days: int = 252,
    ) -> float | None:
        """OLS slope of NVDA returns on benchmark returns, using only data ≤ as_of."""

    def _annualized_volatility(
        self,
        returns: pd.Series,
        as_of: date,
        window_days: int = 60,
    ) -> float | None: ...
```

### Feature Output Schema

Columns produced (one row per `as_of_date`):
- `nvda_return_3m`, `nvda_return_12m`
- `nvda_excess_return_3m_vs_sox`, `nvda_excess_return_12m_vs_sox`
- `nvda_excess_return_3m_vs_spx`, `nvda_excess_return_12m_vs_spx`
- `nvda_volatility_60d`, `nvda_beta_252d_vs_sox`
- `nvda_minus_<peer>_return_12m` for each non-excluded peer

### No-Lookahead Test

Unit test asserts: for any `as_of_date`, recomputing features after appending future-dated rows to `prices` yields identical output. This is the strongest no-lookahead invariant.

---

## Module 4: `src/quarterly_panel.py`

### Responsibility

Build the dense quarterly feature/target matrix with mixed-frequency XBRL handling.

### Class

```python
class QuarterlyPanelBuilder:
    def __init__(self, config: EngineConfig) -> None: ...

    def build_panel(
        self,
        metrics: pd.DataFrame,           # from financial_metrics (has annual+quarterly rows)
        nlp: pd.DataFrame,               # from nlp_features
        market: pd.DataFrame,            # from MarketFeatureBuilder
        prices: pd.DataFrame,            # for forward return target
    ) -> pd.DataFrame:
        """
        Pipeline:
          1. _select_quarterly_skeleton(metrics)  — one row per quarterly period
          2. _attach_fundamental_features(metrics) — handle YTD-derived + LOCF
          3. _attach_nlp_features(nlp) — by filing_date join, no LOCF
          4. _attach_market_features(market) — by as_of_date join
          5. _attach_targets(metrics, prices) — primary/secondary/tertiary targets
          6. _validate_no_lookahead() — strict check
        """

    def _derive_quarterly_from_ytd(
        self,
        ytd_series: pd.Series,
        period_index: pd.Series,  # FY2024-Q1, FY2024-Q2, ...
    ) -> pd.Series:
        """Q_t = YTD_t - YTD_{t-1} within same FY; Q1 = YTD_Q1."""

    def _attach_targets(
        self,
        panel: pd.DataFrame,
        metrics: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame: ...
```

### Mixed-Frequency Handling Rules

| Metric | Source Frequency | Quarterly Strategy |
|---|---|---|
| revenue, net_income, gross_profit, operating_income | quarterly direct | Use as-is |
| R&D_expense, SGA_expense | quarterly direct | Use as-is |
| operating_cash_flow | YTD-cumulative in 10-Q | Derive: Q_t = YTD_t − YTD_{t-1} |
| capex (PaymentsToAcquireProductiveAssets) | YTD-cumulative | Derive same way |
| total_debt, cash, total_equity | instant (period-end) | Use period-end value |
| diluted_shares, diluted_EPS | quarterly direct | Use as-is |

For YTD-derived metrics:
- Q1 of fiscal year = YTD value at Q1 reporting period
- Q2 = YTD at Q2 minus Q1 derived value
- Q3 = YTD at Q3 minus Q2 derived value
- Q4 = annual value (from 10-K) minus YTD at Q3
- Validation: derived quarterly values sum to annual; mismatch ≥1% flags `feature_imputed_capex=True`

### Target Construction Detail

```python
# Primary: revenue_growth_quarterly_YoY at t+1
# For row at quarter t (filing_date_t known):
#   target_period = t+1
#   target_value = (revenue[t+1] - revenue[t-3]) / revenue[t-3]
#   target_available_date = filing_date of quarter t+1's 10-Q/10-K

# Secondary: revenue_growth_annual_FY for next FY
# For row at quarter t in FY_y:
#   target_period = FY_{y+1}
#   target_value = (annual_revenue[FY_{y+1}] - annual_revenue[FY_y]) / annual_revenue[FY_y]
#   target_available_date = filing_date of FY_{y+1}'s 10-K

# Tertiary: excess_return_12m_vs_spx_direction
# For row at quarter t (filing_date_t):
#   nvda_fwd = price[filing_date_t + 252 trading days] / price[filing_date_t] - 1
#   spx_fwd  = ^GSPC[filing_date_t + 252 trading days] / ^GSPC[filing_date_t] - 1
#   target_value = 1 if nvda_fwd > spx_fwd else 0
#   target_available_date = filing_date_t + 365 calendar days
```

### Provenance Columns

Every panel row includes:
- `feature_period`, `feature_available_date`, `feature_accession`
- `target_period`, `target_available_date`
- Per-feature imputation flags (`feature_imputed_<col>`)
- Per-feature source tag (`<col>_source`)

---

## Module 5: `src/ml_decision.py`

### Responsibility

Walk-forward training, significance testing, ablation, decision integration, empirical residual bands.

### Classes

```python
@dataclass
class WalkForwardResult:
    target: str
    model_name: str
    feature_group: str
    n_oos: int
    rmse: float
    mae: float
    r2: float
    directional_accuracy: float
    mcnemar_p_vs_naive: float
    naive_baseline_used: str
    naive_directional_accuracy: float
    oos_predictions: pd.DataFrame  # cols: prediction_date, y_pred, y_true

@dataclass
class ConfidenceTier:
    name: str  # "diagnostic" | "contributing" | "high_confidence"
    weight: float  # adjustment weight
    rationale: str

@dataclass
class MLAdjustment:
    dcf_only_target: float
    ml_predicted_growth: float
    dcf_assumed_growth: float
    growth_delta: float
    weight: float
    adjusted_growth: float
    adjusted_target: float
    confidence_tier: ConfidenceTier
    residual_band_low: float
    residual_band_high: float
    success_criteria_met: list[str]

class MLDecisionEngine:
    def __init__(self, config: EngineConfig) -> None: ...

    def walk_forward(
        self,
        panel: pd.DataFrame,
        target: str,
        feature_columns: list[str],
        ml_config: MLConfig,  # frozen
    ) -> WalkForwardResult: ...

    def naive_baseline_directional_accuracy(
        self,
        panel: pd.DataFrame,
        target: str,
        baseline_type: str,  # "persistence" | "seasonal_naive" | "always_positive" | ...
    ) -> tuple[float, int]: ...

    def mcnemar_exact(
        self,
        model_correct: np.ndarray,
        baseline_correct: np.ndarray,
    ) -> dict:
        """Returns {n_concordant, n_discordant_a, n_discordant_c, p_value}."""

    def compute_economic_error_metrics(
        self,
        wf_result: WalkForwardResult,
        baseline_predictions: np.ndarray,
    ) -> dict:
        """Returns {mae_model, mae_naive_best, mae_improvement_pct, rmse_model, rmse_naive_best}."""

    def check_residual_stability(
        self,
        wf_result: WalkForwardResult,
    ) -> dict:
        """Per Req 16: Ljung-Box, regime split, rolling MAE.
           Returns dict with all metrics and any tier downgrades to apply."""

    def assign_confidence_tier(
        self,
        wf_result: WalkForwardResult,
        ml_config: MLConfig,
    ) -> ConfidenceTier: ...

    def compute_ablation(
        self,
        panel: pd.DataFrame,
        target: str = "rev_growth_annual",
    ) -> pd.DataFrame: ...

    def bootstrap_residual_band(
        self,
        wf_result: WalkForwardResult,
        dcf_callable: Callable[[float], float],
        dcf_assumed_growth: float,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float, float]: ...

    def compute_ml_adjustment(
        self,
        decision_panel_row: pd.Series,         # latest quarter: fundamentals + market + NLP only (analyst data is NEVER in this row)
        wf_result: WalkForwardResult,          # walk-forward credibility
        dcf_assumed_growth: float,
        dcf_callable: Callable[[float], float],
    ) -> MLAdjustment: ...

    def evaluate_success_criteria(
        self,
        adjustment: MLAdjustment,
        ablation: pd.DataFrame,
    ) -> list[str]: ...
```

### Walk-Forward Algorithm (Horizon-Aware)

```python
def walk_forward(panel, target, feature_columns, target_horizon_quarters):
    panel_sorted = panel.sort_values("feature_available_date")
    min_train = ml_config.walk_forward.min_train_quarters[target]
    embargo = target_horizon_quarters  # KEY FIX: embargo equals horizon

    oos_preds = []
    for split_idx in range(min_train, len(panel_sorted) - embargo):
        train = panel_sorted.iloc[: split_idx]
        # Embargo: skip rows whose target_available_date overlaps train end
        train_max_target_date = train["target_available_date"].max()
        test_candidate = panel_sorted.iloc[split_idx]
        if test_candidate["feature_available_date"] <= train_max_target_date:
            continue  # would be lookahead
        test = test_candidate

        model = ElasticNet(**ml_config.elasticnet_params)
        X_train = train[feature_columns].fillna(train[feature_columns].median())
        y_train = train[target]
        valid = y_train.notna()
        model.fit(X_train[valid], y_train[valid])

        X_test = test[feature_columns].fillna(X_train.median())
        y_pred = model.predict(X_test.values.reshape(1, -1))[0]
        oos_preds.append({
            "prediction_date": test["feature_available_date"],
            "y_pred": y_pred,
            "y_true": test[target],
        })

    return aggregate_oos(oos_preds)
```

### Significance Test (McNemar's Exact Test)

```python
from scipy.stats import binomtest

def mcnemar_exact(model_correct: np.ndarray, baseline_correct: np.ndarray) -> dict:
    """McNemar's exact test on paired binary correctness arrays.
    Both arrays must be aligned (same OOS predictions for model and baseline).
    Returns: dict with n_concordant, n_discordant_a, n_discordant_c, p_value
    """
    a = int(((model_correct == 1) & (baseline_correct == 0)).sum())  # model right, baseline wrong
    c = int(((model_correct == 0) & (baseline_correct == 1)).sum())  # model wrong, baseline right
    concordant = int((model_correct == baseline_correct).sum())
    n_disc = a + c
    if n_disc == 0:
        p = 1.0
    elif a <= c:
        p = 1.0  # model not better than baseline (one-sided)
    else:
        p = binomtest(a, n_disc, p=0.5, alternative="greater").pvalue
    return {
        "n_concordant": concordant,
        "n_discordant_a": a,
        "n_discordant_c": c,
        "p_value": p,
    }
```

This replaces the previous aggregate-binomial approximation. McNemar is the standard test for paired binary outcomes; aggregate binomial ignores the pairing structure and can over- or under-state significance.

### Confidence-Tier Logic

```python
def assign_confidence_tier(wf, cfg):
    if wf.n_oos < 25 or wf.directional_accuracy <= 0.50:
        return ConfidenceTier("diagnostic", 0.0, "Insufficient OOS or no directional edge")
    if wf.mcnemar_p_vs_naive > 0.20:
        return ConfidenceTier("diagnostic", 0.0, f"Not significant vs naive (McNemar p={wf.mcnemar_p_vs_naive:.2f})")

    contributing_ok = (
        wf.directional_accuracy >= 0.55
        and wf.n_oos >= 25
        and wf.mcnemar_p_vs_naive < 0.20
    )
    high_conf_ok = (
        wf.directional_accuracy >= 0.60
        and wf.n_oos >= 30
        and wf.mcnemar_p_vs_naive < 0.10
        and wf.r2 >= 0.05  # for regression target
    )

    if high_conf_ok:
        return ConfidenceTier("high_confidence", 0.35, "All thresholds met")
    if contributing_ok:
        return ConfidenceTier("contributing", 0.20, "Contributing thresholds met")
    return ConfidenceTier("diagnostic", 0.0, "Did not meet contributing thresholds")
```

### Empirical Residual Band Algorithm

```python
def bootstrap_residual_band(wf, dcf_callable, dcf_assumed_growth, n_bootstrap=1000):
    """
    1. Collect OOS residuals: residuals = y_true - y_pred.
    2. For each of n_bootstrap iterations:
       - Sample one residual with replacement.
       - Perturbed_growth = dcf_assumed_growth + sampled_residual
       - Compute target price = dcf_callable(perturbed_growth)
    3. Return (5th percentile, 50th, 95th).
    """
    residuals = wf.oos_predictions["y_true"] - wf.oos_predictions["y_pred"]
    rng = np.random.default_rng(seed=42)
    target_prices = []
    for _ in range(n_bootstrap):
        eps = rng.choice(residuals.values, size=1)[0]
        perturbed = dcf_assumed_growth + eps
        target_prices.append(dcf_callable(perturbed))
    target_prices = np.sort(np.array(target_prices))
    return (
        np.percentile(target_prices, 5),
        np.percentile(target_prices, 50),
        np.percentile(target_prices, 95),
    )
```

### Decision Integration (Horizon-Mapped, with N_OOS Fallback)

```python
def compute_ml_adjustment(
    decision_row,
    wf_annual,            # WalkForwardResult for annual target
    wf_quarterly,         # WalkForwardResult for primary quarterly target
    dcf_inputs,           # contains base year-by-year growth trajectory
    dcf_callable,         # function(growth_trajectory) -> target_price
    ml_config,
):
    # Step 1: Decide whether annual target is usable for target-price adjustment
    annual_tier = assign_confidence_tier(wf_annual, ml_config)
    annual_n_oos = wf_annual.n_oos

    if annual_n_oos < ml_config.min_n_oos_for_target_price_adjust or annual_tier.name == "diagnostic":
        # FALLBACK: do not adjust target price; only compute residual band
        adjusted_target = dcf_callable(dcf_inputs.base_growth_trajectory)
        adjustment_status = "no_adjustment_low_n_oos" if annual_n_oos < 25 else "no_adjustment_diagnostic_tier"
        ml_growth_y1 = None
        weight_used = 0.0
    else:
        # Apply horizon-mapped adjustment: full weight on Y1, half on Y2, zero from Y3
        ml_growth_y1_pred = predict_with_full_panel(decision_row, wf_annual, ml_config)
        dcf_y1 = dcf_inputs.base_growth_trajectory[0]
        ml_delta = ml_growth_y1_pred - dcf_y1

        weight_used = annual_tier.weight  # 0.20 or 0.35
        adjusted_traj = list(dcf_inputs.base_growth_trajectory)
        adjusted_traj[0] = adjusted_traj[0] + weight_used * ml_delta
        if len(adjusted_traj) > 1:
            adjusted_traj[1] = adjusted_traj[1] + 0.5 * weight_used * ml_delta
        # Years 3+ unchanged; terminal unchanged
        adjusted_target = dcf_callable(adjusted_traj)
        adjustment_status = "applied_with_horizon_mapping"

    # Step 2: Empirical residual band ALWAYS uses the higher-N quarterly target
    # (per Req 9 — primary target has more residuals than annual)
    if wf_quarterly.n_oos >= 20:
        residual_band = bootstrap_residual_band(
            wf_quarterly, dcf_callable, dcf_inputs.base_growth_trajectory[0]
        )
    else:
        residual_band = None

    return MLAdjustment(
        dcf_only_target=dcf_callable(dcf_inputs.base_growth_trajectory),
        adjusted_target=adjusted_target,
        ml_growth_y1=ml_growth_y1,
        weight_used=weight_used,
        adjustment_status=adjustment_status,
        residual_band_low=residual_band[0] if residual_band else None,
        residual_band_high=residual_band[2] if residual_band else None,
        residual_band_median=residual_band[1] if residual_band else None,
        n_residuals_used=wf_quarterly.n_oos if residual_band else 0,
        # ... other fields
    )
```

### Why Horizon Mapping Matters

The previous design did `adjusted_growth = base_growth + weight × ml_delta` and re-ran the DCF with that single growth applied to all years. This is **financially wrong**: a one-year forecast error should not modify a 10-year CAGR or terminal growth. The corrected design adjusts only Y1 (full weight) and Y2 (half weight); Y3+ and terminal growth use the analyst-judgment trajectory unchanged. This aligns with standard equity-research practice.

### Why the Fallback Matters

If the annual target produces fewer than 25 OOS predictions (likely with 4Q embargo on a 44-quarter panel), promoting it to "contributing" tier would be statistically unsound. The fallback preserves the empirical residual band visualization (driven by the higher-N quarterly target) without making a target-price adjustment that lacks credible OOS validation.

---

## Pipeline Integration

### `run_pipeline.py` — New `ml_v2` Stage (Corrected)

```python
# Insert after existing 'ml' stage, before 'valuation'
if step in ("all", "ml_v2"):
    # 1. Market-price freshness gate (for ML features only)
    market_freshness_gate = MarketPriceFreshnessGate(config)
    market_freshness_report = market_freshness_gate.evaluate_and_refresh(
        state.market_prices,
        config.report_date,
        state.edgar_fetcher,
        required_tickers=["NVDA", "^SOX", "^GSPC", "AMD", "AVGO", "INTC", "QCOM", "MRVL"],
    )
    if market_freshness_report.status == "blocked_nvda_stale":
        state.ml_v2_result = MLV2Result.skipped("nvda_market_data_stale")
    else:
        # 2. Analyst estimates — DECISION-TIME OVERLAY ONLY (NOT a learned feature)
        analyst_fetcher = AnalystEstimateFetcher(config)
        analyst_snapshot = analyst_fetcher.fetch_snapshot("NVDA", config.report_date)
        # Note: AnalystEstimateFetcher has no `to_features()` method.
        # The snapshot is consumed only by MLDecisionEngine.compare_to_analyst_overlay()
        # to produce a comparator narrative for the report.

        # 3. Market features (price-derived; excludes any tickers stale per gate)
        market_builder = MarketFeatureBuilder(config)
        market_features = market_builder.compute_features(
            state.market_prices,
            as_of_dates=quarterly_filing_dates,
            excluded_tickers=market_freshness_report.tickers_excluded_stale,
        )

        # 4. Build quarterly panel — fundamentals + market + NLP only.
        # NO analyst features in the panel.
        panel_builder = QuarterlyPanelBuilder(config)
        panel = panel_builder.build_panel(
            state.metrics, state.nlp_features, market_features, state.market_prices
        )

        # 5. ML decision engine: walk-forward, McNemar significance, ablation
        engine = MLDecisionEngine(config)
        wf_results = {
            target: engine.walk_forward(panel, target, ml_config.feature_groups["full_no_analyst"], ml_config)
            for target in ["rev_growth_quarterly_YoY", "rev_growth_annual", "excess_return_direction"]
        }
        # Per-target McNemar p computed inside walk_forward()
        ablation = engine.compute_ablation(panel)

        # 6. ML adjustment — annual ML prediction only, no analyst features.
        # If annual N_OOS < 25 or tier == diagnostic, fallback applies (no target-price adjustment).
        dcf_callable = build_dcf_callable(state.valuation_inputs)
        ml_adjustment = engine.compute_ml_adjustment(
            decision_row=panel.iloc[-1],          # latest quarter, fundamentals + market + NLP only
            wf_annual=wf_results["rev_growth_annual"],
            wf_quarterly=wf_results["rev_growth_quarterly_YoY"],   # used for empirical residual band
            dcf_inputs=state.valuation_inputs,
            dcf_callable=dcf_callable,
            ml_config=ml_config,
        )

        # 7. Analyst overlay (comparator only, NOT injected into ml_adjustment)
        analyst_overlay = engine.compare_to_analyst_overlay(
            ml_prediction=ml_adjustment.ml_growth_y1,
            analyst_snapshot=analyst_snapshot,
        )
        # analyst_overlay is reported alongside ml_adjustment but does not
        # arithmetically modify the target price.

        success_criteria = engine.evaluate_success_criteria(ml_adjustment, ablation)

        state.ml_v2_result = MLV2Result(
            wf_results=wf_results,
            ablation=ablation,
            ml_adjustment=ml_adjustment,
            analyst_overlay=analyst_overlay,
            market_freshness_report=market_freshness_report,
            success_criteria=success_criteria,
        )

# Valuation stage uses (a) the v1 peer-financials freshness gate and
# (b) the v2 ml_adjustment for target-price modification.
if step in ("all", "valuation"):
    peer_fin_freshness_gate = PeerFinancialsFreshnessGate(config)
    peer_fin_report = peer_fin_freshness_gate.evaluate_and_refresh(
        state.peer_financials, config.report_date, state.edgar_fetcher
    )
    state.peer_financials_freshness_report = peer_fin_report
    # ... existing v1 valuation logic (peer multiples, scenarios, etc.)

# Pass ML adjustment to final recommendation builder
recommendation = valuation.build_final_recommendation(
    ...,
    ml_adjustment=state.ml_v2_result.ml_adjustment if state.ml_v2_result else None,
    analyst_overlay=state.ml_v2_result.analyst_overlay if state.ml_v2_result else None,
)
```

### Key Differences from the Old Block

| Old | New |
|---|---|
| Single `PeerFreshnessGate(config)` | `MarketPriceFreshnessGate` (in ml_v2) + `PeerFinancialsFreshnessGate` (in valuation) |
| `excluded_peers=freshness_report.peers_excluded_stale` | `excluded_tickers=market_freshness_report.tickers_excluded_stale` |
| `decision_row = build_decision_row(panel, analyst_snapshot)` | `decision_row = panel.iloc[-1]` (no analyst injection) |
| `feature_groups["full"]` | `feature_groups["full_no_analyst"]` |
| `compute_ml_adjustment(decision_row, ...)` with implicit analyst use | Explicit `wf_annual`, `wf_quarterly`, no analyst input |
| Analyst snapshot fed into ML adjustment math | Analyst snapshot used only by `compare_to_analyst_overlay` for narrative |
```

---

## Data Flow

```
                        [run_pipeline.py: ml_v2 stage]
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
PeerFreshnessGate          AnalystEstimateFetcher       MarketFeatureBuilder
        │                           │                           │
        │ excluded_peers            │ snapshot (latest only)    │ price-derived
        │                           │                           │ features
        ▼                           │                           ▼
        └────────────┐              │            ┌──────────────┘
                     ▼              │            ▼
              QuarterlyPanelBuilder │     market features
                     │              │     (per as_of_date)
                     │              │            │
                     ▼              │            │
        ml_quarterly_panel.csv      │            │
        (44 rows × ~25 cols,        │            │
        no analyst features)        │            │
                     │              │            │
                     └──────────────┼────────────┘
                                    ▼
                          MLDecisionEngine
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
walk_forward (3 targets)      compute_ablation              compute_ml_adjustment
        │                           │                           │
        │ WF metrics                │ A/B/C/D groups            │ ml_growth, weight,
        │ McNemar p                 │ marginal lift             │ adjusted_target,
        │                           │                           │ residual band,
        ▼                           ▼                           │ success_criteria
significance_tests.csv     ml_ablation_results.csv             │
                                                                ▼
                                                       MLAdjustment object
                                                                │
                                                                ▼
                                              valuation.build_final_recommendation
                                              (ML-adjusted target, scorecard row)
                                                                │
                                                                ▼
                                              report (with appendix methodology
                                              and ablation/CI exhibits)
```

---

## File Outputs

| File | Content |
|---|---|
| `data/raw/analyst_estimates_<YYYYMMDD>.json` | Cached yfinance snapshot |
| `data/processed/ml_quarterly_panel.csv` | Dense quarterly feature/target matrix |
| `data/processed/ml_significance_tests.csv` | Per-target McNemar test results: paired contingency tables (a, c), p-values, MAE-improvement-vs-naive |
| `data/processed/ml_ablation_results.csv` | A/B/C/D ablation table |
| `data/processed/ml_walk_forward_predictions.csv` | All OOS predictions per target |
| `data/processed/ml_residual_band_bootstrap.csv` | 1,000 bootstrapped target prices used to compute the empirical residual band |
| `outputs/figures/ml_ablation.png` | Ablation bar chart |
| `outputs/figures/ml_target_price_distribution.png` | Empirical residual band figure (5th–95th percentile target prices) |
| `outputs/figures/ml_walk_forward_timeseries.png` | OOS predicted vs actual |
| `outputs/peer_freshness_report.json` | Freshness gate output |

---

## Audit Integration

`audit_utils.py` adds the following checks to `audit_status.json`:

- `ml_v2_panel_size`: number of rows in ml_quarterly_panel.csv
- `ml_v2_panel_min_target_count`: rows with non-null primary target
- `ml_v2_walk_forward_n_oos`: per-target OOS count
- `ml_v2_directional_accuracy`: per-target
- `ml_v2_mae_improvement_pct`: per-target MAE improvement vs best naive baseline
- `ml_v2_mcnemar_p`: per-target McNemar p-value (NOT aggregate binomial)
- `ml_v2_confidence_tier_initial`: per-target tier before stability gates
- `ml_v2_confidence_tier_final`: per-target tier after Req 16 downgrades
- `ml_v2_ljung_box_p_lag1`, `ml_v2_ljung_box_p_lag2`, `ml_v2_ljung_box_p_lag4`: per-target
- `ml_v2_regime_mae_ratio`: recent-half MAE / earlier-half MAE
- `ml_v2_recent_window_mae`, `ml_v2_early_window_mae`: per-target
- `ml_v2_residual_scaling_method`: A | B | C (per Req 9.1a)
- `ml_v2_target_price_dcf_only`, `ml_v2_target_price_adjusted`
- `ml_v2_target_price_residual_band_low`, `ml_v2_target_price_residual_band_high`, `ml_v2_target_price_residual_band_n`
- `ml_v2_band_outcome`: `confidence_improving` | `risk_revealing` | `neutral`
- `ml_v2_primary_success_met`: bool (Req 11.1 primary criterion)
- `ml_v2_secondary_success_met`: bool (Req 11.2)
- `ml_v2_tertiary_findings`: list (Req 11.3, descriptive only)
- `ml_v2_decision_driving`: bool (true only if primary success met)
- `ml_v2_pre_registration_valid`: bool (from Req 12.5 manifest)
- `ml_v2_analyst_overlay_status`: `available` | `unavailable` | `agree` | `disagree_<magnitude>`
- `market_price_freshness_status`, `excluded_tickers_market_price_stale`
- `peer_financials_freshness_status`, `excluded_peers_financials_stale`
- `nlp_proxy_qa_complete`: bool (Req 14.4)
- `nlp_proxy_relevant_count`: count of `relevant_mda` + `partial_relevant` labels in qa CSV

The `recommendation_eligibility` gate is **unchanged**: the v1 three-gate architecture still gates the formal rating. The ML v2 layer adjusts magnitude/direction within an already-eligible recommendation; it does not bypass the data validation gate.

---

## Risk Mitigation (Revised)

| Risk | Mitigation |
|---|---|
| Quarterly XBRL gaps for capex/OCF | YTD-derivation rule (Req 1.4) + LOCF for ≤1Q with flag |
| Statistical significance with small N | Pre-registered tier thresholds (Req 7); McNemar paired test (not aggregate binomial); MAE economic-error gate |
| Multiple comparisons across models | Pre-register ElasticNet as primary (Req 6.6); Ridge/Lasso reported as sensitivity only |
| Multiple comparisons across success criteria | Hierarchical success structure (Req 11): primary/secondary/tertiary, only primary counts |
| GBM overfitting | Excluded entirely (Req 6.5) |
| Lookahead in tertiary target | Replaced with point-in-time SPX comparison (Req 6.3) |
| Lookahead in walk-forward (annual target) | Horizon-aware embargo (Req 6.4) |
| Analyst features as learned ML feature (degenerate) | Reclassified as decision-time **overlay** only (Req 2); never enters ML training |
| Quarterly residual ≠ annual residual scale mismatch | Explicit scaling rule (Req 9.1a): default Method A = 4Q rolling aggregation |
| DCF horizon mismatch (1Y forecast → 10Y CAGR) | Horizon-mapped adjustment: full Y1, half Y2, zero Y3+, zero terminal (Req 8) |
| Annual target underpowered (N_OOS < 25) | Explicit fallback (Req 8.5): no target-price adjustment, residual band only |
| Directional accuracy trivially good for high-growth target | MAE-improvement-over-naive gate added to tier promotion (Req 7) |
| Residual autocorrelation / regime shift | Ljung-Box + regime split + rolling MAE → automatic tier downgrade (Req 16) |
| NLP coverage without quality | Manual QA on 10 proxy samples + audit gate on `nlp_proxy_qa.csv` completion (Req 14) |
| Stale peer data | Two independent gates: market-price freshness (ML, Req 5.1–5.5) and peer-financials freshness (valuation, Req 5.6–5.10) |
| Spurious "decision-driving" claim | Hierarchical pre-registered success (Req 11); honest "not decision-driving" outcome (Req 11.5) |
| Selection bias in tier promotion | Pre-registered MLConfig with strengthened provenance manifest (Req 12.5) |
| V2 implementation breaks v1 report | MVP path with v1 fallback (Req 15, mvp_path.md) |
| Reader confusion about confidence | Bootstrap target-price CI with clear DCF-only vs ML-augmented comparison (Req 9) |
