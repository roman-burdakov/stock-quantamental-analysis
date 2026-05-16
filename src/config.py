"""
NVDA Quantamental Engine — Configuration and Data Models.

All parameters, dates, thresholds, data models, keyword dictionaries,
and default scenario assumptions live here. Switching companies requires
only config changes (Req 13.5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root (no-op if file missing)
load_dotenv()


# ---------------------------------------------------------------------------
# Valuation helpers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioAssumptions:
    """Bear / base / bull scenario parameters for DCF valuation."""
    name: str
    probability: float
    revenue_cagr: float
    fcf_margin_start: float
    fcf_margin_terminal: float
    terminal_growth: float
    sbc_treatment: str  # "included_in_fcf" or "adjusted_out"
    analyst_notes: str


@dataclass
class RecommendationThresholds:
    """Thresholds governing Buy / Hold / Sell recommendation logic."""
    buy_min_upside: float = 0.15
    buy_max_bear_downside: float = -0.25
    sell_min_downside: float = -0.20


# ---------------------------------------------------------------------------
# v2 ML Layer Configuration (Reqs 6, 7, 8, 9, 12, Milestone 0)
# ---------------------------------------------------------------------------

@dataclass
class MLConfig:
    """Pre-registered v2 ML configuration.

    This dataclass is the single source of truth for v2 ML decisions.
    It MUST be committed to git BEFORE running walk-forward on the
    latest data (Req 12.2). The pipeline records the git SHA, dirty-tree
    status, and a hash of the canonical-JSON-serialised MLConfig at
    ml_v2 stage entry; the audit verifies that
    `mlconfig_committed_pre_walk_forward` is True for any tier promotion
    above diagnostic.

    No post-hoc tuning. Hyperparameter search (if any) is run on a held-out
    development period and frozen here before walk-forward execution.

    Naming reconciliation (Reqs 6, 8, 9):
    - Primary target = `rev_growth_quarterly_YoY` (used for empirical
      residual band, MVP-Essential confidence-band outcome).
    - Secondary target = `rev_growth_annual_FY` (used for DCF target-price
      adjustment ONLY when annual N_OOS >= 25 and tier >= contributing).
    - Tertiary target = `excess_return_12m_vs_spx_direction` (sensitivity
      only; not a gate).

    Analyst data is NEVER an ML feature (Req 2). It is consumed by
    `compare_to_analyst_overlay` for the report comparator narrative.
    The `analyst_overlay_fields` list below is NOT a feature group; it
    enumerates which AnalystSnapshot fields are surfaced to the report.
    """

    # --- Feature groups (Req 1, 3, 4) ---
    # Keys: "fundamentals", "market", "nlp", "full_no_analyst".
    # Values are lists of column names in ml_quarterly_panel.csv.
    # `full_no_analyst` is the union of fundamentals + market + nlp; it is
    # the feature set used by every walk-forward and ablation. NO analyst
    # group exists in feature_groups.
    feature_groups: dict[str, list[str]] = field(default_factory=lambda: {
        "fundamentals": [
            "gross_margin",
            "operating_margin",
            "fcf_margin",
            "revenue_growth_QoQ",
            "revenue_growth_YoY",
            "rd_intensity",
            "capex_intensity",
            "operating_leverage",
        ],
        "market": [
            "nvda_return_3m",
            "nvda_return_12m",
            "nvda_excess_return_3m_vs_sox",
            "nvda_excess_return_12m_vs_sox",
            "nvda_excess_return_3m_vs_spx",
            "nvda_excess_return_12m_vs_spx",
            "nvda_volatility_60d",
            "nvda_beta_252d_vs_sox",
        ],
        "nlp": [
            "narrative_drift_tfidf",
            "sentiment_polarity",
            "sentiment_delta",
            "mda_length_delta",
            "keyword_ai_accelerated_computing",
            "keyword_data_center",
            "keyword_competition",
            "keyword_supply_constraints",
        ],
        # full_no_analyst is computed at runtime from the other three groups
        # (see MLConfig.full_feature_columns()).
    })

    # --- Analyst overlay fields (Req 2; NOT an ML feature group) ---
    # These are surfaced to the report comparator but never enter the
    # ML training matrix or any walk-forward. The list documents which
    # AnalystSnapshot attributes the comparator uses.
    analyst_overlay_fields: list[str] = field(default_factory=lambda: [
        "revenue_growth_curr_yr",
        "revenue_growth_next_yr",
        "eps_estimate_curr_yr",
        "eps_estimate_next_yr",
        "eps_revision_30d",
        "eps_revision_60d",
        "eps_revision_90d",
        "n_analysts",
        "recommendation_mean",
        "recommendation_delta_30d",
    ])

    # --- Target definitions (Req 6) ---
    # Pre-registered target identifiers. Implementation in
    # quarterly_panel.QuarterlyPanelBuilder._attach_targets must produce
    # columns named exactly as listed below.
    target_definitions: dict[str, str] = field(default_factory=lambda: {
        "primary": "rev_growth_quarterly_YoY",
        "secondary": "rev_growth_annual_FY",
        "tertiary": "excess_return_12m_vs_spx_direction",
    })
    target_definitions_version: str = "1.0.0"  # bump on any target logic change

    # --- Walk-forward parameters per target (Req 6.4) ---
    # Each target has horizon-aware embargo: embargo equals target horizon
    # in quarters to prevent target-window overlap between train and test.
    walk_forward_params: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "rev_growth_quarterly_YoY": {
            "min_train_quarters": 12,
            "embargo_quarters": 1,
            "target_horizon_quarters": 1,
        },
        "rev_growth_annual_FY": {
            "min_train_quarters": 16,
            "embargo_quarters": 4,
            "target_horizon_quarters": 4,
        },
        "excess_return_12m_vs_spx_direction": {
            "min_train_quarters": 16,
            "embargo_quarters": 4,
            "target_horizon_quarters": 4,
        },
    })

    # --- Models (Req 6.5, 6.6) ---
    # ElasticNet is the pre-registered primary model used for the published
    # ML adjustment and empirical residual band. Ridge and LassoLars are
    # reported only as appendix sensitivities. GradientBoosting is excluded
    # entirely (overfitting risk at N approx 30).
    primary_model: str = "elasticnet"
    sensitivity_models: list[str] = field(
        default_factory=lambda: ["ridge", "lassolars"]
    )
    model_hyperparameters: dict[str, dict] = field(default_factory=lambda: {
        "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 10000},
        "ridge": {"alpha": 1.0, "max_iter": 10000},
        "lassolars": {"alpha": 0.01, "max_iter": 10000, "normalize": False},
    })

    # --- Naive baselines per target (Req 6.7) ---
    # Analyst consensus is NOT a baseline: historical revisions are
    # unavailable from public sources. See Req 2 / Req 10 group E.
    naive_baselines: dict[str, list[str]] = field(default_factory=lambda: {
        "rev_growth_quarterly_YoY": [
            "persistence",          # y_{t+1} = y_t
            "seasonal_naive",       # y_{t+1} = y_{t-3} (same quarter last year)
            "trailing_4q_mean",
        ],
        "rev_growth_annual_FY": [
            "trailing_3y_mean",
            "annual_persistence",   # next FY growth = current FY growth
        ],
        "excess_return_12m_vs_spx_direction": [
            "coin_flip",            # 50-50
            "always_positive",      # base rate from history
        ],
    })

    # --- Tier thresholds (Req 7) ---
    # Tier promotion gates. All conditions must be satisfied for the
    # named tier; otherwise the tier defaults down. McNemar paired
    # significance test (NOT aggregate binomial) per Req 7.
    tier_thresholds: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "contributing": {
            "min_n_oos": 25,
            "min_directional_accuracy": 0.55,
            "max_mcnemar_p": 0.20,
            "min_r2": 0.0,
            "min_auc": 0.55,
            "min_mae_improvement_pct": 10.0,
        },
        "high_confidence": {
            "min_n_oos": 30,
            "min_directional_accuracy": 0.60,
            "max_mcnemar_p": 0.10,
            "min_r2": 0.05,
            "min_auc": 0.60,
            "min_mae_improvement_pct": 20.0,
        },
    })

    # --- Adjustment weights (Req 8.2) ---
    # Single source of truth on tier -> weight mapping.
    # Used in horizon-mapped DCF adjustment per Req 8.1.
    adjustment_weights: dict[str, float] = field(default_factory=lambda: {
        "diagnostic": 0.0,
        "contributing": 0.20,
        "high_confidence": 0.35,
    })

    # --- N_OOS fallback for target-price adjustment (Req 8.5) ---
    # If annual target N_OOS < this threshold, ML does NOT adjust target
    # price; only the empirical residual band on the primary quarterly
    # target is reported.
    min_n_oos_for_target_price_adjust: int = 25

    # --- Empirical residual band (Req 9) ---
    # Pre-registered scaling method (Req 9.1a):
    #   "A" = rolling 4-quarter aggregation (preferred default)
    #   "B" = conservative 0.5x scaling
    #   "C" = decoupled reporting (no DCF perturbation)
    residual_scaling_method: str = "A"
    bootstrap_n: int = 1000
    bootstrap_min_n_residuals: int = 20  # below this, no band is produced

    # --- Band-outcome thresholds (Req 11.1, mvp_path.md) ---
    # Confidence-improving = band narrows by >= this fraction of DCF-only width.
    # Risk-revealing = band widens by >= this fraction.
    band_narrows_threshold_pct: float = 10.0
    band_widens_threshold_pct: float = 10.0

    # --- Model stability gates (Req 16) ---
    # Automatic tier downgrades from residual diagnostics.
    ljung_box_lags: list[int] = field(default_factory=lambda: [1, 2, 4])
    ljung_box_alpha: float = 0.05
    regime_split_max_mae_ratio: float = 1.30   # recent_half / earlier_half
    rolling_window_size: int = 8
    rolling_window_max_mae_ratio: float = 1.50  # recent_8 / earliest_8

    # --- NLP coverage / QA (Req 3.6, 14) ---
    nlp_coverage_active_threshold_pct: float = 60.0
    nlp_proxy_qa_sample_size: int = 10
    nlp_proxy_qa_min_relevant: int = 8
    nlp_proxy_low_quality_max_rate_pct: float = 30.0

    # --- Freshness gates (Req 5) ---
    market_price_max_staleness_trading_days: int = 5
    peer_financials_max_staleness_days: int = 30

    # --- Reproducibility ---
    random_seed: int = 42

    def full_feature_columns(self) -> list[str]:
        """Return the union of fundamentals + market + nlp groups.

        This is the feature set used by every walk-forward and the ablation
        full-model row. Analyst overlay fields are NOT included here and
        never enter the ML training matrix (Req 2).
        """
        return (
            self.feature_groups.get("fundamentals", [])
            + self.feature_groups.get("market", [])
            + self.feature_groups.get("nlp", [])
        )

    def canonical_dict(self) -> dict:
        """Return a deterministic dict representation for hashing.

        Used by the pre-registration provenance manifest (Req 12.5) to
        compute a stable SHA256 of the MLConfig contents.
        """
        return {
            "feature_groups": {
                k: sorted(v) for k, v in self.feature_groups.items()
            },
            "analyst_overlay_fields": sorted(self.analyst_overlay_fields),
            "target_definitions": dict(self.target_definitions),
            "target_definitions_version": self.target_definitions_version,
            "walk_forward_params": {
                k: dict(v) for k, v in self.walk_forward_params.items()
            },
            "primary_model": self.primary_model,
            "sensitivity_models": list(self.sensitivity_models),
            "model_hyperparameters": {
                k: dict(v) for k, v in self.model_hyperparameters.items()
            },
            "naive_baselines": {
                k: list(v) for k, v in self.naive_baselines.items()
            },
            "tier_thresholds": {
                k: dict(v) for k, v in self.tier_thresholds.items()
            },
            "adjustment_weights": dict(self.adjustment_weights),
            "min_n_oos_for_target_price_adjust": self.min_n_oos_for_target_price_adjust,
            "residual_scaling_method": self.residual_scaling_method,
            "bootstrap_n": self.bootstrap_n,
            "bootstrap_min_n_residuals": self.bootstrap_min_n_residuals,
            "band_narrows_threshold_pct": self.band_narrows_threshold_pct,
            "band_widens_threshold_pct": self.band_widens_threshold_pct,
            "ljung_box_lags": list(self.ljung_box_lags),
            "ljung_box_alpha": self.ljung_box_alpha,
            "regime_split_max_mae_ratio": self.regime_split_max_mae_ratio,
            "rolling_window_size": self.rolling_window_size,
            "rolling_window_max_mae_ratio": self.rolling_window_max_mae_ratio,
            "nlp_coverage_active_threshold_pct": self.nlp_coverage_active_threshold_pct,
            "nlp_proxy_qa_sample_size": self.nlp_proxy_qa_sample_size,
            "nlp_proxy_qa_min_relevant": self.nlp_proxy_qa_min_relevant,
            "nlp_proxy_low_quality_max_rate_pct": self.nlp_proxy_low_quality_max_rate_pct,
            "market_price_max_staleness_trading_days": self.market_price_max_staleness_trading_days,
            "peer_financials_max_staleness_days": self.peer_financials_max_staleness_days,
            "random_seed": self.random_seed,
        }


# ---------------------------------------------------------------------------
# Validation and eligibility enums / data models (Reqs 14.1, 14.2, 17)
# ---------------------------------------------------------------------------

class DataQualityStatus(Enum):
    """Pipeline-wide data validation state (Req 14.1)."""
    PASS = "pass"                              # All core metrics within tolerance
    PASS_WITH_WARNINGS = "pass_with_warnings"  # Non-critical deviations exist
    DATA_BLOCKED = "data_blocked"              # One or more critical metrics fail


class ComponentStatusEnum(Enum):
    """Per-component quality state (Req 17.4)."""
    USABLE = "usable"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"


class ReportMode(Enum):
    """Report generation mode (Req 17.7)."""
    FORMAL_RATING = "formal_rating"
    DIAGNOSTIC_NOT_RATED = "diagnostic_not_rated"
    FAILED = "failed"


@dataclass
class ValidatedMetric:
    """Cross-check of a parsed metric against a published reference value (Req 14.2)."""
    metric_name: str
    fiscal_year: int
    parsed_value: float | None
    published_value: float | None
    diff_pct: float | None          # abs((parsed - published) / published) * 100
    tolerance: float                # default 1.0 (percent)
    status: str                     # "pass", "warn", "fail", "missing"
    severity: str                   # "critical", "major", "minor"
    blocker: bool                   # True if this failure blocks recommendation


@dataclass
class ComponentStatus:
    """Status of a pipeline component for the recommendation scorecard."""
    component_name: str             # e.g. "data_validation", "ml_signal", "nlp_signal"
    status: ComponentStatusEnum     # usable / diagnostic_only / unavailable / blocked
    reason: str                     # human-readable explanation
    details: dict | None = None     # optional structured details


@dataclass
class RecommendationStatus:
    """Output of the recommendation eligibility gate (Req 17)."""
    eligibility_status: ReportMode  # formal_rating / diagnostic_not_rated / failed
    data_quality_status: DataQualityStatus
    component_statuses: list[ComponentStatus]
    blocking_issues: list[str]      # human-readable list of blockers
    timestamp: str


@dataclass
class ChartStatus:
    """Whether a chart should be rendered, suppressed, or flagged (Req 19.1)."""
    chart_name: str
    renderable: bool
    reason: str                     # "ok", "all_categories_other", "data_blocked", etc.
    fallback_message: str | None = None  # message to show if chart suppressed


@dataclass
class AuditStatus:
    """Machine-readable audit output (Req 22)."""
    overall_status: str             # "pass" or "fail"
    data_quality_status: str
    recommendation_eligibility: str
    component_statuses: dict[str, str]
    checks_run: int
    checks_passed: int
    checks_failed: int
    blocking_issues: list[str]
    failure_details: list[str]
    timestamp: str


# ---------------------------------------------------------------------------
# Core engine configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Master configuration for the quantamental pipeline."""

    # Company identifiers
    ticker: str = "NVDA"
    cik: str = "0001045810"
    company_name: str = "NVIDIA Corporation"

    # Analyst / author
    analyst_name: str = "Roman Burdakov"
    analyst_title: str = "Analyst"

    # Fiscal scope
    start_fiscal_year: int = 2016
    end_fiscal_year: int = 2026

    # Frozen analysis dates
    report_date: str = "2026-05-01"
    price_date: str = "2026-05-01"

    # Output
    output_format: str = "pdf"  # "pdf", "html", "both"
    target_horizon: str = "intrinsic_value_as_of_report_date"

    # --- Peers ---
    core_semiconductor_peers: list[str] = field(
        default_factory=lambda: ["AMD", "AVGO", "INTC", "QCOM", "MRVL"]
    )
    infrastructure_peers: list[str] = field(
        default_factory=lambda: ["TSM", "ASML"]
    )
    ai_capex_context: list[str] = field(
        default_factory=lambda: ["MSFT", "AMZN", "GOOGL", "META"]
    )
    peer_justifications: dict[str, str] = field(default_factory=lambda: {
        "AMD": "Direct GPU/CPU competitor in data-center accelerators",
        "AVGO": "Broadcom — networking/custom-silicon peer, AI infrastructure",
        "INTC": "Legacy CPU incumbent, re-entering discrete GPU/accelerator market",
        "QCOM": "Fabless peer with AI-edge inference exposure",
        "MRVL": "Data-center semiconductor peer (custom silicon, DPUs)",
        "TSM": "Sole leading-edge foundry; NVDA manufacturing dependency",
        "ASML": "Lithography monopoly; capex-cycle proxy for semiconductor supply",
        "MSFT": "Hyperscaler AI capex — Azure/OpenAI demand signal for NVDA GPUs",
        "AMZN": "Hyperscaler AI capex — AWS Trainium/Inferentia + NVDA GPU buyer",
        "GOOGL": "Hyperscaler AI capex — TPU in-house vs NVDA GPU demand signal",
        "META": "Hyperscaler AI capex — large NVDA GPU buyer for LLM training",
    })
    peer_staleness_threshold_days: int = 90

    # --- NLP ---
    tracked_themes: list[str] = field(default_factory=lambda: [
        "ai_accelerated_computing",
        "data_center",
        "export_controls_china",
        "supply_constraints",
        "competition",
        "customer_concentration",
        "gaming_cyclicality",
        "margin_pricing_pressure",
        "inventory_demand_cyclicality",
    ])
    use_embeddings: bool = False
    use_topic_model: bool = False
    use_change_point_detection: bool = False

    # --- ML ---
    ml_target: str = "revenue_growth_yoy"
    min_train_years: int = 3
    primary_model: str = "elasticnet"
    use_tree_model: bool = False
    tree_model_type: str = "gradient_boosting"  # "random_forest" or "gradient_boosting"
    baselines: list[str] = field(
        default_factory=lambda: [
            "last_period",
            "trailing_4q_avg",
            "three_year_avg",
            "linear_trend",
        ]
    )

    # --- v2 ML Layer (Reqs 6-12, 16; pre-registered per Milestone 0) ---
    mlconfig: MLConfig = field(default_factory=MLConfig)

    # --- Valuation ---
    wacc: float = 0.10
    terminal_growth: float = 0.03
    projection_years: int = 10
    scenarios: dict[str, ScenarioAssumptions] = field(default_factory=dict)
    recommendation_thresholds: RecommendationThresholds = field(
        default_factory=RecommendationThresholds
    )

    # --- Reverse-DCF grid defaults (Req 9.7, 16.3) ---
    reverse_dcf_cagr_range: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )
    reverse_dcf_margin_range: list[float] = field(
        default_factory=lambda: [0.20, 0.25, 0.30, 0.35, 0.40]
    )

    # --- Validation (Req 14, 15) ---
    core_validation_metrics: list[str] = field(default_factory=lambda: [
        "revenue", "gross_profit", "operating_income", "net_income",
        "operating_cash_flow", "capex", "FCF", "cash_and_securities",
        "total_debt", "diluted_shares", "diluted_EPS", "r_and_d",
    ])
    validation_tolerance_pct: float = 1.0
    min_nlp_char_mda: int = 500
    min_nlp_char_risk: int = 300
    max_dcf_price_divergence_pct: float = 50.0

    # --- Fallbacks (set per-ticker if XBRL extraction fails) ---
    fallback_diluted_shares: int | None = None  # Set per-ticker if XBRL extraction fails
    fallback_net_cash: int | None = None        # Set per-ticker if XBRL extraction fails

    # --- Split adjustment ---
    # NVIDIA 10:1 split (June 2024). Override for other tickers.
    split_history: list[dict] = field(default_factory=lambda: [
        {
            "date": "2024-06-10",
            "ratio": 10,
            "description": "NVIDIA 10-for-1 stock split",
        },
    ])

    # --- Paths ---
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    interim_dir: Path = field(default_factory=lambda: Path("data/interim"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    templates_dir: Path = field(default_factory=lambda: Path("src/templates"))
    fixtures_dir: Path = field(default_factory=lambda: Path("tests/fixtures"))
    provenance_log: Path = field(
        default_factory=lambda: Path("data/raw/provenance_log.jsonl")
    )

    # --- API keys (loaded from .env) ---
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))

    # --- SEC / runtime ---
    sec_user_agent: str = field(default_factory=lambda: os.getenv("SEC_USER_AGENT", ""))
    sec_rate_limit_rps: int = 10
    sec_max_retries: int = 3
    force_refresh: bool = False
    skip_tests: bool = False
    step: str = "all"


# ---------------------------------------------------------------------------
# Data model dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FilingRecord:
    """Metadata for a single SEC filing."""
    ticker: str
    cik: str
    accession_number: str
    filing_date: str
    source_available_date: str  # = filing_date for SEC filings
    report_period: str
    form_type: str
    sec_url: str
    retrieval_timestamp: str


@dataclass
class FinancialFact:
    """A single XBRL-sourced financial fact.

    Enhanced with fiscal-period selection metadata (Req 15.1) and
    split-adjustment fields so that raw filing values are never overwritten.
    """
    ticker: str
    fiscal_period: str
    fiscal_year: int
    filing_date: str
    source_available_date: str
    accession_number: str
    metric_name: str
    value: Optional[float]
    unit: str
    form_type: str
    # --- Fiscal-period selection metadata (Req 15.1) ---
    fiscal_period_type: str = ""          # "annual", "quarterly", "ytd", "instant"
    period_start: Optional[str] = None    # ISO date string
    period_end: Optional[str] = None      # ISO date string
    duration_days: Optional[int] = None   # computed from period_start/period_end
    frame: Optional[str] = None           # SEC XBRL frame string (e.g., "CY2024")
    xbrl_concept: str = ""                # actual XBRL concept used
    selection_rank: int = 0               # 1 = preferred, 2 = fallback, etc.
    selection_reason: str = ""            # "10-K annual duration", "frame match", "fallback"
    # --- Split adjustment fields ---
    raw_value: Optional[float] = None     # value as reported in filing (never overwritten)
    adjusted_value: Optional[float] = None  # value after split adjustment
    adjustment_factor: float = 1.0        # cumulative split ratio (1.0 if no adjustment)
    adjustment_basis: str = "as_reported"  # "split_adjusted_to_current" or "as_reported"
    validation_basis: str = "as_reported"  # basis used for published-value comparison


@dataclass
class ComputedMetric:
    """A derived financial metric (ratio, growth rate, etc.)."""
    ticker: str
    fiscal_period: str
    filing_date: str
    source_available_date: str
    metric_name: str
    metric_value: Optional[float]
    source_accession: str
    unit: str


@dataclass
class TextSectionRecord:
    """An extracted narrative section from a filing."""
    accession_number: str
    form_type: str
    section_name: str
    text: str
    char_count: int
    filing_date: str
    source_available_date: str
    parse_status: str  # "success", "fallback", "missing"


@dataclass
class SegmentRevenueRecord:
    """A normalised segment/platform revenue line item."""
    ticker: str
    fiscal_period: str
    fiscal_year: int
    filing_date: str
    source_available_date: str
    source_accession: str
    original_label: str
    normalized_category: str
    value: float
    unit: str
    extraction_method: str  # "xbrl_dimension", "table_extraction", "manual"
    mapping_notes: str


@dataclass
class NLPFeatureRecord:
    """A single NLP-derived feature for a filing."""
    filing_date: str
    source_available_date: str
    source_accession: str
    section: str
    feature_type: str
    feature_name: str
    value: float
    prev_filing_date: Optional[str]


@dataclass
class FeatureTargetRecord:
    """One row of the ML feature/target matrix."""
    feature_period: str
    feature_available_date: str
    prediction_date: str
    target_period: str
    target_available_date: str
    target_name: str
    target_value: Optional[float]
    source_accessions: list[str] = field(default_factory=list)
    # Dynamic feature columns are added at runtime via dict / DataFrame.


@dataclass
class ValuationAssumption:
    """A single documented valuation assumption."""
    assumption_name: str
    value: float | str
    source: str  # "analyst_judgment", "historical_median", "config", etc.
    notes: str


@dataclass
class ExhibitRecord:
    """Metadata for a report exhibit (chart / table)."""
    exhibit_id: str
    title: str
    source_caption: str
    data_source: str
    date_range: str
    file_path: str
    exhibit_key: str = ""  # stable semantic key, e.g. "revenue_growth_trend"


@dataclass
class DataQualityIssue:
    """A logged data-quality concern."""
    category: str  # "missing_tag", "fallback_used", "validation_fail", "stale_data"
    detail: str
    filing_or_source: str
    severity: str  # "info", "warning", "error"


@dataclass
class Recommendation:
    """Final investment recommendation with full scorecard.

    Enhanced with eligibility status fields (Req 17.3).
    """
    rating: str
    current_price: float
    target_price: float
    upside_pct: float
    bear_value: float
    bear_probability: float
    base_value: float
    base_probability: float
    bull_value: float
    bull_probability: float
    expected_value: float
    what_must_be_true_buy: str
    what_must_be_true_hold: str
    what_must_be_true_sell: str
    scorecard: dict = field(default_factory=dict)
    target_horizon: str = "intrinsic_value_as_of_report_date"
    # --- Eligibility fields (Req 17.3) ---
    eligibility_status: str = "formal_rating"  # "formal_rating", "diagnostic_not_rated", "failed"
    not_rated_reason: str | None = None        # explanation if not rated
    sanity_check_notes: list[str] = field(default_factory=list)  # bridge explanations, grid warnings


@dataclass
class FinalRecommendation:
    """Canonical single source of truth for all final recommendation fields.

    Every artifact (PDF, HTML, Markdown, executive summary, manifest,
    audit_status, source_attribution, limitations) MUST consume this
    object. No other code path may independently derive rating/target/upside
    for a final artifact.
    """
    run_id: str
    ticker: str
    company_name: str
    report_date: str
    price_date: str
    current_price: float
    rating: str                          # "Buy", "Hold", "Sell", "Not Rated"
    rating_label: str                    # e.g. "Hold / Market Perform"
    intrinsic_value: float               # probability-weighted value
    target_price: float                  # = intrinsic_value
    upside_downside_pct: float           # (target - current) / current
    recommendation_rationale_short: str  # one-sentence
    recommendation_rationale_long: str   # full paragraph
    bear_value: float
    base_value: float
    bull_value: float
    bear_probability: float
    base_probability: float
    bull_probability: float
    probability_weighted_value: float
    rating_thresholds: dict = field(default_factory=dict)
    analyst_override_flag: bool = False
    analyst_override_reason: str = ""
    valuation_method: str = "probability_weighted_dcf"
    valuation_confidence: str = "medium"  # "high", "medium", "low"
    data_quality_status: str = "pass"
    report_mode: str = "formal_rating"
    package_status: str = "formal_rating_pass"
    active_exhibits: list[str] = field(default_factory=list)
    suppressed_exhibits: list[str] = field(default_factory=list)
    timestamp: str = ""
    what_must_be_true_buy: str = ""
    what_must_be_true_hold: str = ""
    what_must_be_true_sell: str = ""
    why_hold_not_sell: str = ""
    why_sell_if_sell: str = ""
    upgrade_triggers: list[str] = field(default_factory=list)
    downgrade_triggers: list[str] = field(default_factory=list)
    scorecard: dict = field(default_factory=dict)
    scorecard_status_matrix: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/manifest/audit consumption."""
        return {
            "run_id": self.run_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "report_date": self.report_date,
            "price_date": self.price_date,
            "current_price": self.current_price,
            "rating": self.rating,
            "rating_label": self.rating_label,
            "intrinsic_value": round(self.intrinsic_value, 2),
            "target_price": round(self.target_price, 2),
            "upside_downside_pct": round(self.upside_downside_pct, 4),
            "recommendation_rationale_short": self.recommendation_rationale_short,
            "bear_value": round(self.bear_value, 2),
            "base_value": round(self.base_value, 2),
            "bull_value": round(self.bull_value, 2),
            "bear_probability": self.bear_probability,
            "base_probability": self.base_probability,
            "bull_probability": self.bull_probability,
            "probability_weighted_value": round(self.probability_weighted_value, 2),
            "analyst_override_flag": self.analyst_override_flag,
            "analyst_override_reason": self.analyst_override_reason,
            "valuation_method": self.valuation_method,
            "valuation_confidence": self.valuation_confidence,
            "data_quality_status": self.data_quality_status,
            "report_mode": self.report_mode,
            "package_status": self.package_status,
            "active_exhibits": self.active_exhibits,
            "suppressed_exhibits": self.suppressed_exhibits,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Peer corporate events during the 10-year analysis window (Req 25.3)
# NVIDIA-specific peer events. Override for other tickers/sectors.
# ---------------------------------------------------------------------------

PEER_CORPORATE_EVENTS: dict[str, list[dict[str, str]]] = {
    "AMD": [
        {
            "date": "2022-02",
            "event": "Acquired Xilinx ($49B)",
            "impact": "Significant revenue mix change; added FPGA/adaptive computing segment. Post-acquisition financials not directly comparable to pre-2022 AMD.",
        },
    ],
    "AVGO": [
        {
            "date": "2018",
            "event": "Acquired CA Technologies",
            "impact": "Added enterprise software segment; changed revenue mix from pure semiconductor.",
        },
        {
            "date": "2019",
            "event": "Acquired Symantec Enterprise division",
            "impact": "Further expanded software segment; reduced semiconductor revenue share.",
        },
        {
            "date": "2023-11",
            "event": "Acquired VMware ($69B)",
            "impact": "Transformative acquisition roughly doubled revenue; post-close financials dominated by software. Pre- and post-VMware multiples are not directly comparable.",
        },
    ],
    "INTC": [
        {
            "date": "2022-10",
            "event": "Mobileye (MBLY) IPO / spin-off",
            "impact": "Separated autonomous-driving unit; reduced consolidated revenue and changed segment reporting.",
        },
        {
            "date": "2021–2024",
            "event": "Major restructuring under CEO Pat Gelsinger",
            "impact": "IDM 2.0 strategy, foundry services launch, significant capex increase and margin compression. Financial profile shifted materially during this period.",
        },
        {
            "date": "2024",
            "event": "Announced Altera spin-off",
            "impact": "Planned separation of programmable-logic (FPGA) business; may further change segment composition.",
        },
    ],
    "QCOM": [
        {
            "date": "2018",
            "event": "Hostile takeover attempt by Broadcom blocked by US government",
            "impact": "No structural change to financials, but strategic uncertainty during the period affected capital allocation and R&D priorities.",
        },
        {
            "date": "2021",
            "event": "Acquired Nuvia",
            "impact": "Added custom CPU core design capability; modest impact on financials but strategic shift toward custom Arm-based compute.",
        },
    ],
    "MRVL": [
        {
            "date": "2021-04",
            "event": "Acquired Inphi ($10B)",
            "impact": "Transformed revenue mix toward data-center interconnect and cloud; post-acquisition financials reflect a materially different business profile.",
        },
    ],
    "TSM": [],  # No major corporate events affecting comparability
    "ASML": [],  # No major corporate events affecting comparability
    "MSFT": [
        {
            "date": "2022",
            "event": "Acquired Nuance Communications",
            "impact": "Added healthcare AI capabilities; modest impact on overall financials given Microsoft's scale.",
        },
        {
            "date": "2023-10",
            "event": "Acquired Activision Blizzard ($69B)",
            "impact": "Largest gaming acquisition in history; added significant gaming revenue. Context peer — used for AI capex signal, not direct financial comparison.",
        },
    ],
    "AMZN": [
        {
            "date": "2022",
            "event": "Acquired MGM",
            "impact": "Added entertainment content; modest impact on overall financials. AWS growth significantly changed revenue mix over the analysis window, shifting Amazon from primarily retail to a cloud-infrastructure company.",
        },
    ],
    "GOOGL": [
        {
            "date": "2015",
            "event": "Restructured from Google to Alphabet",
            "impact": "Holding-company restructure improved segment transparency (Google Services, Google Cloud, Other Bets). No change to underlying business, but financial reporting changed.",
        },
        {
            "date": "2024",
            "event": "Antitrust rulings (DOJ search monopoly case)",
            "impact": "Potential structural remedies pending; no financial impact during analysis window but regulatory overhang affects forward comparability.",
        },
    ],
    "META": [
        {
            "date": "2021-10",
            "event": "Rebranded from Facebook to Meta Platforms",
            "impact": "Strategic pivot toward metaverse/Reality Labs; significant increase in R&D spending on non-core (non-advertising) initiatives.",
        },
        {
            "date": "2022-06",
            "event": "Ticker changed from FB to META",
            "impact": "Ticker change only; no financial impact. Historical data under FB ticker maps to the same entity.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Keyword dictionaries for 9 tracked themes (Req 7.2)
# NVIDIA-specific NLP keyword themes. Override for other tickers/sectors.
# ---------------------------------------------------------------------------

KEYWORD_DICTIONARIES: dict[str, list[str]] = {
    "ai_accelerated_computing": [
        "artificial intelligence", "AI", "accelerated computing",
        "deep learning", "machine learning", "neural network",
        "generative AI", "large language model", "LLM",
        "inference", "training", "GPU computing",
        "CUDA", "Tensor Core", "transformer",
        "foundation model", "AI factory", "accelerator",
    ],
    "data_center": [
        "data center", "datacenter", "cloud computing",
        "hyperscale", "hyperscaler", "cloud service provider",
        "DGX", "HGX", "InfiniBand", "NVLink",
        "networking", "Grace", "Hopper", "Blackwell",
        "sovereign AI", "enterprise AI",
    ],
    "export_controls_china": [
        "export control", "export restriction", "China",
        "PRC", "Entity List", "Bureau of Industry and Security",
        "BIS", "Commerce Department", "trade restriction",
        "sanctions", "geopolitical", "tariff",
        "Huawei", "restricted", "license requirement",
    ],
    "supply_constraints": [
        "supply constraint", "supply chain", "capacity",
        "lead time", "allocation", "shortage",
        "wafer", "CoWoS", "packaging",
        "foundry", "TSMC", "manufacturing",
        "production ramp", "yield", "procurement",
    ],
    "competition": [
        "competition", "competitive", "competitor",
        "AMD", "Intel", "custom silicon",
        "ASIC", "TPU", "Trainium",
        "Inferentia", "alternative", "market share",
        "switching cost", "moat", "ecosystem",
    ],
    "customer_concentration": [
        "customer concentration", "significant customer",
        "large customer", "major customer",
        "hyperscaler", "cloud provider",
        "top customer", "revenue concentration",
        "single customer", "key customer",
    ],
    "gaming_cyclicality": [
        "gaming", "GeForce", "RTX",
        "consumer", "desktop", "notebook",
        "cyclical", "seasonality", "channel inventory",
        "cryptocurrency", "crypto mining",
        "PC market", "add-in board",
    ],
    "margin_pricing_pressure": [
        "gross margin", "operating margin", "pricing",
        "average selling price", "ASP", "cost",
        "margin pressure", "margin expansion",
        "product mix", "premium pricing",
        "pricing power", "cost reduction",
    ],
    "inventory_demand_cyclicality": [
        "inventory", "demand", "backlog",
        "order", "book-to-bill", "channel",
        "sell-through", "sell-in", "deferred revenue",
        "reserve", "write-down", "excess inventory",
        "demand visibility", "cyclicality",
    ],
}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_default_config() -> EngineConfig:
    """Return default EngineConfig calibrated for NVIDIA (NVDA).

    Bear / base / bull scenarios, keyword dictionaries, peer corporate events,
    and split history are NVIDIA-specific. Override these fields when analyzing
    a different ticker.

    Validates that scenario probabilities sum to 1.0.
    """
    scenarios: dict[str, ScenarioAssumptions] = {
        "bear": ScenarioAssumptions(
            name="bear",
            probability=0.25,
            revenue_cagr=0.10,
            fcf_margin_start=0.30,
            fcf_margin_terminal=0.25,
            terminal_growth=0.025,
            sbc_treatment="included_in_fcf",
            analyst_notes=(
                "Export controls tighten further; hyperscaler capex "
                "decelerates; custom-silicon alternatives gain share."
            ),
        ),
        "base": ScenarioAssumptions(
            name="base",
            probability=0.50,
            revenue_cagr=0.20,
            fcf_margin_start=0.35,
            fcf_margin_terminal=0.30,
            terminal_growth=0.03,
            sbc_treatment="included_in_fcf",
            analyst_notes=(
                "AI infrastructure spend grows steadily; NVDA retains "
                "dominant GPU share; margins normalise from peak levels."
            ),
        ),
        "bull": ScenarioAssumptions(
            name="bull",
            probability=0.25,
            revenue_cagr=0.30,
            fcf_margin_start=0.40,
            fcf_margin_terminal=0.35,
            terminal_growth=0.035,
            sbc_treatment="included_in_fcf",
            analyst_notes=(
                "Sovereign-AI and enterprise adoption accelerate; "
                "networking/software attach rates expand TAM; "
                "competitive moat widens."
            ),
        ),
    }

    # Validate scenario probabilities sum to 1.0
    total_prob = sum(s.probability for s in scenarios.values())
    if abs(total_prob - 1.0) > 1e-6:
        import warnings
        warnings.warn(
            f"Scenario probabilities sum to {total_prob:.4f}, not 1.0. "
            "Expected value calculation will be distorted.",
            stacklevel=2,
        )

    return EngineConfig(scenarios=scenarios)
