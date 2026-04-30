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
