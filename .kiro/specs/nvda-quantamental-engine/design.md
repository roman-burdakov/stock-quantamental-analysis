# Design Document: NVDA Quantamental Engine

## Overview

Python pipeline: SEC ingestion → XBRL parsing → **data validation gate** → segment normalization → financial metrics → filing text extraction → NLP narrative drift → ML driver model → DCF/reverse-DCF → **recommendation eligibility gate** → report → **audit consistency check**. Designed for MIT 15.C51: credible, reproducible, A+ quality.

**Post-mortem context:** The initial pipeline produced a Sell recommendation with −77.7% downside because XBRL parsing returned FY2025 revenue as $26.974B instead of the published $130.497B. This design revision adds three hard gates: (1) data validation gate after XBRL parsing, (2) recommendation eligibility gate before report generation, and (3) audit consistency check after report generation.

Core principles:
1. **source_available_date everywhere**: Every data row carries the date it became publicly available. Report uses only data where `source_available_date <= report_date`. ML validation uses `feature_available_date <= prediction_date`.
2. **Full provenance**: Every data point traces to a filing or API call.
3. **Parameterized config**: Switching tickers requires only config changes.
4. **Caching**: All downloads cached; reruns skip fetching unless `--force-refresh`.
5. **Report-first**: Pipeline ordered to produce a working report early.
6. **Fail-safe validation**: No formal recommendation without validated data. Garbage in → blocked, not garbage out.

## Architecture

```
src/config.py               → All parameters, dates, thresholds, data models
src/edgar_fetch.py          → SEC submissions, companyfacts, filing docs, market data
src/xbrl_parser.py          → Companyfacts → structured financials + fiscal-year selection
src/data_validation.py      → Published-value validation gate + DataQualityStatus
src/segment_revenue.py      → Segment/platform revenue normalization
src/filing_text_parser.py   → Filing HTML → section text (Item-number based)
src/financial_metrics.py    → XBRL data → ratios, growth, TTM, FCF margin
src/nlp_features.py         → Section text → TF-IDF similarity, keyword scores
src/ml_models.py            → Features → Ridge/ElasticNet, walk-forward, baselines
src/valuation.py            → DCF, reverse-DCF grid, peer multiples, scenarios
src/charts.py               → 6-8 required figures with source captions + chart gates
src/report_utils.py         → Jinja2 → Markdown + PDF/HTML, report mode support
src/audit_utils.py          → Attribution, prompt log, data quality, self-audit, consistency
scripts/run_pipeline.py     → One-command orchestration with gates
tests/fixtures/             → Offline test data + regression fixtures
```

### Pipeline Flow with Gates

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
│  Ingestion   │───▶│ XBRL Parsing │───▶│ DATA VALIDATION GATE│
│ (SEC, mkt)   │    │ (fiscal-year  │    │ (published values)  │
└─────────────┘    │  selection)   │    │ → DataQualityStatus  │
                   └──────────────┘    └──────────┬──────────┘
                                                   │
                          ┌────────────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │ Metrics / Segments / NLP / ML        │
        │ (each component tracks its own       │
        │  ComponentStatus)                    │
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │ Valuation (DCF, reverse-DCF, peers)  │
        │ (blocked if DATA_BLOCKED on inputs)  │
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │ RECOMMENDATION ELIGIBILITY GATE      │
        │ → RecommendationStatus               │
        │ → formal_rating / diagnostic / failed│
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │ Report Generation                    │
        │ (mode: formal_rating,                │
        │  diagnostic_not_rated, failed)       │
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │ AUDIT CONSISTENCY CHECK              │
        │ → audit_status.json                  │
        │ → cross-file contradiction detection │
        └─────────────────────────────────────┘
```

### Data Flow

| Stage | Output | Directory |
|-------|--------|-----------|
| Ingestion | Raw JSON, HTML, CSV | `data/raw/` |
| Parsing | Structured tables, section text | `data/interim/` |
| **Validation** | **data_quality_report.md, audit_status.json** | **`outputs/`** |
| Metrics + Segments + NLP | Metrics CSV, segment CSV, NLP CSV | `data/processed/` |
| ML | Feature/target matrix, model audit | `data/processed/`, `outputs/` |
| Valuation + Report | Report, figures, audit files | `outputs/` |
| **Audit** | **audit_status.json (final)** | **`outputs/`** |

### Point-in-Time Controls

Two modes:
1. **Current-date analysis** (report generation): Uses all data with `source_available_date <= report_date`.
2. **Historical ML validation** (walk-forward): Strict no-lookahead. Feature/target matrix enforces `feature_available_date <= prediction_date` AND `target_available_date > prediction_date` per row.

### Report Modes and Final Package Status

The report generator operates in one of three modes based on pipeline gate results:

| Mode | Trigger | Cover Page Rating | Valuation Label | Scorecard |
|------|---------|-------------------|-----------------|-----------|
| `formal_rating` | All blocking gates pass | Buy / Hold / Sell | Formal | Full with usable components |
| `diagnostic_not_rated` | One or more blocking gates fail | "Not Rated — Data Validation Required" | "Diagnostic only — do not use for recommendation" | Shows blocked/diagnostic statuses |
| `failed` | Pipeline error | "Report Generation Failed" | N/A | N/A |

Non-blocking diagnostic components (ML, NLP, segments, peers) do NOT trigger `diagnostic_not_rated`. They are excluded from score direction and disclosed, but the report can still carry a formal rating if all blocking gates pass.

Final package status (in audit_status.json):
- `formal_rating_pass`: formal rating issued, audit consistency passes, no contradictions
- `diagnostic_not_rated_pass`: Not Rated issued, blockers disclosed, audit consistency passes
- `failed`: contradictions found, or pipeline error — not submission-ready

### Two-Phase Audit

1. **Pre-report eligibility audit** (before report rendering): Evaluates DataQualityStatus, valuation input status, component statuses → determines RecommendationStatus and ReportMode.
2. **Post-report consistency audit** (after report rendering): Cross-checks limitations.md, data_quality_report.md, model_audit.md, and final report for contradictions. If contradictions found, pipeline either regenerates report with Audit Warnings section or marks final package as `failed`.

### Required Outputs

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

## Data Models

### Core Data Models (existing, enhanced)

```python
@dataclass
class FilingRecord:
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
    """Enhanced with fiscal-period selection metadata."""
    ticker: str
    fiscal_period: str
    fiscal_year: int
    filing_date: str
    source_available_date: str
    accession_number: str
    metric_name: str
    value: float | None
    unit: str
    form_type: str
    # NEW fields for fiscal-year selection correctness
    fiscal_period_type: str       # "annual", "quarterly", "ytd", "instant"
    period_start: str | None      # ISO date
    period_end: str | None        # ISO date
    duration_days: int | None     # computed from period_start/end
    frame: str | None             # SEC XBRL frame string (e.g., "CY2024")
    xbrl_concept: str             # actual XBRL concept used
    selection_rank: int           # 1 = preferred, 2 = fallback, etc.
    selection_reason: str         # "10-K annual duration", "frame match", "fallback"
    # Split adjustment fields
    raw_value: float | None       # value as reported in filing (never overwritten)
    adjusted_value: float | None  # value after split adjustment
    adjustment_factor: float      # cumulative split ratio (1.0 if no adjustment)
    adjustment_basis: str         # "split_adjusted_to_current" or "as_reported"
    validation_basis: str         # basis used for published-value comparison

@dataclass
class ComputedMetric:
    ticker: str
    fiscal_period: str
    filing_date: str
    source_available_date: str
    metric_name: str
    metric_value: float | None
    source_accession: str
    unit: str

@dataclass
class TextSectionRecord:
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
    filing_date: str
    source_available_date: str
    source_accession: str
    section: str
    feature_type: str
    feature_name: str
    value: float
    prev_filing_date: str | None

@dataclass
class FeatureTargetRecord:
    feature_period: str
    feature_available_date: str
    prediction_date: str
    target_period: str
    target_available_date: str
    target_name: str
    target_value: float | None
    source_accessions: list[str]
    # + dynamic feature columns

@dataclass
class ValuationAssumption:
    assumption_name: str
    value: float | str
    source: str  # "analyst_judgment", "historical_median", "config", etc.
    notes: str

@dataclass
class ExhibitRecord:
    exhibit_id: str
    title: str
    source_caption: str
    data_source: str
    date_range: str
    file_path: str

@dataclass
class DataQualityIssue:
    category: str  # "missing_tag", "fallback_used", "validation_fail", "stale_data", "fiscal_year_mismatch"
    detail: str
    filing_or_source: str
    severity: str  # "info", "warning", "error", "critical"
```

### NEW Data Models: Validation and Eligibility

```python
from enum import Enum

class DataQualityStatus(Enum):
    """Pipeline-wide data validation state."""
    PASS = "pass"                          # All core metrics within tolerance
    PASS_WITH_WARNINGS = "pass_with_warnings"  # Non-critical deviations exist
    DATA_BLOCKED = "data_blocked"          # One or more critical metrics fail

class ComponentStatusEnum(Enum):
    """Per-component quality state."""
    USABLE = "usable"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"

class ReportMode(Enum):
    """Report generation mode."""
    FORMAL_RATING = "formal_rating"
    DIAGNOSTIC_NOT_RATED = "diagnostic_not_rated"
    FAILED = "failed"

@dataclass
class ValidatedMetric:
    """Cross-check of a parsed metric against a published reference value."""
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
    component_name: str             # "data_validation", "ml_signal", "nlp_signal", "peer_multiples", "valuation", "segment_chart"
    status: ComponentStatusEnum     # usable / diagnostic_only / unavailable / blocked
    reason: str                     # human-readable explanation
    details: dict | None = None     # optional structured details

@dataclass
class RecommendationStatus:
    """Output of the recommendation eligibility gate."""
    eligibility_status: ReportMode  # formal_rating / diagnostic_not_rated / failed
    data_quality_status: DataQualityStatus
    component_statuses: list[ComponentStatus]
    blocking_issues: list[str]      # human-readable list of blockers
    timestamp: str

@dataclass
class ChartStatus:
    """Whether a chart should be rendered, suppressed, or flagged."""
    chart_name: str
    renderable: bool
    reason: str                     # "ok", "all_categories_other", "data_blocked", etc.
    fallback_message: str | None    # message to show if chart suppressed

@dataclass
class AuditStatus:
    """Machine-readable audit output."""
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

@dataclass
class Recommendation:
    """Enhanced with eligibility status."""
    rating: str                     # "Buy", "Hold", "Sell", "Not Rated"
    eligibility_status: str         # "formal_rating", "diagnostic_not_rated", "failed"
    not_rated_reason: str | None    # explanation if not rated
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
    scorecard: dict                 # component_name → {status, value, reason}
    target_horizon: str
    sanity_check_notes: list[str]   # bridge explanations for >50% gap, grid warnings, etc.
```

## Components

### `src/config.py`

```python
@dataclass
class EngineConfig:
    ticker: str = "NVDA"
    cik: str = "0001045810"
    company_name: str = "NVIDIA Corporation"
    start_fiscal_year: int = 2016
    end_fiscal_year: int = 2026
    report_date: str = "2026-05-01"
    price_date: str = "2026-05-01"
    output_format: str = "pdf"

    # Validation
    core_validation_metrics: list[str]  # revenue, gross_profit, operating_income, net_income, operating_cash_flow, capex, FCF, cash_and_securities, total_debt, diluted_shares, diluted_EPS, R&D
    validation_tolerance_pct: float = 1.0
    min_nlp_char_mda: int = 500
    min_nlp_char_risk: int = 300
    max_dcf_price_divergence_pct: float = 50.0

    # Peers
    core_semiconductor_peers: list[str]
    infrastructure_peers: list[str]
    ai_capex_context: list[str]
    peer_justifications: dict[str, str]
    peer_staleness_threshold_days: int = 90

    # NLP, ML, Valuation (unchanged from original)
    # ...

    # Split adjustment
    split_history: list[dict]  # [{date, ratio, description}]
```

### `src/xbrl_parser.py` (enhanced)

```python
class XBRLParser:
    CONCEPT_MAP: dict[str, list[str]]

    def parse_companyfacts(self, facts_json: dict) -> pd.DataFrame:
        """Output includes source_available_date, fiscal_period_type, period_start,
        period_end, duration_days, frame, xbrl_concept per fact."""

    def _classify_period_type(self, period_start: str, period_end: str, duration: int) -> str:
        """Returns 'annual' (350-380d), 'quarterly' (<100d), 'ytd' (100-340d), 'instant'."""

    def _select_annual_fact(self, candidates: pd.DataFrame, metric: str, fy: int) -> pd.Series:
        """Fiscal-year selection algorithm:
        1. Filter to annual-duration facts (350-380 days)
        2. Prefer 10-K form_type
        3. Prefer frame-tagged facts
        4. Latest amendment wins
        5. Record selection_rank and selection_reason
        Rejects quarterly (<100d) and YTD (100-340d) facts."""

    def _handle_amendments(self, facts: pd.DataFrame) -> pd.DataFrame:
        """Deterministic amendment handling: latest filing_date for same fy wins."""

    def validate_against_published(self, parsed: pd.DataFrame, known_values: dict) -> list[ValidatedMetric]:
        """Returns list of ValidatedMetric with tolerance check, severity, blocker status."""

    def compute_data_quality_status(self, validations: list[ValidatedMetric]) -> DataQualityStatus:
        """PASS if all pass, PASS_WITH_WARNINGS if non-critical warnings, DATA_BLOCKED if any critical blocker."""

    def generate_data_quality_report(self, parsed, validations, status) -> str:
        """Missing tags, fallbacks, coverage, validation → data_quality_report.md."""
```

### `src/data_validation.py` (NEW)

```python
class DataValidationGate:
    """Hard gate between XBRL parsing and downstream pipeline stages."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.core_metrics = config.core_validation_metrics
        self.tolerance = config.validation_tolerance_pct

    def validate_parsed_data(self, parsed_df: pd.DataFrame, published_values: dict) -> tuple[DataQualityStatus, list[ValidatedMetric]]:
        """Run validation for all core metrics across all fiscal years.
        Returns (status, list of ValidatedMetric)."""

    def check_fiscal_year_selection(self, parsed_df: pd.DataFrame) -> list[DataQualityIssue]:
        """Verify no quarterly/YTD values were selected as annual.
        Flag any fact where duration_days < 340 was used as annual."""

    def generate_validation_summary(self, status: DataQualityStatus, metrics: list[ValidatedMetric]) -> dict:
        """Machine-readable summary for audit_status.json."""

    def should_block_recommendation(self, status: DataQualityStatus) -> bool:
        """Returns True if status is DATA_BLOCKED."""

    def get_diagnostic_label(self, status: DataQualityStatus) -> str:
        """Returns appropriate label for valuation outputs."""
```

### `src/segment_revenue.py` (enhanced)

```python
class SegmentRevenueNormalizer:
    LABEL_MAPPING: dict[str, str]

    def normalize(self, xbrl_segments: pd.DataFrame, filing_texts: dict = None) -> pd.DataFrame:
        """Output: SegmentRevenueRecord schema."""

    def check_segment_quality(self, normalized: pd.DataFrame) -> ChartStatus:
        """Returns ChartStatus. Blocked if all categories are 'Other'/'Unclassified'.
        Warns if segment totals differ from reported revenue by >5%."""

    def generate_mapping_report(self) -> str: ...
```

### `src/filing_text_parser.py` (enhanced)

```python
class FilingTextParser:
    SECTIONS_10K = {"business": "Item 1", "risk_factors": "Item 1A", "mda": "Item 7", "quant": "Item 7A"}
    SECTIONS_10Q = {"risk_factors": "Part II, Item 1A", "mda": "Part I, Item 2"}

    def parse_filing(self, html: str, accession: str, form_type: str) -> list[TextSectionRecord]: ...

    def check_extraction_quality(self, records: list[TextSectionRecord], config: EngineConfig) -> ComponentStatus:
        """Returns ComponentStatus for NLP. diagnostic_only if >50% filings below char threshold."""

    def generate_coverage_report(self, results: list[TextSectionRecord]) -> pd.DataFrame: ...
```

### `src/financial_metrics.py`

```python
class FinancialMetricsCalculator:
    def compute_all_metrics(self, xbrl_data: pd.DataFrame) -> pd.DataFrame:
        """Output includes source_available_date. Computes historical FCF margin."""
    def compute_ttm(self, quarterly: pd.DataFrame, metric: str) -> pd.DataFrame: ...
    def flag_inflection_points(self, metrics: pd.DataFrame) -> pd.DataFrame: ...
```

### `src/nlp_features.py` (enhanced)

```python
class NLPFeatureExtractor:
    TRACKED_THEMES: list[str]
    KEYWORD_DICTIONARIES: dict[str, list[str]]

    def compute_tfidf_similarity(self, texts: list[TextSectionRecord]) -> pd.DataFrame: ...
    def compute_keyword_scores(self, text: str) -> dict[str, float]: ...
    def compute_narrative_drift(self, sections: list[TextSectionRecord]) -> pd.DataFrame: ...
    def extract_shift_snippets(self, pairs: list, n: int = 5) -> list[dict]: ...

    def assess_nlp_quality(self, sections: list[TextSectionRecord], config: EngineConfig) -> ComponentStatus:
        """Returns ComponentStatus. diagnostic_only if extraction quality is poor."""
```

### `src/ml_models.py` (enhanced)

```python
class MLDriverModel:
    def build_feature_target_matrix(self, metrics, nlp) -> pd.DataFrame: ...
    def validate_no_lookahead_matrix(self, matrix) -> bool: ...
    def train_primary_model(self, features, target) -> tuple: ...
    def walk_forward_validate(self, features, target) -> dict: ...
    def compute_baselines(self, target) -> dict: ...
    def evaluate(self, predictions, actuals) -> dict: ...
    def generate_model_audit(self, model, baselines, features) -> str: ...

    def assess_ml_quality(self, model_results: dict, baseline_results: dict) -> ComponentStatus:
        """Returns ComponentStatus.
        diagnostic_only if: all coefficients near-zero, or MAE worse than all baselines.
        Checks: max(|coef|) < 0.001 → degenerate; MAE > all baseline MAEs → underperforms."""

    def get_coefficient_summary(self, model) -> dict:
        """Returns {feature: coef} for audit and quality assessment."""
```

### `src/valuation.py` (enhanced)

```python
class ValuationModule:
    def compute_dcf(self, assumptions, base_revenue, wacc, net_cash, shares) -> dict: ...
    def compute_sensitivity_table(self, ...) -> pd.DataFrame: ...
    def compute_reverse_dcf_grid(self, price, shares, cash, wacc, cagr_range, margin_range) -> pd.DataFrame: ...
    def solve_implied_cagr(self, ...) -> float: ...
    def solve_implied_margin(self, ...) -> float: ...
    def compute_peer_multiples(self, peer_fin, nvda_fin) -> pd.DataFrame: ...
    def build_scenarios(self, base_revenue, wacc, net_cash, shares) -> dict: ...
    def reconcile_historical_fcf_margin(self, metrics) -> pd.DataFrame: ...

    def check_valuation_inputs(self, data_quality_status: DataQualityStatus, validated_metrics: list) -> ComponentStatus:
        """Returns blocked if any valuation-critical input failed validation."""

    def generate_sanity_checks(self, target_price: float, current_price: float,
                                reverse_dcf_grid: pd.DataFrame, bear_value: float,
                                config: EngineConfig) -> list[str]:
        """Returns list of sanity-check notes:
        - Bridge explanation if |target - current| / current > 50%
        - Warning if grid doesn't contain current price ±10%
        - Flag if bear downside > -40%
        - Check sensitivity table spans current price"""

    def filter_peer_multiples(self, peer_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Filter NaN/negative EV rows. Separate semi/infra/context peers.
        Returns (clean_df, exclusion_log)."""

    def generate_recommendation(self, scenarios, current_price, reverse_dcf,
                                 ml_signal, narrative_signal,
                                 recommendation_status: RecommendationStatus) -> Recommendation:
        """Scorecard-based recommendation. If recommendation_status is not formal_rating,
        rating = 'Not Rated' with explanation."""
```

### `src/charts.py` (enhanced)

```python
class ChartGenerator:
    def plot_revenue_segment_mix(self, segment_data, chart_status: ChartStatus) -> Path | None:
        """Suppressed if chart_status.renderable is False. Shows fallback_message instead."""
    def plot_margin_trends(self, metrics) -> Path: ...
    def plot_fcf_trend(self, metrics) -> Path: ...
    def plot_narrative_drift(self, nlp) -> Path: ...
    def plot_dcf_scenarios(self, scenarios, diagnostic_label: str | None = None) -> Path: ...
    def plot_reverse_dcf_grid(self, grid, current_price: float, grid_warning: str | None = None) -> Path: ...
    def plot_recommendation_scorecard(self, scorecard, recommendation_status: RecommendationStatus) -> Path: ...

    def _save_with_attribution(self, fig, filename, source_caption) -> Path: ...
```

### `src/report_utils.py` (enhanced)

```python
class ReportGenerator:
    def build_report_context(self, metrics, segments, nlp, model, valuation, charts, audit,
                              recommendation_status: RecommendationStatus) -> dict:
        """Assembles context. Excludes data with source_available_date > report_date.
        Includes report_mode from recommendation_status."""

    def generate_markdown_report(self, context, report_mode: ReportMode) -> str:
        """Renders report. In diagnostic_not_rated mode:
        - Cover page shows 'Not Rated — Data Validation Required'
        - Valuation sections labeled 'Diagnostic only'
        - Scorecard shows component statuses"""

    def generate_pdf(self, markdown) -> Path: ...
    def generate_html(self, markdown) -> Path: ...
    def generate_executive_summary(self, context, report_mode: ReportMode) -> str: ...
```

### `src/audit_utils.py` (enhanced)

```python
class AuditModule:
    def generate_source_attribution(self, provenance, exhibits) -> str: ...
    def generate_prompt_log(self, prompts) -> str: ...
    def generate_data_dictionary(self, schema) -> str: ...

    def generate_limitations(self, gaps, assumptions, validation_failures: list[ValidatedMetric]) -> str:
        """MUST include validation failures if any exist.
        SHALL NOT output 'No data quality issues recorded' when failures are present."""

    def perform_self_audit(self, report_md, attribution) -> str: ...
    def validate_exhibit_attribution(self, report_md, attribution) -> list[str]: ...
    def validate_report_date_filtering(self, context, report_date) -> list[str]: ...

    def check_audit_consistency(self, limitations_md: str, data_quality_report_md: str,
                                 report_md: str, data_quality_status: DataQualityStatus,
                                 model_audit_md: str, ml_results: dict) -> AuditStatus:
        """Post-report consistency audit (phase 2). Cross-file contradiction detection:
        1. Fail if limitations says 'no issues' but data_quality_report has failures
        2. Fail if report shows Buy/Hold/Sell but DATA_BLOCKED
        3. Fail if model_audit claims outperformance but walk-forward shows otherwise
        4. Fail if ML diagnostic_only is presented as supporting rating direction
        5. Fail if NLP diagnostic_only is presented as supporting rating direction
        6. Fail if valuation uses YTD/quarterly fallback for core annual metrics
        Returns AuditStatus with all checks.
        If any check fails, final package status = 'failed' (not submission-ready)."""

    def run_pre_report_audit(self, data_quality_status: DataQualityStatus,
                              component_statuses: list[ComponentStatus],
                              valuation_input_status: ComponentStatus) -> RecommendationStatus:
        """Pre-report eligibility audit (phase 1). Determines ReportMode before rendering."""

    def generate_audit_status_json(self, audit_status: AuditStatus, package_status: str) -> str:
        """Writes outputs/audit_status.json.
        package_status: 'formal_rating_pass', 'diagnostic_not_rated_pass', or 'failed'."""
```

### `scripts/run_pipeline.py` (enhanced)

```
Usage: python scripts/run_pipeline.py --ticker NVDA --report-date 2026-05-01 --price-date 2026-05-01
Flags: --force-refresh, --skip-tests, --output-format pdf|html|both,
       --step all|ingest|parse|validate|metrics|segments|text|nlp|ml|valuation|report|audit

Pipeline stages with gates:
1. Ingest (SEC + market)
2. Parse (XBRL with fiscal-year selection)
3. VALIDATE (data validation gate → DataQualityStatus)
   - If DATA_BLOCKED: pipeline continues but all downstream outputs are diagnostic-only
4. Metrics, Segments, Text, NLP, ML (each tracks ComponentStatus)
5. Valuation (blocked if valuation inputs failed)
6. ELIGIBILITY (recommendation eligibility gate → RecommendationStatus)
7. Report (mode determined by RecommendationStatus)
8. AUDIT (consistency check → audit_status.json)
```

### Split-Adjustment Policy

NVIDIA executed a 10:1 stock split on June 10, 2024. The engine handles this as follows:
- Raw XBRL facts are NEVER overwritten. Both raw and adjusted values are stored:
  - `raw_value`: the value as reported in the filing
  - `adjusted_value`: the value after split adjustment
  - `adjustment_factor`: the cumulative split ratio applied (e.g., 10.0 for pre-split facts)
  - `adjustment_basis`: "split_adjusted_to_current" or "as_reported"
  - `validation_basis`: which basis the published reference uses for comparison
- Per-share valuation (DCF per-share value) uses current-price-compatible diluted shares
- Historical EPS/share-count validation compares on the same basis as the published reference
- Charts displaying per-share values label the split basis
- If split basis is unresolved (e.g., cannot determine whether published reference is pre- or post-split), formal valuation is blocked (DataQualityStatus = DATA_BLOCKED)
- Tests verify: raw facts preserved, adjusted facts reproducible, current price and diluted shares use compatible basis, 10x share-count mismatch triggers DATA_BLOCKED

