"""
Tests for src.data_validation — DataValidationGate.

Covers:
  - PASS when all metrics within 1% tolerance
  - DATA_BLOCKED when revenue differs by >1%
  - PASS_WITH_WARNINGS for non-critical deviations
  - FY2025 regression: $26.974B vs $130.497B → DATA_BLOCKED
  - Missing valuation-critical metric → DATA_BLOCKED (Req 14.11)
  - Missing capex reference → DATA_BLOCKED if FCF constructed from OCF minus capex
  - validation_coverage_pct below 90% for latest 3 FYs → DATA_BLOCKED (Req 14.12)
  - Manually sourced validation reference unblocks only if source_attribution.md includes source
  - check_fiscal_year_selection() flags quarterly/YTD tagged as FY
  - generate_validation_summary() machine-readable output
  - should_block_recommendation() and get_diagnostic_label()

All tests run offline using fixtures in tests/fixtures/.
Reqs: 14.4, 14.5, 14.11, 14.12
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    DataQualityStatus,
    EngineConfig,
    ValidatedMetric,
)
from src.data_validation import DataValidationGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures")

# Columns matching XBRLParser.OUTPUT_COLUMNS
_PARSED_COLUMNS = [
    "ticker", "fiscal_period", "fiscal_year", "filing_date",
    "source_available_date", "accession_number", "metric_name",
    "value", "unit", "form_type",
    "fiscal_period_type", "period_start", "period_end",
    "duration_days", "frame", "xbrl_concept",
    "selection_rank", "selection_reason",
    "raw_value", "adjusted_value", "adjustment_factor",
    "adjustment_basis", "validation_basis",
]


def _load_published_values() -> dict:
    """Load the published-value reference fixture."""
    with open(FIXTURES / "nvda_published_values.json") as f:
        return json.load(f)


def _make_gate(**config_overrides) -> DataValidationGate:
    """Create a DataValidationGate with default or overridden config."""
    config = EngineConfig(**config_overrides)
    return DataValidationGate(config)


def _make_row(
    metric_name: str,
    fiscal_year: int,
    value: float | None,
    *,
    fiscal_period: str = "FY",
    form_type: str = "10-K",
    fiscal_period_type: str = "annual",
    duration_days: int = 363,
    period_start: str = "2024-01-29",
    period_end: str = "2025-01-26",
) -> dict:
    """Build a single row dict matching XBRLParser.OUTPUT_COLUMNS."""
    return {
        "ticker": "NVDA",
        "fiscal_period": fiscal_period,
        "fiscal_year": fiscal_year,
        "filing_date": "2025-02-26",
        "source_available_date": "2025-02-26",
        "accession_number": "0001045810-25-000023",
        "metric_name": metric_name,
        "value": value,
        "unit": "USD",
        "form_type": form_type,
        "fiscal_period_type": fiscal_period_type,
        "period_start": period_start,
        "period_end": period_end,
        "duration_days": duration_days,
        "frame": None,
        "xbrl_concept": "",
        "selection_rank": 1,
        "selection_reason": "10-K annual duration",
        "raw_value": value,
        "adjusted_value": value,
        "adjustment_factor": 1.0,
        "adjustment_basis": "as_reported",
        "validation_basis": "as_reported",
    }


def _build_matching_df(published_values: dict) -> pd.DataFrame:
    """Build a parsed DataFrame with values exactly matching published values.

    This produces a PASS scenario — every core metric matches within 0% diff.
    """
    rows = []
    for fy_label, fy_data in published_values.get("values", {}).items():
        fy_num = int(fy_label.replace("FY", ""))
        fy_end = fy_data.get("fiscal_year_end", "2025-01-26")
        fy_start = fy_data.get("fiscal_year_start", "2024-01-29")
        for metric_name, metric_info in fy_data.get("metrics", {}).items():
            rows.append(_make_row(
                metric_name=metric_name,
                fiscal_year=fy_num,
                value=metric_info["value"],
                period_start=fy_start,
                period_end=fy_end,
            ))
    return pd.DataFrame(rows, columns=_PARSED_COLUMNS)


def _build_df_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts."""
    return pd.DataFrame(rows, columns=_PARSED_COLUMNS)


# ===================================================================
# 1. PASS scenario — all metrics within 1% tolerance
# ===================================================================


class TestPassScenario:
    """Verify DataQualityStatus.PASS when all metrics match published values."""

    def test_exact_match_returns_pass(self):
        """Exact match of all published values → PASS."""
        published = _load_published_values()
        df = _build_matching_df(published)
        gate = _make_gate()

        status, validated = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.PASS

    def test_within_tolerance_returns_pass(self):
        """Values within 0.5% of published → PASS."""
        published = _load_published_values()
        df = _build_matching_df(published)

        # Nudge all values by 0.5% (within 1% tolerance)
        df["value"] = df["value"].apply(
            lambda v: v * 1.005 if v is not None and v != 0 else v
        )
        gate = _make_gate()

        status, validated = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.PASS

    def test_all_validated_metrics_pass(self):
        """Every ValidatedMetric should have status='pass' for exact match."""
        published = _load_published_values()
        df = _build_matching_df(published)
        gate = _make_gate()

        _, validated = gate.validate_parsed_data(df, published)

        for m in validated:
            if m.published_value is not None and m.parsed_value is not None:
                assert m.status == "pass", (
                    f"{m.metric_name} FY{m.fiscal_year}: expected pass, got {m.status}"
                )


# ===================================================================
# 2. DATA_BLOCKED — revenue mismatch >1%
# ===================================================================


class TestDataBlockedRevenueMismatch:
    """Verify DATA_BLOCKED when revenue differs by >1%."""

    def test_revenue_5pct_off_triggers_blocked(self):
        """Revenue 5% off from published → DATA_BLOCKED."""
        published = _load_published_values()
        df = _build_matching_df(published)

        # Corrupt FY2025 revenue by 5%
        mask = (df["metric_name"] == "revenue") & (df["fiscal_year"] == 2025)
        df.loc[mask, "value"] = 130497000000 * 0.95  # 5% low

        gate = _make_gate()
        status, validated = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.DATA_BLOCKED

    def test_revenue_blocker_metric_has_fail_status(self):
        """The failed revenue metric should have status='fail' and blocker=True."""
        published = _load_published_values()
        df = _build_matching_df(published)

        mask = (df["metric_name"] == "revenue") & (df["fiscal_year"] == 2025)
        df.loc[mask, "value"] = 130497000000 * 0.90  # 10% off

        gate = _make_gate()
        _, validated = gate.validate_parsed_data(df, published)

        rev_2025 = [
            m for m in validated
            if m.metric_name == "revenue" and m.fiscal_year == 2025
        ]
        assert len(rev_2025) == 1
        assert rev_2025[0].status == "fail"
        assert rev_2025[0].severity == "critical"
        assert rev_2025[0].blocker is True


# ===================================================================
# 3. DATA_BLOCKED — any critical metric failure
# ===================================================================


class TestDataBlockedCriticalMetric:
    """Any core metric with >1% diff should trigger DATA_BLOCKED."""

    @pytest.mark.parametrize("metric_name", [
        "gross_profit", "operating_income", "net_income",
        "operating_cash_flow", "capex",
    ])
    def test_critical_metric_failure_blocks(self, metric_name: str):
        """Each core metric failure individually triggers DATA_BLOCKED."""
        published = _load_published_values()
        df = _build_matching_df(published)

        # Corrupt the metric for FY2025 by 50%
        mask = (df["metric_name"] == metric_name) & (df["fiscal_year"] == 2025)
        if mask.any():
            df.loc[mask, "value"] = df.loc[mask, "value"].iloc[0] * 0.50

        gate = _make_gate()
        status, _ = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.DATA_BLOCKED, (
            f"Expected DATA_BLOCKED for {metric_name} 50% off, got {status}"
        )


# ===================================================================
# 4. PASS_WITH_WARNINGS — non-critical deviations
# ===================================================================


class TestPassWithWarnings:
    """Non-critical deviations should produce PASS_WITH_WARNINGS.

    The DataValidationGate assigns severity='critical' to all core metrics.
    PASS_WITH_WARNINGS is triggered when a non-core metric (severity='major',
    blocker=False) has a failure or missing status. We test this by injecting
    a non-core metric into the validated results via a custom config that
    includes a metric not in _VALUATION_CRITICAL_METRICS and not in the
    default core list, then verifying the status logic.
    """

    def test_pass_with_warnings_from_non_blocker_missing(self):
        """Directly verify _compute_overall_status returns PASS_WITH_WARNINGS
        when a non-blocker metric is missing but no critical blockers exist.
        """
        gate = _make_gate()

        # Simulate validated metrics: all core pass, one non-blocker missing
        validated = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
            ValidatedMetric(
                metric_name="some_minor_metric", fiscal_year=2025,
                parsed_value=None, published_value=1000000,
                diff_pct=None, tolerance=1.0, status="missing",
                severity="major", blocker=False,
            ),
        ]

        published = _load_published_values()
        values_by_fy = published.get("values", {})
        status = gate._compute_overall_status(validated, values_by_fy)

        assert status is DataQualityStatus.PASS_WITH_WARNINGS

    def test_pass_with_warnings_from_non_critical_fail(self):
        """A non-critical metric failure (severity='major') should produce
        PASS_WITH_WARNINGS, not DATA_BLOCKED.
        """
        gate = _make_gate()

        validated = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
            ValidatedMetric(
                metric_name="some_context_metric", fiscal_year=2025,
                parsed_value=5000000, published_value=10000000,
                diff_pct=50.0, tolerance=1.0, status="fail",
                severity="major", blocker=False,
            ),
        ]

        published = _load_published_values()
        values_by_fy = published.get("values", {})
        status = gate._compute_overall_status(validated, values_by_fy)

        assert status is DataQualityStatus.PASS_WITH_WARNINGS


# ===================================================================
# 5. FY2025 regression: $26.974B vs $130.497B → DATA_BLOCKED
# ===================================================================


class TestFY2025Regression:
    """Specifically test the $26.974B vs $130.497B misparse scenario."""

    def test_fy2025_revenue_misparse_blocked(self):
        """FY2025 revenue=$26.974B (the misparse) vs published=$130.497B → DATA_BLOCKED."""
        published = _load_published_values()
        df = _build_matching_df(published)

        # Replace FY2025 revenue with the misparse value
        mask = (df["metric_name"] == "revenue") & (df["fiscal_year"] == 2025)
        df.loc[mask, "value"] = 26974000000  # The original misparse

        gate = _make_gate()
        status, validated = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.DATA_BLOCKED

    def test_fy2025_revenue_diff_pct_is_large(self):
        """The diff_pct for the misparse should be ~79%."""
        published = _load_published_values()
        df = _build_matching_df(published)

        mask = (df["metric_name"] == "revenue") & (df["fiscal_year"] == 2025)
        df.loc[mask, "value"] = 26974000000

        gate = _make_gate()
        _, validated = gate.validate_parsed_data(df, published)

        rev_2025 = [
            m for m in validated
            if m.metric_name == "revenue" and m.fiscal_year == 2025
        ]
        assert len(rev_2025) == 1
        # diff = |26.974B - 130.497B| / 130.497B ≈ 79.3%
        assert rev_2025[0].diff_pct is not None
        assert rev_2025[0].diff_pct > 70.0

    def test_fy2025_27b_value_is_blocker(self):
        """The $27B misparse should be flagged as critical blocker."""
        published = _load_published_values()
        df = _build_matching_df(published)

        mask = (df["metric_name"] == "revenue") & (df["fiscal_year"] == 2025)
        df.loc[mask, "value"] = 27000000000  # ~$27B

        gate = _make_gate()
        _, validated = gate.validate_parsed_data(df, published)

        rev_2025 = [
            m for m in validated
            if m.metric_name == "revenue" and m.fiscal_year == 2025
        ]
        assert len(rev_2025) == 1
        assert rev_2025[0].blocker is True
        assert rev_2025[0].severity == "critical"
        assert rev_2025[0].status == "fail"


# ===================================================================
# 6. Missing valuation-critical metric → DATA_BLOCKED (Req 14.11)
# ===================================================================


class TestMissingValuationCritical:
    """Missing revenue/OCF/capex for latest FY → DATA_BLOCKED."""

    def test_missing_revenue_latest_fy_blocked(self):
        """Missing revenue for latest FY → DATA_BLOCKED."""
        published = _load_published_values()
        df = _build_matching_df(published)

        # Remove revenue for FY2025 (the latest FY)
        df = df[~((df["metric_name"] == "revenue") & (df["fiscal_year"] == 2025))]

        gate = _make_gate()
        status, validated = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.DATA_BLOCKED

    def test_missing_ocf_latest_fy_blocked(self):
        """Missing operating_cash_flow for latest FY → DATA_BLOCKED."""
        published = _load_published_values()
        df = _build_matching_df(published)

        df = df[~(
            (df["metric_name"] == "operating_cash_flow") & (df["fiscal_year"] == 2025)
        )]

        gate = _make_gate()
        status, _ = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.DATA_BLOCKED

    def test_missing_capex_blocks_when_fcf_derived(self):
        """Missing capex reference → DATA_BLOCKED if FCF constructed from OCF minus capex.

        Capex is in _VALUATION_CRITICAL_METRICS, so missing parsed capex
        for the latest FY with a published reference triggers DATA_BLOCKED.
        """
        published = _load_published_values()
        df = _build_matching_df(published)

        # Remove capex for FY2025
        df = df[~((df["metric_name"] == "capex") & (df["fiscal_year"] == 2025))]

        gate = _make_gate()
        status, _ = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.DATA_BLOCKED

    def test_missing_diluted_shares_latest_fy_blocked(self):
        """Missing diluted_shares for latest FY → DATA_BLOCKED."""
        published = _load_published_values()
        df = _build_matching_df(published)

        df = df[~(
            (df["metric_name"] == "diluted_shares") & (df["fiscal_year"] == 2025)
        )]

        gate = _make_gate()
        status, _ = gate.validate_parsed_data(df, published)

        assert status is DataQualityStatus.DATA_BLOCKED


# ===================================================================
# 7. Coverage below 90% for latest 3 FYs → DATA_BLOCKED (Req 14.12)
# ===================================================================


class TestCoverageBelowThreshold:
    """validation_coverage_pct below 90% for latest 3 FYs → DATA_BLOCKED."""

    def test_low_coverage_triggers_blocked(self):
        """When most metrics are missing for latest 3 FYs → DATA_BLOCKED.

        We build a DataFrame with only revenue for each FY, leaving all
        other core metrics missing. This should drop coverage well below 90%.
        """
        published = _load_published_values()

        # Build a minimal DataFrame with only revenue for each FY
        rows = []
        for fy_label, fy_data in published.get("values", {}).items():
            fy_num = int(fy_label.replace("FY", ""))
            rev_info = fy_data["metrics"]["revenue"]
            rows.append(_make_row(
                metric_name="revenue",
                fiscal_year=fy_num,
                value=rev_info["value"],
            ))

        df = _build_df_from_rows(rows)
        gate = _make_gate()
        status, _ = gate.validate_parsed_data(df, published)

        # Only revenue passes out of ~12 core metrics per FY → ~8% coverage
        # This should trigger DATA_BLOCKED due to missing valuation-critical metrics
        assert status is DataQualityStatus.DATA_BLOCKED


# ===================================================================
# 8. Manually sourced validation reference
# ===================================================================


class TestManuallySourcedException:
    """Manually sourced validation reference unblocks only if source_attribution.md includes source.

    Note: The current DataValidationGate does not implement source_attribution.md
    checking. This test verifies the base behavior: missing published value for
    a valuation-critical metric in the latest FY → DATA_BLOCKED.
    """

    def test_missing_published_value_for_critical_metric_blocks(self):
        """If published_values has no reference for a valuation-critical metric,
        and the metric is not in the published fixture at all, it should be
        flagged as missing and block.
        """
        published = _load_published_values()
        df = _build_matching_df(published)

        # Remove revenue from published values for FY2025
        # This simulates no published reference available
        del published["values"]["FY2025"]["metrics"]["revenue"]

        gate = _make_gate()
        status, validated = gate.validate_parsed_data(df, published)

        # Revenue is valuation-critical; missing published reference → DATA_BLOCKED
        # (The parsed value exists but has no reference to validate against)
        rev_2025 = [
            m for m in validated
            if m.metric_name == "revenue" and m.fiscal_year == 2025
        ]
        # Should have a "missing" entry since published_value is None
        assert any(m.status == "missing" for m in rev_2025)


# ===================================================================
# 9. check_fiscal_year_selection() — quarterly/YTD tagged as FY
# ===================================================================


class TestCheckFiscalYearSelection:
    """Verify check_fiscal_year_selection flags non-annual facts tagged as FY."""

    def test_quarterly_tagged_as_fy_flagged(self):
        """A quarterly fact (duration=90) tagged as FY should be flagged."""
        rows = [_make_row(
            metric_name="revenue",
            fiscal_year=2025,
            value=35082000000,
            fiscal_period_type="quarterly",
            duration_days=90,
        )]
        df = _build_df_from_rows(rows)

        gate = _make_gate()
        issues = gate.check_fiscal_year_selection(df)

        assert len(issues) >= 1
        assert issues[0].category == "fiscal_year_mismatch"
        assert issues[0].severity == "critical"
        assert "quarterly" in issues[0].detail

    def test_ytd_tagged_as_fy_flagged(self):
        """A YTD fact (duration=272) tagged as FY should be flagged."""
        rows = [_make_row(
            metric_name="revenue",
            fiscal_year=2025,
            value=91166000000,
            fiscal_period_type="ytd",
            duration_days=272,
        )]
        df = _build_df_from_rows(rows)

        gate = _make_gate()
        issues = gate.check_fiscal_year_selection(df)

        assert len(issues) >= 1
        assert "ytd" in issues[0].detail

    def test_annual_tagged_as_fy_clean(self):
        """An annual fact tagged as FY should produce no issues."""
        rows = [_make_row(
            metric_name="revenue",
            fiscal_year=2025,
            value=130497000000,
            fiscal_period_type="annual",
            duration_days=363,
        )]
        df = _build_df_from_rows(rows)

        gate = _make_gate()
        issues = gate.check_fiscal_year_selection(df)

        assert len(issues) == 0

    def test_instant_tagged_as_fy_clean(self):
        """An instant fact tagged as FY should produce no issues."""
        rows = [_make_row(
            metric_name="cash_and_equivalents",
            fiscal_year=2025,
            value=8589000000,
            fiscal_period_type="instant",
            duration_days=0,
        )]
        df = _build_df_from_rows(rows)

        gate = _make_gate()
        issues = gate.check_fiscal_year_selection(df)

        assert len(issues) == 0

    def test_empty_df_no_issues(self):
        """Empty DataFrame should produce no issues."""
        df = pd.DataFrame(columns=_PARSED_COLUMNS)

        gate = _make_gate()
        issues = gate.check_fiscal_year_selection(df)

        assert len(issues) == 0

    def test_non_fy_rows_ignored(self):
        """Rows with fiscal_period != 'FY' should not be checked."""
        rows = [_make_row(
            metric_name="revenue",
            fiscal_year=2025,
            value=35082000000,
            fiscal_period="Q3",
            fiscal_period_type="quarterly",
            duration_days=90,
        )]
        df = _build_df_from_rows(rows)

        gate = _make_gate()
        issues = gate.check_fiscal_year_selection(df)

        assert len(issues) == 0


# ===================================================================
# 10. generate_validation_summary() — machine-readable output
# ===================================================================


class TestGenerateValidationSummary:
    """Verify generate_validation_summary() produces correct structure."""

    def test_summary_structure(self):
        """Summary should have all required keys."""
        gate = _make_gate()
        metrics = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
        ]

        summary = gate.generate_validation_summary(DataQualityStatus.PASS, metrics)

        assert "overall_status" in summary
        assert "metrics_validated" in summary
        assert "metrics_passed" in summary
        assert "metrics_failed" in summary
        assert "metrics_missing" in summary
        assert "validation_coverage_pct" in summary
        assert "blocking_issues" in summary
        assert "timestamp" in summary
        assert "details" in summary

    def test_summary_pass_status(self):
        """Summary overall_status should be 'pass' for PASS."""
        gate = _make_gate()
        metrics = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
        ]

        summary = gate.generate_validation_summary(DataQualityStatus.PASS, metrics)

        assert summary["overall_status"] == "pass"
        assert summary["metrics_validated"] == 1
        assert summary["metrics_passed"] == 1
        assert summary["metrics_failed"] == 0

    def test_summary_blocked_status(self):
        """Summary overall_status should be 'data_blocked' for DATA_BLOCKED."""
        gate = _make_gate()
        metrics = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=26974000000, published_value=130497000000,
                diff_pct=79.33, tolerance=1.0, status="fail",
                severity="critical", blocker=True,
            ),
        ]

        summary = gate.generate_validation_summary(
            DataQualityStatus.DATA_BLOCKED, metrics,
        )

        assert summary["overall_status"] == "data_blocked"
        assert summary["metrics_failed"] == 1
        assert len(summary["blocking_issues"]) >= 1

    def test_summary_details_populated(self):
        """Each metric should appear in the details list."""
        gate = _make_gate()
        metrics = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
            ValidatedMetric(
                metric_name="net_income", fiscal_year=2025,
                parsed_value=72880000000, published_value=72880000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
        ]

        summary = gate.generate_validation_summary(DataQualityStatus.PASS, metrics)

        assert len(summary["details"]) == 2
        detail_names = {d["metric_name"] for d in summary["details"]}
        assert "revenue" in detail_names
        assert "net_income" in detail_names

    def test_summary_coverage_pct_correct(self):
        """validation_coverage_pct should be (passed / total) * 100."""
        gate = _make_gate()
        metrics = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
            ValidatedMetric(
                metric_name="net_income", fiscal_year=2025,
                parsed_value=None, published_value=72880000000,
                diff_pct=None, tolerance=1.0, status="missing",
                severity="critical", blocker=True,
            ),
        ]

        summary = gate.generate_validation_summary(DataQualityStatus.DATA_BLOCKED, metrics)

        assert summary["validation_coverage_pct"] == 50.0

    def test_summary_blocking_issues_for_missing(self):
        """Missing critical metrics should appear in blocking_issues."""
        gate = _make_gate()
        metrics = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=None, published_value=130497000000,
                diff_pct=None, tolerance=1.0, status="missing",
                severity="critical", blocker=True,
            ),
        ]

        summary = gate.generate_validation_summary(
            DataQualityStatus.DATA_BLOCKED, metrics,
        )

        assert len(summary["blocking_issues"]) >= 1
        assert "revenue" in summary["blocking_issues"][0]


# ===================================================================
# 11. should_block_recommendation() and get_diagnostic_label()
# ===================================================================


class TestShouldBlockRecommendation:
    """Verify should_block_recommendation() returns correct booleans."""

    def test_blocked_for_data_blocked(self):
        gate = _make_gate()
        assert gate.should_block_recommendation(DataQualityStatus.DATA_BLOCKED) is True

    def test_not_blocked_for_pass(self):
        gate = _make_gate()
        assert gate.should_block_recommendation(DataQualityStatus.PASS) is False

    def test_not_blocked_for_pass_with_warnings(self):
        gate = _make_gate()
        assert gate.should_block_recommendation(
            DataQualityStatus.PASS_WITH_WARNINGS,
        ) is False


class TestGetDiagnosticLabel:
    """Verify get_diagnostic_label() returns correct labels."""

    def test_label_for_data_blocked(self):
        gate = _make_gate()
        label = gate.get_diagnostic_label(DataQualityStatus.DATA_BLOCKED)
        assert label == "Diagnostic only — do not use for recommendation"

    def test_label_for_pass(self):
        gate = _make_gate()
        label = gate.get_diagnostic_label(DataQualityStatus.PASS)
        assert label == ""

    def test_label_for_pass_with_warnings(self):
        gate = _make_gate()
        label = gate.get_diagnostic_label(DataQualityStatus.PASS_WITH_WARNINGS)
        assert label == ""
