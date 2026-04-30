"""
Tests for report mode support (Milestone 13).

Validates:
- diagnostic_not_rated mode renders correct cover page
- Valuation diagnostic label appears
- Scorecard shows all component statuses
- Audit warnings section appears when audit fails
- Failed mode renders failure cover page
- Formal rating mode renders normal cover page

Reqs: 14.6, 14.7, 17.2, 22.5
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    AuditStatus,
    ChartStatus,
    ComponentStatus,
    ComponentStatusEnum,
    DataQualityStatus,
    EngineConfig,
    Recommendation,
    RecommendationStatus,
    ReportMode,
    get_default_config,
)
from src.report_utils import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> EngineConfig:
    return get_default_config()


@pytest.fixture
def generator(config: EngineConfig) -> ReportGenerator:
    return ReportGenerator(config)


@pytest.fixture
def empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "fiscal_period", "metric_name", "metric_value",
        "source_available_date", "filing_date",
    ])


@pytest.fixture
def empty_segments() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "fiscal_period", "normalized_category", "value",
        "source_available_date",
    ])


@pytest.fixture
def empty_nlp() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "filing_date", "section", "feature_type", "feature_name",
        "value", "source_available_date",
    ])


@pytest.fixture
def sample_recommendation() -> Recommendation:
    return Recommendation(
        rating="Buy",
        current_price=120.0,
        target_price=150.0,
        upside_pct=0.25,
        bear_value=80.0,
        bear_probability=0.25,
        base_value=140.0,
        base_probability=0.50,
        bull_value=200.0,
        bull_probability=0.25,
        expected_value=140.0,
        what_must_be_true_buy="AI capex sustains",
        what_must_be_true_hold="Growth normalizes",
        what_must_be_true_sell="Competition erodes moat",
        scorecard={"valuation_upside": 0.8, "ml_signal": 0.3},
        eligibility_status="formal_rating",
    )


def _make_recommendation_status(
    mode: ReportMode,
    blocking_issues: list[str] | None = None,
    component_statuses: list[ComponentStatus] | None = None,
) -> RecommendationStatus:
    """Helper to create a RecommendationStatus."""
    if component_statuses is None:
        component_statuses = [
            ComponentStatus(
                component_name="data_validation",
                status=ComponentStatusEnum.USABLE,
                reason="All metrics pass",
            ),
            ComponentStatus(
                component_name="ml_signal",
                status=ComponentStatusEnum.DIAGNOSTIC_ONLY,
                reason="Underperforms baselines",
            ),
            ComponentStatus(
                component_name="valuation",
                status=ComponentStatusEnum.USABLE,
                reason="Inputs validated",
            ),
        ]
    return RecommendationStatus(
        eligibility_status=mode,
        data_quality_status=DataQualityStatus.PASS if mode == ReportMode.FORMAL_RATING else DataQualityStatus.DATA_BLOCKED,
        component_statuses=component_statuses,
        blocking_issues=blocking_issues or [],
        timestamp="2026-04-25T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Test: diagnostic_not_rated mode renders correct cover page
# ---------------------------------------------------------------------------

class TestDiagnosticNotRatedMode:
    def test_cover_page_shows_not_rated(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Cover page should show 'Not Rated — Data Validation Required'."""
        rec_status = _make_recommendation_status(
            ReportMode.DIAGNOSTIC_NOT_RATED,
            blocking_issues=["Data validation gate: DataQualityStatus is DATA_BLOCKED"],
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Not Rated" in content
        assert "Data Validation Required" in content

    def test_cover_page_shows_diagnostic_target(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Cover page should show 'N/A — Diagnostic Only' for target price."""
        rec_status = _make_recommendation_status(
            ReportMode.DIAGNOSTIC_NOT_RATED,
            blocking_issues=["Data validation blocked"],
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Diagnostic Only" in content

    def test_valuation_diagnostic_label_appears(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Valuation section should show diagnostic-only label."""
        rec_status = _make_recommendation_status(
            ReportMode.DIAGNOSTIC_NOT_RATED,
            blocking_issues=["Revenue validation failed"],
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Diagnostic only" in content
        assert "do not use for recommendation" in content

    def test_blocking_issues_listed(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Blocking issues should appear in the report."""
        blocking = [
            "Data validation gate: DataQualityStatus is DATA_BLOCKED",
            "Revenue FY2025 differs by 383%",
        ]
        rec_status = _make_recommendation_status(
            ReportMode.DIAGNOSTIC_NOT_RATED,
            blocking_issues=blocking,
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        for issue in blocking:
            assert issue in content


# ---------------------------------------------------------------------------
# Test: scorecard shows all component statuses
# ---------------------------------------------------------------------------

class TestScorecardComponentStatuses:
    def test_component_statuses_in_report(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Report should show component statuses table."""
        components = [
            ComponentStatus("data_validation", ComponentStatusEnum.USABLE, "All pass"),
            ComponentStatus("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY, "Underperforms baselines"),
            ComponentStatus("nlp_signal", ComponentStatusEnum.UNAVAILABLE, "Extraction failed"),
            ComponentStatus("valuation", ComponentStatusEnum.BLOCKED, "Revenue blocked"),
        ]
        rec_status = _make_recommendation_status(
            ReportMode.DIAGNOSTIC_NOT_RATED,
            blocking_issues=["Valuation blocked"],
            component_statuses=components,
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        # All component statuses should appear
        assert "usable" in content
        assert "diagnostic_only" in content
        assert "unavailable" in content
        assert "blocked" in content

    def test_formal_rating_shows_component_statuses(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Even formal_rating mode should show component statuses."""
        components = [
            ComponentStatus("data_validation", ComponentStatusEnum.USABLE, "All pass"),
            ComponentStatus("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY, "Limited data"),
            ComponentStatus("valuation", ComponentStatusEnum.USABLE, "Inputs validated"),
        ]
        rec_status = _make_recommendation_status(
            ReportMode.FORMAL_RATING,
            component_statuses=components,
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Component Status Overview" in content


# ---------------------------------------------------------------------------
# Test: audit warnings section appears when audit fails
# ---------------------------------------------------------------------------

class TestAuditWarnings:
    def test_audit_warnings_appear_in_report(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Audit warnings should appear when audit_status has failures."""
        audit_status = AuditStatus(
            overall_status="fail",
            data_quality_status="pass",
            recommendation_eligibility="formal_rating",
            component_statuses={"data_validation": "usable"},
            checks_run=3,
            checks_passed=2,
            checks_failed=1,
            blocking_issues=[],
            failure_details=[
                "Limitations says 'no issues' but data_quality_report has failures",
            ],
            timestamp="2026-04-25T00:00:00Z",
        )
        rec_status = _make_recommendation_status(ReportMode.FORMAL_RATING)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            audit={"audit_status": audit_status},
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Audit Warnings" in content
        assert "Limitations says" in content

    def test_no_audit_warnings_when_audit_passes(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """No audit warnings section when audit passes."""
        audit_status = AuditStatus(
            overall_status="pass",
            data_quality_status="pass",
            recommendation_eligibility="formal_rating",
            component_statuses={},
            checks_run=3,
            checks_passed=3,
            checks_failed=0,
            blocking_issues=[],
            failure_details=[],
            timestamp="2026-04-25T00:00:00Z",
        )
        rec_status = _make_recommendation_status(ReportMode.FORMAL_RATING)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            audit={"audit_status": audit_status},
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Audit Warnings" not in content

    def test_audit_warnings_from_dict(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Audit warnings from dict-based audit_status should also work."""
        rec_status = _make_recommendation_status(ReportMode.FORMAL_RATING)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            audit={
                "audit_status": {
                    "overall_status": "fail",
                    "failure_details": ["Model audit claims outperformance but results disagree"],
                },
            },
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Audit Warnings" in content
        assert "Model audit claims" in content


# ---------------------------------------------------------------------------
# Test: failed mode
# ---------------------------------------------------------------------------

class TestFailedMode:
    def test_failed_mode_cover_page(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Failed mode should show 'Report Generation Failed'."""
        rec_status = _make_recommendation_status(ReportMode.FAILED)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Report Generation Failed" in content

    def test_failed_mode_no_valuation_section(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Failed mode should not render valuation or other body sections."""
        rec_status = _make_recommendation_status(ReportMode.FAILED)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        # Body sections should not appear in failed mode
        assert "## Valuation" not in content
        assert "## Company Overview" not in content


# ---------------------------------------------------------------------------
# Test: formal_rating mode
# ---------------------------------------------------------------------------

class TestFormalRatingMode:
    def test_formal_rating_normal_cover(
        self, generator, empty_metrics, empty_segments, empty_nlp, sample_recommendation
    ):
        """Formal rating mode should show normal Buy/Hold/Sell rating."""
        rec_status = _make_recommendation_status(ReportMode.FORMAL_RATING)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            valuation={"recommendation": sample_recommendation},
            recommendation_status=rec_status,
        )
        report_path = generator.generate_markdown_report(context)
        content = report_path.read_text(encoding="utf-8")

        assert "Buy" in content
        assert "Not Rated" not in content
        assert "Report Generation Failed" not in content


# ---------------------------------------------------------------------------
# Test: executive summary reflects report mode
# ---------------------------------------------------------------------------

class TestExecutiveSummaryModes:
    def test_diagnostic_executive_summary(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Executive summary should reflect diagnostic mode."""
        rec_status = _make_recommendation_status(
            ReportMode.DIAGNOSTIC_NOT_RATED,
            blocking_issues=["Revenue validation failed"],
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        summary_path = generator.generate_executive_summary(context)
        content = summary_path.read_text(encoding="utf-8")

        assert "Not Rated" in content
        assert "Data Validation Required" in content

    def test_failed_executive_summary(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Executive summary should reflect failed mode."""
        rec_status = _make_recommendation_status(ReportMode.FAILED)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        summary_path = generator.generate_executive_summary(context)
        content = summary_path.read_text(encoding="utf-8")

        assert "Report Generation Failed" in content

    def test_report_mode_override(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Explicit report_mode parameter should override context."""
        rec_status = _make_recommendation_status(ReportMode.FORMAL_RATING)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        # Override to diagnostic
        report_path = generator.generate_markdown_report(
            context, report_mode=ReportMode.DIAGNOSTIC_NOT_RATED
        )
        content = report_path.read_text(encoding="utf-8")

        assert "Not Rated" in content
        assert "Diagnostic only" in content


# ---------------------------------------------------------------------------
# Test: build_report_context includes report_mode
# ---------------------------------------------------------------------------

class TestBuildReportContext:
    def test_context_includes_report_mode(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Context dict should include report_mode from RecommendationStatus."""
        rec_status = _make_recommendation_status(ReportMode.DIAGNOSTIC_NOT_RATED)
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        assert context["report_mode"] == "diagnostic_not_rated"

    def test_context_defaults_to_formal_rating(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Without RecommendationStatus, context defaults to formal_rating."""
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
        )
        assert context["report_mode"] == "formal_rating"

    def test_context_includes_blocking_issues(
        self, generator, empty_metrics, empty_segments, empty_nlp
    ):
        """Context should include blocking_issues from RecommendationStatus."""
        rec_status = _make_recommendation_status(
            ReportMode.DIAGNOSTIC_NOT_RATED,
            blocking_issues=["Revenue blocked", "Valuation blocked"],
        )
        context = generator.build_report_context(
            metrics=empty_metrics,
            segments=empty_segments,
            nlp_features=empty_nlp,
            recommendation_status=rec_status,
        )
        assert len(context["blocking_issues"]) == 2
        assert "Revenue blocked" in context["blocking_issues"]
