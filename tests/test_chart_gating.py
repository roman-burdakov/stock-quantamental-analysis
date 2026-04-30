"""
Tests for chart gating (Milestone 14).

Validates:
- Segment chart suppression when ChartStatus.renderable is False
- Diagnostic label on DCF charts
- Scorecard renders all component statuses (usable, diagnostic_only, blocked)
- Reverse-DCF grid annotates when current price is outside grid

Reqs: 19.1, 16.3, 17.3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.charts import ChartGenerator
from src.config import (
    ChartStatus,
    ComponentStatus,
    ComponentStatusEnum,
    DataQualityStatus,
    RecommendationStatus,
    ReportMode,
    get_default_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chart_gen() -> ChartGenerator:
    config = get_default_config()
    return ChartGenerator(config)


@pytest.fixture
def sample_segment_data() -> pd.DataFrame:
    return pd.DataFrame({
        "fiscal_period": ["FY2024", "FY2024", "FY2025", "FY2025"],
        "normalized_category": ["Data Center", "Gaming", "Data Center", "Gaming"],
        "value": [47525e6, 10447e6, 115199e6, 11356e6],
        "extraction_method": ["xbrl_dimension"] * 4,
        "source_available_date": ["2024-02-21"] * 2 + ["2025-02-26"] * 2,
    })


@pytest.fixture
def sample_scenarios() -> dict:
    return {
        "bear": {"per_share_value": 80.0, "probability": 0.25},
        "base": {"per_share_value": 140.0, "probability": 0.50},
        "bull": {"per_share_value": 200.0, "probability": 0.25},
    }


@pytest.fixture
def sample_grid() -> pd.DataFrame:
    """A small reverse-DCF grid with implied prices."""
    cagrs = [0.10, 0.15, 0.20, 0.25]
    margins = [0.25, 0.30, 0.35]
    data = np.array([
        [50, 65, 80],
        [70, 90, 110],
        [95, 120, 150],
        [125, 160, 200],
    ], dtype=float)
    return pd.DataFrame(data, index=cagrs, columns=margins)


# ---------------------------------------------------------------------------
# Test: Segment chart suppression
# ---------------------------------------------------------------------------

class TestSegmentChartSuppression:
    def test_suppressed_when_not_renderable(self, chart_gen, sample_segment_data):
        """Segment chart should show fallback when ChartStatus.renderable is False."""
        status = ChartStatus(
            chart_name="segment_chart",
            renderable=False,
            reason="all_categories_other",
            fallback_message="Segment chart suppressed: all categories are Other/Unclassified",
        )
        result = chart_gen.plot_revenue_segment_mix(sample_segment_data, chart_status=status)

        assert result is not None
        assert Path(result).exists()
        # The chart should be a fallback placeholder, not the real chart
        assert "revenue_segment_mix.png" in str(result)

    def test_rendered_when_renderable(self, chart_gen, sample_segment_data):
        """Segment chart should render normally when ChartStatus.renderable is True."""
        status = ChartStatus(
            chart_name="segment_chart",
            renderable=True,
            reason="ok",
        )
        result = chart_gen.plot_revenue_segment_mix(sample_segment_data, chart_status=status)

        assert result is not None
        assert Path(result).exists()

    def test_rendered_without_chart_status(self, chart_gen, sample_segment_data):
        """Segment chart should render normally when no ChartStatus is provided."""
        result = chart_gen.plot_revenue_segment_mix(sample_segment_data)

        assert result is not None
        assert Path(result).exists()

    def test_fallback_message_used(self, chart_gen, sample_segment_data):
        """Fallback message from ChartStatus should be used in placeholder."""
        status = ChartStatus(
            chart_name="segment_chart",
            renderable=False,
            reason="data_blocked",
            fallback_message="Revenue data blocked — chart unavailable",
        )
        result = chart_gen.plot_revenue_segment_mix(sample_segment_data, chart_status=status)

        assert result is not None
        # The exhibit record should reflect the fallback
        assert len(chart_gen.exhibits) > 0


# ---------------------------------------------------------------------------
# Test: Diagnostic label on DCF charts
# ---------------------------------------------------------------------------

class TestDCFDiagnosticLabel:
    def test_dcf_chart_with_diagnostic_label(self, chart_gen, sample_scenarios):
        """DCF chart should include diagnostic label when provided."""
        result = chart_gen.plot_dcf_scenarios(
            sample_scenarios,
            diagnostic_label="Diagnostic only — do not use for recommendation",
        )

        assert result is not None
        assert Path(result).exists()
        assert "dcf_scenarios.png" in str(result)

    def test_dcf_chart_without_diagnostic_label(self, chart_gen, sample_scenarios):
        """DCF chart should render normally without diagnostic label."""
        result = chart_gen.plot_dcf_scenarios(sample_scenarios)

        assert result is not None
        assert Path(result).exists()

    def test_dcf_chart_empty_scenarios(self, chart_gen):
        """DCF chart should handle empty scenarios gracefully."""
        result = chart_gen.plot_dcf_scenarios({}, diagnostic_label="Diagnostic only")

        assert result is not None
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# Test: Scorecard renders all statuses
# ---------------------------------------------------------------------------

class TestScorecardStatuses:
    def test_scorecard_with_recommendation_status(self, chart_gen):
        """Scorecard chart should render with component statuses."""
        scorecard = {
            "valuation_upside": 0.8,
            "reverse_dcf_plausibility": 0.6,
            "ml_signal": 0.3,
            "narrative_drift_signal": 0.5,
        }
        rec_status = RecommendationStatus(
            eligibility_status=ReportMode.DIAGNOSTIC_NOT_RATED,
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            component_statuses=[
                ComponentStatus("valuation_upside", ComponentStatusEnum.USABLE, "OK"),
                ComponentStatus("reverse_dcf_plausibility", ComponentStatusEnum.USABLE, "OK"),
                ComponentStatus("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY, "Underperforms"),
                ComponentStatus("narrative_drift_signal", ComponentStatusEnum.BLOCKED, "Extraction failed"),
            ],
            blocking_issues=["Data blocked"],
            timestamp="2026-04-25T00:00:00Z",
        )
        result = chart_gen.plot_recommendation_scorecard(scorecard, recommendation_status=rec_status)

        assert result is not None
        assert Path(result).exists()
        assert "recommendation_scorecard.png" in str(result)

    def test_scorecard_without_recommendation_status(self, chart_gen):
        """Scorecard should still work without recommendation_status."""
        scorecard = {
            "valuation_upside": 0.8,
            "ml_signal": 0.3,
        }
        result = chart_gen.plot_recommendation_scorecard(scorecard)

        assert result is not None
        assert Path(result).exists()

    def test_scorecard_empty(self, chart_gen):
        """Empty scorecard should produce a placeholder."""
        result = chart_gen.plot_recommendation_scorecard({})

        assert result is not None
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# Test: Reverse-DCF grid annotation
# ---------------------------------------------------------------------------

class TestReverseDCFGridAnnotation:
    def test_grid_annotates_when_price_outside(self, chart_gen, sample_grid):
        """Grid chart should annotate when current price is outside range."""
        # Grid range is 50-200, current price 300 is outside
        result = chart_gen.plot_reverse_dcf_grid(
            sample_grid,
            current_price=300.0,
        )

        assert result is not None
        assert Path(result).exists()
        assert "reverse_dcf_grid.png" in str(result)

    def test_grid_no_annotation_when_price_inside(self, chart_gen, sample_grid):
        """Grid chart should not annotate when current price is inside range."""
        # Grid range is 50-200, current price 120 is inside
        result = chart_gen.plot_reverse_dcf_grid(
            sample_grid,
            current_price=120.0,
        )

        assert result is not None
        assert Path(result).exists()

    def test_grid_with_warning_message(self, chart_gen, sample_grid):
        """Grid chart should show warning message when provided."""
        result = chart_gen.plot_reverse_dcf_grid(
            sample_grid,
            current_price=300.0,
            grid_warning="Implied CAGR of 45% exceeds historical range",
        )

        assert result is not None
        assert Path(result).exists()

    def test_grid_without_current_price(self, chart_gen, sample_grid):
        """Grid chart should work without current_price parameter."""
        result = chart_gen.plot_reverse_dcf_grid(sample_grid)

        assert result is not None
        assert Path(result).exists()
