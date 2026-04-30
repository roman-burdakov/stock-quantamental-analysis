"""
Tests for additional charts beyond the required 6-8 (Task 17.6).

Validates:
- plot_revenue_growth_trend with synthetic metrics data
- plot_peer_multiples_comparison with synthetic peer data
- plot_keyword_theme_heatmap with synthetic NLP data
- Each chart saves to the correct output path and returns a valid Path
- Each chart produces valid ExhibitRecord metadata
- Edge cases: empty data, missing columns, single data point
- Charts handle NaN values gracefully
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.charts import ChartGenerator
from src.config import ExhibitRecord, get_default_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chart_gen() -> ChartGenerator:
    config = get_default_config()
    return ChartGenerator(config)


@pytest.fixture
def sample_metrics_with_growth() -> pd.DataFrame:
    """Synthetic metrics DataFrame containing revenue_growth_YoY rows."""
    periods = ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024", "FY2025"]
    growth_values = [0.41, 0.53, -0.17, 1.22, 1.14, 0.94]
    return pd.DataFrame({
        "fiscal_period": periods,
        "metric_name": ["revenue_growth_YoY"] * len(periods),
        "metric_value": growth_values,
    })


@pytest.fixture
def sample_peer_multiples() -> pd.DataFrame:
    """Synthetic peer multiples DataFrame with EV/Revenue and P/E."""
    return pd.DataFrame({
        "ticker": ["NVDA", "AMD", "AVGO", "INTC", "QCOM"],
        "EV/Revenue": [25.0, 10.5, 12.3, 2.1, 5.8],
        "P/E": [55.0, 120.0, 30.0, np.nan, 18.0],
    })


@pytest.fixture
def sample_nlp_with_keywords() -> pd.DataFrame:
    """Synthetic NLP features DataFrame with keyword_score rows."""
    dates = ["2023-02-22", "2023-08-23", "2024-02-21", "2024-08-28"]
    themes = ["ai_accelerated_computing", "data_center", "export_controls_china"]
    rows = []
    for d in dates:
        for t in themes:
            rows.append({
                "filing_date": d,
                "source_available_date": d,
                "source_accession": f"acc_{d}",
                "section": "mda",
                "feature_type": "keyword_score",
                "feature_name": t,
                "value": np.random.default_rng(42).random(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: plot_revenue_growth_trend
# ---------------------------------------------------------------------------

class TestRevenueGrowthTrend:
    def test_basic_rendering(self, chart_gen, sample_metrics_with_growth):
        """Chart renders with valid growth data and returns a Path."""
        result = chart_gen.plot_revenue_growth_trend(sample_metrics_with_growth)

        assert result is not None
        assert isinstance(result, Path)
        assert result.exists()
        assert "revenue_growth_trend.png" in str(result)

    def test_saves_to_correct_path(self, chart_gen, sample_metrics_with_growth):
        """Chart file is saved inside outputs/figures/."""
        result = chart_gen.plot_revenue_growth_trend(sample_metrics_with_growth)

        assert "figures" in str(result.parent)
        assert result.suffix == ".png"

    def test_produces_exhibit_record(self, chart_gen, sample_metrics_with_growth):
        """Chart appends a valid ExhibitRecord to the generator's exhibits list."""
        initial_count = len(chart_gen.exhibits)
        chart_gen.plot_revenue_growth_trend(sample_metrics_with_growth)

        assert len(chart_gen.exhibits) == initial_count + 1
        record = chart_gen.exhibits[-1]
        assert isinstance(record, ExhibitRecord)
        assert record.exhibit_id
        assert record.title
        assert record.source_caption
        assert record.file_path

    def test_empty_metrics(self, chart_gen):
        """Empty DataFrame produces a fallback placeholder chart."""
        result = chart_gen.plot_revenue_growth_trend(pd.DataFrame())

        assert result is not None
        assert isinstance(result, Path)
        assert result.exists()

    def test_no_growth_rows(self, chart_gen):
        """Metrics with no revenue_growth_YoY rows produce a fallback."""
        df = pd.DataFrame({
            "fiscal_period": ["FY2024"],
            "metric_name": ["gross_margin"],
            "metric_value": [0.75],
        })
        result = chart_gen.plot_revenue_growth_trend(df)

        assert result is not None
        assert result.exists()

    def test_single_data_point(self, chart_gen):
        """Chart handles a single growth data point without error."""
        df = pd.DataFrame({
            "fiscal_period": ["FY2025"],
            "metric_name": ["revenue_growth_YoY"],
            "metric_value": [0.94],
        })
        result = chart_gen.plot_revenue_growth_trend(df)

        assert result is not None
        assert result.exists()

    def test_nan_values(self, chart_gen):
        """Chart handles NaN growth values gracefully."""
        df = pd.DataFrame({
            "fiscal_period": ["FY2023", "FY2024", "FY2025"],
            "metric_name": ["revenue_growth_YoY"] * 3,
            "metric_value": [0.5, np.nan, 0.94],
        })
        result = chart_gen.plot_revenue_growth_trend(df)

        assert result is not None
        assert result.exists()

    def test_negative_growth_values(self, chart_gen):
        """Chart renders negative growth values (bars should be red)."""
        df = pd.DataFrame({
            "fiscal_period": ["FY2022", "FY2023"],
            "metric_name": ["revenue_growth_YoY"] * 2,
            "metric_value": [-0.17, -0.05],
        })
        result = chart_gen.plot_revenue_growth_trend(df)

        assert result is not None
        assert result.exists()


# ---------------------------------------------------------------------------
# Tests: plot_peer_multiples_comparison
# ---------------------------------------------------------------------------

class TestPeerMultiplesComparison:
    def test_basic_rendering(self, chart_gen, sample_peer_multiples):
        """Chart renders with valid peer data and returns a Path."""
        result = chart_gen.plot_peer_multiples_comparison(sample_peer_multiples)

        assert result is not None
        assert isinstance(result, Path)
        assert result.exists()
        assert "peer_multiples_comparison.png" in str(result)

    def test_saves_to_correct_path(self, chart_gen, sample_peer_multiples):
        """Chart file is saved inside outputs/figures/."""
        result = chart_gen.plot_peer_multiples_comparison(sample_peer_multiples)

        assert "figures" in str(result.parent)
        assert result.suffix == ".png"

    def test_produces_exhibit_record(self, chart_gen, sample_peer_multiples):
        """Chart appends a valid ExhibitRecord."""
        initial_count = len(chart_gen.exhibits)
        chart_gen.plot_peer_multiples_comparison(sample_peer_multiples)

        assert len(chart_gen.exhibits) == initial_count + 1
        record = chart_gen.exhibits[-1]
        assert isinstance(record, ExhibitRecord)
        assert record.exhibit_id
        assert record.file_path

    def test_empty_peer_data(self, chart_gen):
        """Empty DataFrame produces a fallback placeholder."""
        result = chart_gen.plot_peer_multiples_comparison(pd.DataFrame())

        assert result is not None
        assert result.exists()

    def test_missing_multiple_columns(self, chart_gen):
        """DataFrame without EV/Revenue or P/E columns produces a fallback."""
        df = pd.DataFrame({
            "ticker": ["NVDA", "AMD"],
            "market_cap": [1e12, 2e11],
        })
        result = chart_gen.plot_peer_multiples_comparison(df)

        assert result is not None
        assert result.exists()

    def test_single_peer(self, chart_gen):
        """Chart handles a single peer row without error."""
        df = pd.DataFrame({
            "ticker": ["NVDA"],
            "EV/Revenue": [25.0],
            "P/E": [55.0],
        })
        result = chart_gen.plot_peer_multiples_comparison(df)

        assert result is not None
        assert result.exists()

    def test_nan_multiples(self, chart_gen):
        """Chart handles NaN values in multiple columns gracefully."""
        df = pd.DataFrame({
            "ticker": ["NVDA", "AMD", "INTC"],
            "EV/Revenue": [25.0, np.nan, 2.1],
            "P/E": [np.nan, np.nan, 18.0],
        })
        result = chart_gen.plot_peer_multiples_comparison(df)

        assert result is not None
        assert result.exists()

    def test_all_nan_multiples(self, chart_gen):
        """All-NaN multiples produce a fallback chart."""
        df = pd.DataFrame({
            "ticker": ["NVDA", "AMD"],
            "EV/Revenue": [np.nan, np.nan],
            "P/E": [np.nan, np.nan],
        })
        result = chart_gen.plot_peer_multiples_comparison(df)

        assert result is not None
        assert result.exists()

    def test_alternative_column_names(self, chart_gen):
        """Chart recognizes alternative column naming conventions."""
        df = pd.DataFrame({
            "ticker": ["NVDA", "AMD"],
            "ev_to_revenue": [25.0, 10.5],
            "pe_ratio": [55.0, 120.0],
        })
        result = chart_gen.plot_peer_multiples_comparison(df)

        assert result is not None
        assert result.exists()
        assert "peer_multiples_comparison.png" in str(result)


# ---------------------------------------------------------------------------
# Tests: plot_keyword_theme_heatmap
# ---------------------------------------------------------------------------

class TestKeywordThemeHeatmap:
    def test_basic_rendering(self, chart_gen, sample_nlp_with_keywords):
        """Chart renders with valid keyword score data and returns a Path."""
        result = chart_gen.plot_keyword_theme_heatmap(sample_nlp_with_keywords)

        assert result is not None
        assert isinstance(result, Path)
        assert result.exists()
        assert "keyword_theme_heatmap.png" in str(result)

    def test_saves_to_correct_path(self, chart_gen, sample_nlp_with_keywords):
        """Chart file is saved inside outputs/figures/."""
        result = chart_gen.plot_keyword_theme_heatmap(sample_nlp_with_keywords)

        assert "figures" in str(result.parent)
        assert result.suffix == ".png"

    def test_produces_exhibit_record(self, chart_gen, sample_nlp_with_keywords):
        """Chart appends a valid ExhibitRecord."""
        initial_count = len(chart_gen.exhibits)
        chart_gen.plot_keyword_theme_heatmap(sample_nlp_with_keywords)

        assert len(chart_gen.exhibits) == initial_count + 1
        record = chart_gen.exhibits[-1]
        assert isinstance(record, ExhibitRecord)
        assert record.exhibit_id
        assert record.title
        assert record.source_caption

    def test_empty_nlp_data(self, chart_gen):
        """Empty DataFrame produces a fallback placeholder."""
        result = chart_gen.plot_keyword_theme_heatmap(pd.DataFrame())

        assert result is not None
        assert result.exists()

    def test_no_keyword_score_rows(self, chart_gen):
        """NLP data without keyword_score feature_type produces a fallback."""
        df = pd.DataFrame({
            "filing_date": ["2024-02-21"],
            "section": ["mda"],
            "feature_type": ["tfidf_similarity"],
            "feature_name": ["mda"],
            "value": [0.85],
        })
        result = chart_gen.plot_keyword_theme_heatmap(df)

        assert result is not None
        assert result.exists()

    def test_single_filing_date(self, chart_gen):
        """Chart handles a single filing date (one row in heatmap)."""
        df = pd.DataFrame({
            "filing_date": ["2024-02-21", "2024-02-21"],
            "section": ["mda", "mda"],
            "feature_type": ["keyword_score", "keyword_score"],
            "feature_name": ["ai_accelerated_computing", "data_center"],
            "value": [0.05, 0.03],
        })
        result = chart_gen.plot_keyword_theme_heatmap(df)

        assert result is not None
        assert result.exists()

    def test_single_theme(self, chart_gen):
        """Chart handles a single theme across multiple dates."""
        df = pd.DataFrame({
            "filing_date": ["2023-02-22", "2024-02-21"],
            "section": ["mda", "mda"],
            "feature_type": ["keyword_score", "keyword_score"],
            "feature_name": ["ai_accelerated_computing", "ai_accelerated_computing"],
            "value": [0.03, 0.07],
        })
        result = chart_gen.plot_keyword_theme_heatmap(df)

        assert result is not None
        assert result.exists()

    def test_nan_keyword_values(self, chart_gen):
        """Chart handles NaN keyword score values gracefully."""
        df = pd.DataFrame({
            "filing_date": ["2023-02-22", "2024-02-21", "2024-08-28"],
            "section": ["mda"] * 3,
            "feature_type": ["keyword_score"] * 3,
            "feature_name": ["ai_accelerated_computing"] * 3,
            "value": [0.05, np.nan, 0.08],
        })
        result = chart_gen.plot_keyword_theme_heatmap(df)

        assert result is not None
        assert result.exists()

    def test_zero_keyword_values(self, chart_gen):
        """Chart handles all-zero keyword scores."""
        df = pd.DataFrame({
            "filing_date": ["2023-02-22", "2024-02-21"],
            "section": ["mda", "mda"],
            "feature_type": ["keyword_score", "keyword_score"],
            "feature_name": ["data_center", "data_center"],
            "value": [0.0, 0.0],
        })
        result = chart_gen.plot_keyword_theme_heatmap(df)

        assert result is not None
        assert result.exists()


# ---------------------------------------------------------------------------
# Tests: ExhibitRecord metadata consistency across all additional charts
# ---------------------------------------------------------------------------

class TestExhibitRecordConsistency:
    def test_all_additional_charts_produce_unique_exhibit_ids(
        self, chart_gen, sample_metrics_with_growth,
        sample_peer_multiples, sample_nlp_with_keywords,
    ):
        """Each additional chart produces a unique exhibit_id."""
        chart_gen.plot_revenue_growth_trend(sample_metrics_with_growth)
        chart_gen.plot_peer_multiples_comparison(sample_peer_multiples)
        chart_gen.plot_keyword_theme_heatmap(sample_nlp_with_keywords)

        ids = [ex.exhibit_id for ex in chart_gen.exhibits]
        assert len(ids) == len(set(ids)), "Exhibit IDs must be unique"

    def test_all_exhibit_file_paths_exist(
        self, chart_gen, sample_metrics_with_growth,
        sample_peer_multiples, sample_nlp_with_keywords,
    ):
        """All exhibit file_path values point to existing files."""
        chart_gen.plot_revenue_growth_trend(sample_metrics_with_growth)
        chart_gen.plot_peer_multiples_comparison(sample_peer_multiples)
        chart_gen.plot_keyword_theme_heatmap(sample_nlp_with_keywords)

        for ex in chart_gen.exhibits:
            assert Path(ex.file_path).exists(), f"Missing file: {ex.file_path}"

    def test_exhibit_records_have_required_fields(
        self, chart_gen, sample_metrics_with_growth,
        sample_peer_multiples, sample_nlp_with_keywords,
    ):
        """All ExhibitRecords have non-empty required fields."""
        chart_gen.plot_revenue_growth_trend(sample_metrics_with_growth)
        chart_gen.plot_peer_multiples_comparison(sample_peer_multiples)
        chart_gen.plot_keyword_theme_heatmap(sample_nlp_with_keywords)

        for ex in chart_gen.exhibits:
            assert ex.exhibit_id, "exhibit_id must not be empty"
            assert ex.title, "title must not be empty"
            assert ex.source_caption, "source_caption must not be empty"
            assert ex.file_path, "file_path must not be empty"
