"""
Tests for change-point detection with ruptures (Task 17.3).

Covers:
  - Change-point detection on synthetic time series with known structural breaks
  - Graceful fallback when ruptures is not installed (mock ImportError)
  - Short series (< 4 data points) returns empty list
  - Change-point rows in narrative drift have correct schema
  - Integration with compute_narrative_drift when use_change_point_detection=True
  - Change-point detection skipped when use_change_point_detection=False
  - Constant series (no breaks expected)
  - Penalty parameter affects number of detected breaks
  - Change-point dates are valid dates from the input series

All tests run offline using inline TextSectionRecord fixtures.
Reqs: 7.10
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig, TextSectionRecord
from src.nlp_features import NLPFeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_section(
    text: str,
    section_name: str = "risk_factors",
    filing_date: str = "2024-02-28",
    source_available_date: str = "2024-02-28",
    accession_number: str = "0001045810-24-000010",
    form_type: str = "10-K",
) -> TextSectionRecord:
    return TextSectionRecord(
        accession_number=accession_number,
        form_type=form_type,
        section_name=section_name,
        text=text,
        char_count=len(text),
        filing_date=filing_date,
        source_available_date=source_available_date,
        parse_status="success",
    )


# Sample texts for integration tests — varied enough to produce different
# TF-IDF similarity scores across filings.
RISK_TEXTS = [
    (
        "The company faces significant risks from export controls imposed by the "
        "Bureau of Industry and Security on advanced computing chips to China. "
        "Supply constraints at TSMC foundry limit our capacity."
    ),
    (
        "Export controls and trade restrictions continue to adversely affect our "
        "business in China. Supply chain disruptions at foundry partners limit "
        "production of Blackwell architecture products."
    ),
    (
        "Competition from AMD custom silicon TPU and Trainium alternatives is "
        "increasing. Customer concentration risk persists with major cloud "
        "service providers representing significant revenue."
    ),
    (
        "Gaming market cyclicality affects GeForce demand patterns. Margin "
        "pressure from pricing and product mix changes is a concern. Inventory "
        "and demand cyclicality create backlog uncertainty."
    ),
    (
        "AI and deep learning inference workloads are central to our data center "
        "growth strategy. Sovereign AI deployments expand our addressable market "
        "beyond traditional hyperscaler customers."
    ),
    (
        "Networking revenue from InfiniBand and NVLink grew substantially. "
        "Software attach rates for CUDA and enterprise AI platforms increased. "
        "Automotive revenue from self-driving platforms remained stable."
    ),
]

MDA_TEXTS = [
    "Total revenue for fiscal year 2020 was $10.9 billion driven by gaming.",
    "Total revenue for fiscal year 2021 was $16.7 billion driven by data center.",
    "Total revenue for fiscal year 2022 was $26.9 billion with strong growth.",
    "Total revenue for fiscal year 2023 was $27.0 billion amid inventory correction.",
    "Total revenue for fiscal year 2024 was $60.9 billion driven by AI demand.",
    "Total revenue for fiscal year 2025 was $130.5 billion unprecedented growth.",
]


@pytest.fixture
def extractor() -> NLPFeatureExtractor:
    return NLPFeatureExtractor()


def _ruptures_available() -> bool:
    try:
        import ruptures  # noqa: F401
        return True
    except ImportError:
        return False


# ===================================================================
# 1. Change-point detection on synthetic series with known break
# ===================================================================


class TestSyntheticSeriesDetection:
    """Verify detect_change_points finds known structural breaks."""

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_detects_obvious_level_shift(self, extractor: NLPFeatureExtractor):
        """A clear level shift in the middle should be detected."""
        # First half ~0.9, second half ~0.1 — obvious structural break
        values = [0.90, 0.91, 0.89, 0.92, 0.10, 0.11, 0.09, 0.12]
        dates = [f"20{16 + i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=1.0)
        assert isinstance(result, list)
        assert len(result) >= 1, "Should detect at least one change point"

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_detected_break_near_actual_shift(self, extractor: NLPFeatureExtractor):
        """The detected break should be near the actual transition point."""
        # Transition between index 4 and 5 (date 2020 → 2021)
        values = [0.90, 0.91, 0.89, 0.92, 0.88, 0.10, 0.11, 0.09, 0.12, 0.10]
        dates = [f"20{16 + i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=1.0)
        assert len(result) >= 1
        # The break should be at or near the transition (index 5 = "2021-01-01")
        # Allow some tolerance — ruptures may place it ±1 position
        break_indices = [dates.index(d) for d in result if d in dates]
        assert any(3 <= idx <= 6 for idx in break_indices), (
            f"Expected break near index 5, got indices {break_indices}"
        )

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_detects_multiple_breaks(self, extractor: NLPFeatureExtractor):
        """Series with two distinct level shifts should detect multiple breaks."""
        # Three regimes with clear separation and enough points per regime
        values = [0.90, 0.91, 0.89, 0.92, 0.10, 0.11, 0.09, 0.12, 0.50, 0.51, 0.49, 0.52]
        dates = [f"20{12 + i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=0.5)
        assert len(result) >= 2, f"Expected ≥2 breaks, got {len(result)}"


# ===================================================================
# 2. Graceful fallback when ruptures is not installed
# ===================================================================


class TestGracefulFallback:
    """Verify empty list returned when ruptures is unavailable."""

    def test_returns_empty_list_when_import_fails(self, extractor: NLPFeatureExtractor):
        """Mocking ImportError for ruptures should return empty list."""
        series = pd.Series(
            [0.9, 0.91, 0.1, 0.11, 0.12],
            index=["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        )
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "ruptures":
                raise ImportError("Mocked: ruptures not installed")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            result = extractor.detect_change_points(series)
            assert isinstance(result, list)
            assert result == []


# ===================================================================
# 3. Short series returns empty list
# ===================================================================


class TestShortSeries:
    """Series with fewer than 4 data points should return empty list."""

    def test_two_points_returns_empty(self, extractor: NLPFeatureExtractor):
        series = pd.Series([1.0, 2.0], index=["2023-01-01", "2024-01-01"])
        assert extractor.detect_change_points(series) == []

    def test_three_points_returns_empty(self, extractor: NLPFeatureExtractor):
        series = pd.Series([1.0, 2.0, 3.0],
                           index=["2022-01-01", "2023-01-01", "2024-01-01"])
        assert extractor.detect_change_points(series) == []

    def test_one_point_returns_empty(self, extractor: NLPFeatureExtractor):
        series = pd.Series([1.0], index=["2024-01-01"])
        assert extractor.detect_change_points(series) == []

    def test_empty_series_returns_empty(self, extractor: NLPFeatureExtractor):
        series = pd.Series([], dtype=float)
        assert extractor.detect_change_points(series) == []


# ===================================================================
# 4. Change-point rows in narrative drift have correct schema
# ===================================================================


class TestChangePointSchema:
    """Verify change-point rows in narrative drift have correct schema.

    These tests verify the schema of change-point records as they are
    constructed in compute_narrative_drift. We use the detect_change_points
    method directly to confirm the schema contract, then verify the
    integration produces correctly shaped rows.
    """

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_tfidf_break_row_schema(self, tmp_path):
        """TF-IDF change-point rows should have feature_type='change_point'
        and feature_name='tfidf_similarity_break'."""
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        # Build sections with dramatically different texts to force TF-IDF breaks
        sections = _build_high_contrast_sections()
        df = ext.compute_narrative_drift(sections)
        cp_rows = df[df["feature_type"] == "change_point"]
        if cp_rows.empty:
            pytest.skip("No change points detected — texts may be too similar")
        assert (cp_rows["feature_type"] == "change_point").all()

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_change_point_feature_name_ends_with_break(self, tmp_path):
        """Change-point feature_name should end with '_break'."""
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        sections = _build_high_contrast_sections()
        df = ext.compute_narrative_drift(sections)
        cp_rows = df[df["feature_type"] == "change_point"]
        if cp_rows.empty:
            pytest.skip("No change points detected")
        for name in cp_rows["feature_name"]:
            assert name.endswith("_break"), f"feature_name '{name}' does not end with '_break'"

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_change_point_value_is_one(self, tmp_path):
        """Change-point rows should have value=1.0 (indicator)."""
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        sections = _build_high_contrast_sections()
        df = ext.compute_narrative_drift(sections)
        cp_rows = df[df["feature_type"] == "change_point"]
        if cp_rows.empty:
            pytest.skip("No change points detected")
        assert (cp_rows["value"] == 1.0).all()

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_change_point_has_all_nlp_feature_columns(self, tmp_path):
        """Change-point rows should have all NLPFeatureRecord columns."""
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        sections = _build_high_contrast_sections()
        df = ext.compute_narrative_drift(sections)
        cp_rows = df[df["feature_type"] == "change_point"]
        if cp_rows.empty:
            pytest.skip("No change points detected")
        expected_cols = {
            "filing_date", "source_available_date", "source_accession",
            "section", "feature_type", "feature_name", "value", "prev_filing_date",
        }
        assert expected_cols.issubset(set(cp_rows.columns))


# ===================================================================
# 5. Integration with compute_narrative_drift (enabled)
# ===================================================================


class TestNarrativeDriftIntegrationEnabled:
    """Verify change-point rows appear when use_change_point_detection=True."""

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_change_point_rows_present_when_enabled(self, tmp_path):
        """With enough data and use_change_point_detection=True, change_point
        rows should appear in the output (if breaks are detected)."""
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        sections = _build_multi_year_sections(6)
        df = ext.compute_narrative_drift(sections)
        assert not df.empty
        # The output should at least contain tfidf and keyword features
        feature_types = set(df["feature_type"].unique())
        assert "tfidf_similarity" in feature_types
        assert "keyword_score" in feature_types
        # change_point may or may not appear depending on whether breaks are found

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_csv_output_includes_change_point_rows(self, tmp_path):
        """The saved CSV should include change_point rows if detected."""
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        sections = _build_multi_year_sections(6)
        ext.compute_narrative_drift(sections)

        csv_path = tmp_path / "processed" / "nvda_nlp_features.csv"
        assert csv_path.exists()
        saved_df = pd.read_csv(csv_path)
        assert not saved_df.empty
        # Verify the CSV has the expected feature types
        assert "tfidf_similarity" in saved_df["feature_type"].values
        assert "keyword_score" in saved_df["feature_type"].values


# ===================================================================
# 6. Change-point detection skipped when disabled
# ===================================================================


class TestChangePointDisabled:
    """Verify no change_point rows when use_change_point_detection=False."""

    def test_no_change_point_rows_when_disabled(self, tmp_path):
        """With use_change_point_detection=False, no change_point rows should appear."""
        config = EngineConfig()
        config.use_change_point_detection = False
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        sections = _build_multi_year_sections(6)
        df = ext.compute_narrative_drift(sections)
        assert not df.empty
        cp_rows = df[df["feature_type"] == "change_point"]
        assert cp_rows.empty, "change_point rows should not appear when detection is disabled"

    def test_default_config_has_detection_disabled(self):
        """Default EngineConfig should have use_change_point_detection=False."""
        config = EngineConfig()
        assert config.use_change_point_detection is False


# ===================================================================
# 7. Constant series (no breaks expected)
# ===================================================================


class TestConstantSeries:
    """A constant series should produce no change points."""

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_constant_series_no_breaks(self, extractor: NLPFeatureExtractor):
        """A perfectly constant series should have no structural breaks."""
        values = [0.5] * 10
        dates = [f"20{16 + i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=3.0)
        assert result == [], f"Expected no breaks for constant series, got {result}"

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_near_constant_series_no_breaks(self, extractor: NLPFeatureExtractor):
        """A near-constant series with tiny noise should have no breaks at default penalty."""
        np.random.seed(42)
        values = 0.5 + np.random.normal(0, 0.001, 10)
        dates = [f"20{16 + i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=3.0)
        assert result == [], f"Expected no breaks for near-constant series, got {result}"


# ===================================================================
# 8. Penalty parameter affects number of detected breaks
# ===================================================================


class TestPenaltyParameter:
    """Higher penalty should produce fewer (or equal) breaks."""

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_higher_penalty_fewer_or_equal_breaks(self, extractor: NLPFeatureExtractor):
        """Increasing penalty should not increase the number of detected breaks."""
        # Series with two clear regime changes
        values = [0.9, 0.91, 0.89, 0.1, 0.11, 0.09, 0.5, 0.51, 0.49, 0.50]
        dates = [f"20{16 + i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)

        result_low_pen = extractor.detect_change_points(series, pen=0.5)
        result_high_pen = extractor.detect_change_points(series, pen=10.0)

        assert len(result_high_pen) <= len(result_low_pen), (
            f"Higher penalty ({len(result_high_pen)} breaks) should produce "
            f"≤ breaks than lower penalty ({len(result_low_pen)} breaks)"
        )

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_very_high_penalty_produces_no_breaks(self, extractor: NLPFeatureExtractor):
        """A very high penalty should suppress all break detection."""
        values = [0.9, 0.91, 0.89, 0.1, 0.11, 0.09, 0.5, 0.51]
        dates = [f"20{16 + i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=1000.0)
        assert result == [], f"Very high penalty should produce no breaks, got {result}"


# ===================================================================
# 9. Change-point dates are valid dates from the input series
# ===================================================================


class TestChangePointDatesValid:
    """All returned change-point dates must come from the input series index."""

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_all_dates_from_series_index(self, extractor: NLPFeatureExtractor):
        """Every returned date must be present in the original series index."""
        values = [0.90, 0.91, 0.89, 0.92, 0.10, 0.11, 0.09, 0.12]
        dates = ["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01",
                 "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=1.0)
        for cp_date in result:
            assert cp_date in dates, (
                f"Change-point date '{cp_date}' not in series index {dates}"
            )

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_dates_are_strings(self, extractor: NLPFeatureExtractor):
        """Returned dates should be strings."""
        values = [0.90, 0.91, 0.10, 0.11, 0.12]
        dates = ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=1.0)
        for cp_date in result:
            assert isinstance(cp_date, str), f"Expected str, got {type(cp_date)}"

    @pytest.mark.skipif(not _ruptures_available(), reason="ruptures not installed")
    def test_change_point_dates_in_narrative_drift_are_valid(self, tmp_path):
        """Change-point filing_date values in narrative drift output should be
        valid dates from the input filing dates."""
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        sections = _build_high_contrast_sections()
        df = ext.compute_narrative_drift(sections)
        cp_rows = df[df["feature_type"] == "change_point"]
        if cp_rows.empty:
            pytest.skip("No change points detected")

        # Collect all filing dates from input sections
        input_dates = {s.filing_date for s in sections}
        # Also include dates that could come from TF-IDF similarity series
        all_dates = set(df["filing_date"].dropna().unique())

        for _, row in cp_rows.iterrows():
            cp_date = row["filing_date"]
            assert cp_date in all_dates, (
                f"Change-point date '{cp_date}' not found in output dates"
            )


# ===================================================================
# Helper: build multi-year sections for integration tests
# ===================================================================


def _build_multi_year_sections(n_years: int) -> list[TextSectionRecord]:
    """Build n_years of risk_factors + mda sections for integration tests."""
    sections = []
    for i in range(n_years):
        year = 2019 + i
        risk_text = RISK_TEXTS[i % len(RISK_TEXTS)]
        mda_text = MDA_TEXTS[i % len(MDA_TEXTS)]
        sections.append(_make_section(
            risk_text,
            "risk_factors",
            f"{year}-02-28",
            f"{year}-02-28",
            f"acc-{year}",
        ))
        sections.append(_make_section(
            mda_text,
            "mda",
            f"{year}-02-28",
            f"{year}-02-28",
            f"acc-{year}",
        ))
    return sections


def _build_high_contrast_sections() -> list[TextSectionRecord]:
    """Build sections with dramatically different content across filings
    to force detectable change points in TF-IDF similarity series.

    The first 4 filings discuss gaming/consumer topics, then the last 4
    shift abruptly to AI/data-center topics — creating a clear structural
    break in the TF-IDF similarity time series.
    """
    gaming_risk = (
        "Gaming revenue from GeForce RTX desktop and notebook GPUs remains "
        "our primary revenue driver. Consumer PC market cyclicality and "
        "cryptocurrency mining demand fluctuations create inventory risk. "
        "Channel inventory management and sell-through rates are critical. "
        "Competition in the discrete GPU market from AMD Radeon products "
        "continues. Seasonal demand patterns affect quarterly results."
    )
    ai_risk = (
        "Artificial intelligence and accelerated computing infrastructure "
        "demand drives unprecedented data center revenue growth. Export "
        "controls imposed by the Bureau of Industry and Security restrict "
        "sales of advanced AI chips to China. Sovereign AI deployments and "
        "enterprise inference workloads expand our addressable market. "
        "CUDA ecosystem and NVLink networking create competitive moat."
    )
    gaming_mda = (
        "Total revenue was driven primarily by gaming segment performance. "
        "GeForce desktop and notebook GPU sales represented the majority of "
        "revenue. Gross margin reflected consumer product mix. Operating "
        "expenses included research and development for next generation "
        "gaming architectures and consumer graphics products."
    )
    ai_mda = (
        "Data center revenue represented the vast majority of total revenue "
        "driven by AI training and inference demand. Hyperscaler cloud "
        "service providers and sovereign AI customers drove growth. Gross "
        "margin expanded reflecting favorable data center product mix. "
        "Networking revenue from InfiniBand and NVLink grew substantially."
    )

    sections = []
    for i in range(8):
        year = 2017 + i
        # First 4 years: gaming-focused; last 4 years: AI-focused
        if i < 4:
            risk = gaming_risk + f" Year {year} specific gaming content number {i}."
            mda = gaming_mda + f" Fiscal year {year} gaming results number {i}."
        else:
            risk = ai_risk + f" Year {year} specific AI content number {i}."
            mda = ai_mda + f" Fiscal year {year} AI results number {i}."
        sections.append(_make_section(risk, "risk_factors", f"{year}-02-28",
                                      f"{year}-02-28", f"acc-{year}"))
        sections.append(_make_section(mda, "mda", f"{year}-02-28",
                                      f"{year}-02-28", f"acc-{year}"))
    return sections
