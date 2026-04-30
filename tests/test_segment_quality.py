"""
Tests for segment quality check and footnote generation.

Validates: Requirements 19.1, 19.2, 19.3
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import ChartStatus, EngineConfig
from src.segment_revenue import SegmentRevenueNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment_row(
    category: str,
    fiscal_year: int = 2024,
    value: float = 1000.0,
    extraction_method: str = "xbrl_dimension",
) -> dict:
    """Build a single normalized segment row for testing."""
    return {
        "ticker": "NVDA",
        "fiscal_period": "FY",
        "fiscal_year": fiscal_year,
        "filing_date": f"{fiscal_year}-02-21",
        "source_available_date": f"{fiscal_year}-02-21",
        "source_accession": f"0001045810-{fiscal_year}-000001",
        "original_label": category.replace("_", " ").title(),
        "normalized_category": category,
        "value": value,
        "unit": "USD",
        "extraction_method": extraction_method,
        "mapping_notes": "",
    }


# ---------------------------------------------------------------------------
# 1. Chart blocked when all categories are "Other"
# ---------------------------------------------------------------------------


class TestSegmentQualityBlocked:
    """Req 19.1: Chart suppressed when all categories are Other/Unclassified."""

    def test_all_other_blocks_chart(self):
        """When every row is 'other', chart should not be renderable."""
        rows = [
            _make_segment_row("other", fiscal_year=2024, value=5000),
            _make_segment_row("other", fiscal_year=2023, value=4000),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df)

        assert isinstance(status, ChartStatus)
        assert status.renderable is False
        assert status.reason == "all_categories_other"
        assert status.fallback_message is not None

    def test_all_unclassified_blocks_chart(self):
        """'Unclassified' should also trigger blocking."""
        rows = [
            _make_segment_row("unclassified", fiscal_year=2024, value=5000),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df)

        assert status.renderable is False
        assert status.reason == "all_categories_other"

    def test_mix_of_other_and_unclassified_blocks(self):
        """Mix of 'other' and 'unclassified' should still block."""
        rows = [
            _make_segment_row("other", fiscal_year=2024, value=3000),
            _make_segment_row("unclassified", fiscal_year=2024, value=2000),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df)

        assert status.renderable is False

    def test_empty_dataframe_blocks_chart(self):
        """Empty data should block the chart."""
        df = pd.DataFrame(columns=[
            "ticker", "fiscal_period", "fiscal_year", "filing_date",
            "source_available_date", "source_accession", "original_label",
            "normalized_category", "value", "unit", "extraction_method",
            "mapping_notes",
        ])
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df)

        assert status.renderable is False
        assert status.reason == "no_segment_data"

    def test_valid_categories_allow_chart(self):
        """When at least one real category exists, chart is renderable."""
        rows = [
            _make_segment_row("data_center", fiscal_year=2024, value=50000),
            _make_segment_row("gaming", fiscal_year=2024, value=10000),
            _make_segment_row("other", fiscal_year=2024, value=500),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df)

        assert status.renderable is True


# ---------------------------------------------------------------------------
# 2. Reconciliation warning when totals differ >5%
# ---------------------------------------------------------------------------


class TestSegmentReconciliation:
    """Req 19.2: Warn when segment totals differ from reported revenue by >5%."""

    def test_reconciliation_warning_when_diff_exceeds_5pct(self):
        """Segment total that differs from reported revenue by >5% triggers warning."""
        rows = [
            _make_segment_row("data_center", fiscal_year=2024, value=50000),
            _make_segment_row("gaming", fiscal_year=2024, value=10000),
        ]
        df = pd.DataFrame(rows)
        # Reported revenue is 100000, segments sum to 60000 → 40% diff
        reported = {2024: 100000.0}
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df, reported_revenue=reported)

        assert status.renderable is True
        assert status.reason == "reconciliation_warning"
        assert status.fallback_message is not None
        assert "40.0%" in status.fallback_message

    def test_no_warning_when_within_5pct(self):
        """Segment total within 5% of reported revenue → no warning."""
        rows = [
            _make_segment_row("data_center", fiscal_year=2024, value=96000),
            _make_segment_row("gaming", fiscal_year=2024, value=4000),
        ]
        df = pd.DataFrame(rows)
        reported = {2024: 100000.0}
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df, reported_revenue=reported)

        assert status.renderable is True
        assert status.reason == "ok"

    def test_no_reported_revenue_no_warning(self):
        """Without reported revenue, no reconciliation check is performed."""
        rows = [
            _make_segment_row("data_center", fiscal_year=2024, value=50000),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        status = normalizer.check_segment_quality(df)

        assert status.renderable is True
        assert status.reason == "ok"


# ---------------------------------------------------------------------------
# 3. Footnote generation
# ---------------------------------------------------------------------------


class TestSegmentFootnotes:
    """Req 19.3: Footnotes label segment type and fallback extraction years."""

    def test_reportable_segments_footnote(self):
        """Chart with reportable segments gets appropriate footnote."""
        rows = [
            _make_segment_row("compute_and_networking", fiscal_year=2024, value=47000),
            _make_segment_row("graphics", fiscal_year=2024, value=15000),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        footnotes = normalizer.generate_footnotes(df)

        assert any("reportable segments" in fn.lower() for fn in footnotes)

    def test_market_platform_footnote(self):
        """Chart with market/platform categories gets appropriate footnote."""
        rows = [
            _make_segment_row("data_center", fiscal_year=2024, value=50000),
            _make_segment_row("gaming", fiscal_year=2024, value=10000),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        footnotes = normalizer.generate_footnotes(df)

        assert any("market/platform" in fn.lower() for fn in footnotes)

    def test_mixed_segments_footnote(self):
        """Chart with both types gets a mixed footnote."""
        rows = [
            _make_segment_row("compute_and_networking", fiscal_year=2024, value=47000),
            _make_segment_row("data_center", fiscal_year=2023, value=30000),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        footnotes = normalizer.generate_footnotes(df)

        assert any("mix" in fn.lower() for fn in footnotes)

    def test_fallback_extraction_footnote(self):
        """Fiscal years using table_extraction get a fallback footnote."""
        rows = [
            _make_segment_row("data_center", fiscal_year=2024, value=50000, extraction_method="xbrl_dimension"),
            _make_segment_row("data_center", fiscal_year=2020, value=10000, extraction_method="table_extraction"),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        footnotes = normalizer.generate_footnotes(df)

        assert any("fallback" in fn.lower() for fn in footnotes)
        assert any("FY2020" in fn for fn in footnotes)

    def test_no_fallback_no_fallback_footnote(self):
        """When all extraction is XBRL, no fallback footnote appears."""
        rows = [
            _make_segment_row("data_center", fiscal_year=2024, value=50000, extraction_method="xbrl_dimension"),
        ]
        df = pd.DataFrame(rows)
        normalizer = SegmentRevenueNormalizer()
        footnotes = normalizer.generate_footnotes(df)

        assert not any("fallback" in fn.lower() for fn in footnotes)

    def test_empty_dataframe_no_footnotes(self):
        """Empty data produces no footnotes."""
        df = pd.DataFrame(columns=[
            "ticker", "fiscal_period", "fiscal_year", "filing_date",
            "source_available_date", "source_accession", "original_label",
            "normalized_category", "value", "unit", "extraction_method",
            "mapping_notes",
        ])
        normalizer = SegmentRevenueNormalizer()
        footnotes = normalizer.generate_footnotes(df)

        assert footnotes == []
