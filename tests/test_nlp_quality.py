"""
Tests for NLP quality gate: text extraction quality check and NLP quality assessment.

Validates: Requirements 20.1, 20.2, 20.3
"""

from __future__ import annotations

import pytest

from src.config import (
    ComponentStatus,
    ComponentStatusEnum,
    EngineConfig,
    TextSectionRecord,
)
from src.filing_text_parser import FilingTextParser
from src.nlp_features import NLPFeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(
    section_name: str = "mda",
    char_count: int = 1000,
    accession: str = "acc-001",
    filing_date: str = "2024-02-28",
) -> TextSectionRecord:
    """Build a TextSectionRecord with a given char_count."""
    text = "x" * char_count
    return TextSectionRecord(
        accession_number=accession,
        form_type="10-K",
        section_name=section_name,
        text=text,
        char_count=char_count,
        filing_date=filing_date,
        source_available_date=filing_date,
        parse_status="success" if char_count > 0 else "missing",
    )


# ---------------------------------------------------------------------------
# 1. check_extraction_quality: diagnostic_only when char_count below threshold
# ---------------------------------------------------------------------------


class TestExtractionQualityThresholds:
    """Req 20.1: diagnostic_only when section char_count is below threshold."""

    def test_mda_below_threshold_single_filing(self):
        """A single filing with MD&A below 500 chars → diagnostic_only."""
        records = [
            _make_section("mda", char_count=100, accession="acc-001"),
            _make_section("risk_factors", char_count=400, accession="acc-001"),
        ]
        parser = FilingTextParser()
        status = parser.check_extraction_quality(records)

        assert isinstance(status, ComponentStatus)
        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY

    def test_risk_factors_below_threshold(self):
        """A single filing with Risk Factors below 300 chars → diagnostic_only."""
        records = [
            _make_section("mda", char_count=600, accession="acc-001"),
            _make_section("risk_factors", char_count=100, accession="acc-001"),
        ]
        parser = FilingTextParser()
        status = parser.check_extraction_quality(records)

        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY

    def test_both_above_threshold_is_usable(self):
        """Both sections above threshold → usable."""
        records = [
            _make_section("mda", char_count=600, accession="acc-001"),
            _make_section("risk_factors", char_count=400, accession="acc-001"),
        ]
        parser = FilingTextParser()
        status = parser.check_extraction_quality(records)

        assert status.status == ComponentStatusEnum.USABLE

    def test_custom_thresholds_from_config(self):
        """Custom thresholds from config are respected."""
        config = EngineConfig()
        config.min_nlp_char_mda = 2000
        config.min_nlp_char_risk = 1000

        records = [
            _make_section("mda", char_count=1500, accession="acc-001"),
            _make_section("risk_factors", char_count=800, accession="acc-001"),
        ]
        parser = FilingTextParser(config)
        status = parser.check_extraction_quality(records, config)

        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY


# ---------------------------------------------------------------------------
# 2. check_extraction_quality: diagnostic_only when >50% filings have poor extraction
# ---------------------------------------------------------------------------


class TestExtractionQualityMajority:
    """Req 20.2: diagnostic_only when >50% of filings have poor extraction."""

    def test_majority_poor_triggers_diagnostic(self):
        """3 out of 5 filings (60%) below threshold → diagnostic_only."""
        records = []
        for i in range(5):
            accession = f"acc-{i:03d}"
            if i < 3:
                # Poor: MD&A below 500
                records.append(_make_section("mda", char_count=100, accession=accession))
                records.append(_make_section("risk_factors", char_count=400, accession=accession))
            else:
                # Good: both above threshold
                records.append(_make_section("mda", char_count=1000, accession=accession))
                records.append(_make_section("risk_factors", char_count=500, accession=accession))

        parser = FilingTextParser()
        status = parser.check_extraction_quality(records)

        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY
        assert status.details is not None
        assert status.details["poor_count"] == 3
        assert status.details["total_count"] == 5

    def test_minority_poor_is_usable(self):
        """2 out of 5 filings (40%) below threshold → usable."""
        records = []
        for i in range(5):
            accession = f"acc-{i:03d}"
            if i < 2:
                # Poor
                records.append(_make_section("mda", char_count=100, accession=accession))
                records.append(_make_section("risk_factors", char_count=400, accession=accession))
            else:
                # Good
                records.append(_make_section("mda", char_count=1000, accession=accession))
                records.append(_make_section("risk_factors", char_count=500, accession=accession))

        parser = FilingTextParser()
        status = parser.check_extraction_quality(records)

        assert status.status == ComponentStatusEnum.USABLE
        assert status.details["poor_count"] == 2

    def test_exactly_50pct_is_usable(self):
        """Exactly 50% poor (not >50%) → usable."""
        records = []
        for i in range(4):
            accession = f"acc-{i:03d}"
            if i < 2:
                records.append(_make_section("mda", char_count=100, accession=accession))
                records.append(_make_section("risk_factors", char_count=400, accession=accession))
            else:
                records.append(_make_section("mda", char_count=1000, accession=accession))
                records.append(_make_section("risk_factors", char_count=500, accession=accession))

        parser = FilingTextParser()
        status = parser.check_extraction_quality(records)

        assert status.status == ComponentStatusEnum.USABLE

    def test_no_relevant_sections_is_unavailable(self):
        """No MD&A or Risk Factors sections → unavailable."""
        records = [
            _make_section("business", char_count=5000, accession="acc-001"),
        ]
        parser = FilingTextParser()
        status = parser.check_extraction_quality(records)

        assert status.status == ComponentStatusEnum.UNAVAILABLE


# ---------------------------------------------------------------------------
# 3. assess_nlp_quality: diagnostic status propagates to scorecard
# ---------------------------------------------------------------------------


class TestNLPQualityPropagation:
    """Req 20.3: NLP diagnostic_only status propagates from extraction quality."""

    def test_poor_extraction_propagates_to_nlp(self):
        """When extraction is diagnostic_only, NLP should also be diagnostic_only."""
        records = [
            _make_section("mda", char_count=100, accession="acc-001"),
            _make_section("risk_factors", char_count=100, accession="acc-001"),
        ]
        extractor = NLPFeatureExtractor()
        status = extractor.assess_nlp_quality(records)

        assert isinstance(status, ComponentStatus)
        assert status.component_name == "nlp_signal"
        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY
        assert "diagnostic-only" in status.reason.lower() or "diagnostic" in status.reason.lower()

    def test_good_extraction_makes_nlp_usable(self):
        """When extraction is usable, NLP should also be usable."""
        records = [
            _make_section("mda", char_count=1000, accession="acc-001"),
            _make_section("risk_factors", char_count=500, accession="acc-001"),
        ]
        extractor = NLPFeatureExtractor()
        status = extractor.assess_nlp_quality(records)

        assert status.component_name == "nlp_signal"
        assert status.status == ComponentStatusEnum.USABLE

    def test_unavailable_extraction_propagates(self):
        """When extraction is unavailable, NLP should also be unavailable."""
        records = [
            _make_section("business", char_count=5000, accession="acc-001"),
        ]
        extractor = NLPFeatureExtractor()
        status = extractor.assess_nlp_quality(records)

        assert status.status == ComponentStatusEnum.UNAVAILABLE

    def test_nlp_quality_with_mixed_filings(self):
        """Majority good filings → NLP usable even with some poor ones."""
        records = []
        # 1 poor filing
        records.append(_make_section("mda", char_count=50, accession="acc-001"))
        records.append(_make_section("risk_factors", char_count=50, accession="acc-001"))
        # 3 good filings
        for i in range(2, 5):
            acc = f"acc-{i:03d}"
            records.append(_make_section("mda", char_count=2000, accession=acc))
            records.append(_make_section("risk_factors", char_count=1000, accession=acc))

        extractor = NLPFeatureExtractor()
        status = extractor.assess_nlp_quality(records)

        assert status.status == ComponentStatusEnum.USABLE

    def test_nlp_quality_details_include_counts(self):
        """Status details should include poor_count and total_count."""
        records = [
            _make_section("mda", char_count=1000, accession="acc-001"),
            _make_section("risk_factors", char_count=500, accession="acc-001"),
        ]
        extractor = NLPFeatureExtractor()
        status = extractor.assess_nlp_quality(records)

        assert status.details is not None
        assert "poor_count" in status.details
        assert "total_count" in status.details
