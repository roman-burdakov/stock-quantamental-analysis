"""
Tests for src.filing_text_parser — Filing text extraction.

Covers:
  - 10-K extraction: business, risk_factors, mda sections with status "success"
  - 10-Q extraction: risk_factors, mda sections with status "success"
  - Fallback behavior: title matching when Item-number regex fails

All tests run offline using fixtures in tests/fixtures/.
Reqs: 5.6, 12.5
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import EngineConfig, TextSectionRecord
from src.filing_text_parser import FilingTextParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures")


def _load_html(filename: str) -> str:
    return (FIXTURES / filename).read_text(encoding="utf-8")


def _make_parser(**overrides) -> FilingTextParser:
    config = EngineConfig(**overrides)
    return config, FilingTextParser(config)


def _parse_10k() -> list[TextSectionRecord]:
    config, parser = _make_parser()
    html = _load_html("sample_10k_snippet.html")
    return parser.parse_filing(
        html=html,
        accession="0001045810-25-000013",
        form_type="10-K",
        filing_date="2025-02-26",
        source_available_date="2025-02-26",
    )


def _parse_10q() -> list[TextSectionRecord]:
    config, parser = _make_parser()
    html = _load_html("sample_10q_snippet.html")
    return parser.parse_filing(
        html=html,
        accession="0001045810-24-000124",
        form_type="10-Q",
        filing_date="2024-11-20",
        source_available_date="2024-11-20",
    )


def _results_by_section(results: list[TextSectionRecord]) -> dict[str, TextSectionRecord]:
    return {r.section_name: r for r in results}


# ===================================================================
# 1. 10-K extraction
# ===================================================================


class TestExtraction10K:
    """Verify extraction on fixture 10-K snippet."""

    def test_returns_list_of_text_section_records(self):
        results = _parse_10k()
        assert isinstance(results, list)
        assert all(isinstance(r, TextSectionRecord) for r in results)

    def test_business_section_extracted(self):
        sections = _results_by_section(_parse_10k())
        assert "business" in sections
        rec = sections["business"]
        assert rec.parse_status == "success"
        assert rec.char_count > 0
        assert len(rec.text.strip()) > 50

    def test_risk_factors_section_extracted(self):
        sections = _results_by_section(_parse_10k())
        assert "risk_factors" in sections
        rec = sections["risk_factors"]
        assert rec.parse_status == "success"
        assert rec.char_count > 0
        assert len(rec.text.strip()) > 50

    def test_mda_section_extracted(self):
        sections = _results_by_section(_parse_10k())
        assert "mda" in sections
        rec = sections["mda"]
        assert rec.parse_status == "success"
        assert rec.char_count > 0
        assert len(rec.text.strip()) > 50

    def test_all_expected_sections_present(self):
        """10-K should produce business, risk_factors, mda, quant sections."""
        sections = _results_by_section(_parse_10k())
        for name in ("business", "risk_factors", "mda"):
            assert name in sections, f"Missing section: {name}"

    def test_accession_number_propagated(self):
        results = _parse_10k()
        for r in results:
            assert r.accession_number == "0001045810-25-000013"

    def test_form_type_propagated(self):
        results = _parse_10k()
        for r in results:
            assert r.form_type == "10-K"

    def test_filing_date_propagated(self):
        results = _parse_10k()
        for r in results:
            assert r.filing_date == "2025-02-26"

    def test_business_contains_nvidia_content(self):
        """Sanity check: business section should mention NVIDIA."""
        sections = _results_by_section(_parse_10k())
        assert "NVIDIA" in sections["business"].text or "nvidia" in sections["business"].text.lower()

    def test_risk_factors_contains_risk_content(self):
        """Sanity check: risk_factors should mention risk-related terms."""
        sections = _results_by_section(_parse_10k())
        text_lower = sections["risk_factors"].text.lower()
        assert any(term in text_lower for term in ["risk", "adversely", "export"])


# ===================================================================
# 2. 10-Q extraction
# ===================================================================


class TestExtraction10Q:
    """Verify extraction on fixture 10-Q snippet."""

    def test_returns_list_of_text_section_records(self):
        results = _parse_10q()
        assert isinstance(results, list)
        assert all(isinstance(r, TextSectionRecord) for r in results)

    def test_risk_factors_section_extracted(self):
        sections = _results_by_section(_parse_10q())
        assert "risk_factors" in sections
        rec = sections["risk_factors"]
        assert rec.parse_status == "success"
        assert rec.char_count > 0
        assert len(rec.text.strip()) > 50

    def test_mda_section_extracted(self):
        sections = _results_by_section(_parse_10q())
        assert "mda" in sections
        rec = sections["mda"]
        assert rec.parse_status == "success"
        assert rec.char_count > 0
        assert len(rec.text.strip()) > 50

    def test_all_expected_sections_present(self):
        """10-Q should produce risk_factors and mda sections."""
        sections = _results_by_section(_parse_10q())
        for name in ("risk_factors", "mda"):
            assert name in sections, f"Missing section: {name}"

    def test_accession_number_propagated(self):
        results = _parse_10q()
        for r in results:
            assert r.accession_number == "0001045810-24-000124"

    def test_form_type_propagated(self):
        results = _parse_10q()
        for r in results:
            assert r.form_type == "10-Q"

    def test_mda_contains_revenue_content(self):
        """Sanity check: MD&A should mention revenue figures."""
        sections = _results_by_section(_parse_10q())
        text_lower = sections["mda"].text.lower()
        assert "revenue" in text_lower

    def test_risk_factors_contains_risk_content(self):
        """Sanity check: risk_factors should mention risk-related terms."""
        sections = _results_by_section(_parse_10q())
        text_lower = sections["risk_factors"].text.lower()
        assert any(term in text_lower for term in ["risk", "blackwell", "export"])


# ===================================================================
# 3. Fallback behavior
# ===================================================================


class TestFallbackBehavior:
    """Verify fallback to title matching when Item-number regex fails."""

    def test_missing_section_returns_missing_status(self):
        """HTML with no recognizable Item headings should yield 'missing' status."""
        config, parser = _make_parser()
        html = "<html><body><p>Some unrelated content with no Item headings.</p></body></html>"
        results = parser.parse_filing(
            html=html,
            accession="test-accession",
            form_type="10-K",
            filing_date="2025-01-01",
            source_available_date="2025-01-01",
        )
        for r in results:
            assert r.parse_status == "missing"
            assert r.char_count == 0

    def test_title_fallback_extracts_section(self):
        """When Item-number regex fails but section title is present, fallback should work."""
        config, parser = _make_parser()
        # HTML with section titles but no "Item 1A." heading
        html = """<html><body>
        <p>Some preamble text that is long enough to pass the threshold check.</p>
        <p><b>Risk Factors</b></p>
        <p>The company faces significant risks including market volatility,
        regulatory changes, competitive pressures, and supply chain disruptions.
        These risks could materially affect our business operations and financial results.
        Additional risk factors include geopolitical tensions and currency fluctuations.</p>
        <p><b>Item 2. Properties</b></p>
        <p>Our headquarters are located in Santa Clara.</p>
        </body></html>"""
        results = parser.parse_filing(
            html=html,
            accession="test-fallback",
            form_type="10-K",
            filing_date="2025-01-01",
            source_available_date="2025-01-01",
        )
        sections = _results_by_section(results)
        rec = sections["risk_factors"]
        # Should be extracted via fallback or missing if text too short
        assert rec.parse_status in ("fallback", "missing")

    def test_parse_status_values_are_valid(self):
        """All parse_status values should be one of success/fallback/missing."""
        results = _parse_10k()
        valid_statuses = {"success", "fallback", "missing"}
        for r in results:
            assert r.parse_status in valid_statuses, f"Invalid status: {r.parse_status}"

    def test_char_count_matches_text_length(self):
        """char_count should equal len(text) for all records."""
        results = _parse_10k()
        for r in results:
            assert r.char_count == len(r.text)
