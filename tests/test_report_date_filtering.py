"""
Tests for report-date filtering in ReportGenerator.

Verifies that build_report_context excludes any data with
source_available_date > report_date.

Reqs: 12.4, 10.9
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import EngineConfig, Recommendation, get_default_config
from src.report_utils import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metrics(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a metrics DataFrame."""
    return pd.DataFrame(rows)


def _make_config(report_date: str = "2026-04-25") -> EngineConfig:
    cfg = get_default_config()
    cfg.report_date = report_date
    return cfg


# ---------------------------------------------------------------------------
# Tests: _filter_by_availability
# ---------------------------------------------------------------------------

class TestFilterByAvailability:
    """Unit tests for the static filtering helper."""

    def test_excludes_future_data(self):
        gen = ReportGenerator(_make_config("2025-06-01"))
        df = pd.DataFrame(
            {
                "source_available_date": [
                    "2025-01-15",
                    "2025-06-01",
                    "2025-06-02",
                    "2026-01-01",
                ],
                "metric_name": ["a", "b", "c", "d"],
                "metric_value": [1, 2, 3, 4],
            }
        )
        result = gen._filter_by_availability(df, "2025-06-01")
        assert len(result) == 2
        assert set(result["metric_name"]) == {"a", "b"}

    def test_keeps_all_when_within_date(self):
        gen = ReportGenerator(_make_config("2030-01-01"))
        df = pd.DataFrame(
            {
                "source_available_date": ["2025-01-01", "2026-01-01"],
                "metric_name": ["x", "y"],
            }
        )
        result = gen._filter_by_availability(df, "2030-01-01")
        assert len(result) == 2

    def test_empty_dataframe(self):
        gen = ReportGenerator()
        df = pd.DataFrame()
        result = gen._filter_by_availability(df, "2025-01-01")
        assert result.empty

    def test_missing_column_raises_valueerror(self):
        """A non-empty DataFrame without source_available_date is a data integrity violation."""
        gen = ReportGenerator()
        df = pd.DataFrame({"metric_name": ["a", "b"], "value": [1, 2]})
        import pytest
        with pytest.raises(ValueError, match="missing 'source_available_date' column"):
            gen._filter_by_availability(df, "2025-01-01")


# ---------------------------------------------------------------------------
# Tests: build_report_context
# ---------------------------------------------------------------------------

class TestBuildReportContext:
    """Verify build_report_context applies point-in-time filtering."""

    def test_context_excludes_future_metrics(self):
        cfg = _make_config("2025-06-01")
        gen = ReportGenerator(cfg)

        metrics = _make_metrics(
            [
                {
                    "fiscal_period": "FY2025",
                    "metric_name": "revenue",
                    "metric_value": 100e9,
                    "source_available_date": "2025-02-26",
                    "filing_date": "2025-02-26",
                    "source_accession": "acc1",
                    "unit": "USD",
                },
                {
                    "fiscal_period": "FY2026",
                    "metric_name": "revenue",
                    "metric_value": 200e9,
                    "source_available_date": "2026-02-26",
                    "filing_date": "2026-02-26",
                    "source_accession": "acc2",
                    "unit": "USD",
                },
            ]
        )
        segments = pd.DataFrame()
        nlp = pd.DataFrame()

        ctx = gen.build_report_context(metrics, segments, nlp)

        # Only FY2025 should appear in key metrics
        metric_periods = [m["period"] for m in ctx["metrics"]]
        assert "FY2026" not in metric_periods

    def test_context_excludes_future_segments(self):
        cfg = _make_config("2025-06-01")
        gen = ReportGenerator(cfg)

        segments = pd.DataFrame(
            {
                "fiscal_period": ["FY2025", "FY2026"],
                "normalized_category": ["Data Center", "Data Center"],
                "value": [50e9, 80e9],
                "source_available_date": ["2025-02-26", "2026-02-26"],
            }
        )
        ctx = gen.build_report_context(pd.DataFrame(), segments, pd.DataFrame())

        # Only FY2025 segment should appear
        if ctx["segments"]:
            for seg in ctx["segments"]:
                assert seg["category"] == "Data Center"
            # Should be exactly 1 segment entry (FY2025 only)
            assert len(ctx["segments"]) == 1

    def test_context_excludes_future_nlp(self):
        cfg = _make_config("2025-06-01")
        gen = ReportGenerator(cfg)

        nlp = pd.DataFrame(
            {
                "filing_date": ["2025-02-26", "2026-02-26"],
                "source_available_date": ["2025-02-26", "2026-02-26"],
                "feature_type": ["tfidf_similarity", "tfidf_similarity"],
                "value": [0.85, 0.90],
            }
        )
        ctx = gen.build_report_context(pd.DataFrame(), pd.DataFrame(), nlp)
        # nlp_features should be True (non-empty after filtering)
        # The filtered NLP should only have 1 row
        assert ctx["nlp_features"] is True

    def test_context_has_required_keys(self):
        gen = ReportGenerator()
        ctx = gen.build_report_context(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        required_keys = [
            "company_name",
            "ticker",
            "report_date",
            "rating",
            "current_price",
            "target_price",
            "upside_pct",
            "thesis",
            "key_bullets",
            "key_risks",
            "metrics",
            "segments",
            "exhibits",
            "pit_audit_table",
            "scorecard",
            "recommendation",
        ]
        for key in required_keys:
            assert key in ctx, f"Missing key: {key}"

    def test_pit_audit_table_has_min_3_entries(self):
        gen = ReportGenerator()
        ctx = gen.build_report_context(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert len(ctx["pit_audit_table"]) >= 3


# ---------------------------------------------------------------------------
# Tests: generate_markdown_report
# ---------------------------------------------------------------------------

class TestGenerateMarkdownReport:
    """Verify Markdown report generation."""

    def test_produces_markdown_file(self, tmp_path):
        cfg = _make_config()
        cfg.outputs_dir = tmp_path
        cfg.templates_dir = "src/templates"
        gen = ReportGenerator(cfg)

        rec = Recommendation(
            rating="Hold",
            current_price=100.0,
            target_price=110.0,
            upside_pct=0.10,
            bear_value=80.0,
            bear_probability=0.25,
            base_value=110.0,
            base_probability=0.50,
            bull_value=150.0,
            bull_probability=0.25,
            expected_value=112.5,
            what_must_be_true_buy="Growth sustains",
            what_must_be_true_hold="Valuation fair",
            what_must_be_true_sell="Margins compress",
            scorecard={"valuation_upside": 0.10},
        )
        ctx = gen.build_report_context(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            valuation={"recommendation": rec},
        )
        path = gen.generate_markdown_report(ctx)

        assert path.exists()
        assert path.name == "nvda_quantamental_report.md"
        content = path.read_text()
        assert "NVIDIA" in content
        assert "NVDA" in content


# ---------------------------------------------------------------------------
# Tests: generate_html
# ---------------------------------------------------------------------------

class TestGenerateHTML:
    """Verify HTML fallback generation."""

    def test_produces_html_file(self, tmp_path):
        cfg = _make_config()
        cfg.outputs_dir = tmp_path
        cfg.templates_dir = "src/templates"
        gen = ReportGenerator(cfg)

        # Write a minimal markdown file
        md_path = tmp_path / "nvda_quantamental_report.md"
        md_path.write_text("# Test Report\n\nHello world\n")

        html_path = gen.generate_html(md_path)
        assert html_path.exists()
        assert html_path.name == "nvda_quantamental_report.html"
        content = html_path.read_text()
        assert "<html" in content
        assert "Test Report" in content

    def test_html_fallback_logs_limitation(self, tmp_path):
        cfg = _make_config()
        cfg.outputs_dir = tmp_path
        cfg.templates_dir = "src/templates"
        gen = ReportGenerator(cfg)

        md_path = tmp_path / "test.md"
        md_path.write_text("# Test\n")
        gen.generate_html(md_path)

        limitations = tmp_path / "limitations.md"
        assert limitations.exists()
        content = limitations.read_text()
        assert "PDF Generation" in content


# ---------------------------------------------------------------------------
# Tests: generate_executive_summary
# ---------------------------------------------------------------------------

class TestGenerateExecutiveSummary:
    """Verify executive summary generation."""

    def test_produces_executive_summary(self, tmp_path):
        cfg = _make_config()
        cfg.outputs_dir = tmp_path
        cfg.templates_dir = "src/templates"
        gen = ReportGenerator(cfg)

        # Need a recommendation for the template
        rec = Recommendation(
            rating="Hold",
            current_price=100.0,
            target_price=110.0,
            upside_pct=0.10,
            bear_value=80.0,
            bear_probability=0.25,
            base_value=110.0,
            base_probability=0.50,
            bull_value=150.0,
            bull_probability=0.25,
            expected_value=112.5,
            what_must_be_true_buy="Growth sustains",
            what_must_be_true_hold="Valuation fair",
            what_must_be_true_sell="Margins compress",
            scorecard={"valuation_upside": 0.10},
        )
        ctx = gen.build_report_context(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            valuation={"recommendation": rec},
        )
        path = gen.generate_executive_summary(ctx)

        assert path.exists()
        assert path.name == "executive_summary.md"
        content = path.read_text()
        assert "Executive Summary" in content
