"""
Tests for src/audit_utils.py — AuditModule.

Validates source attribution, data dictionary, limitations,
self-audit, exhibit attribution validation, and report-date
filtering validation.

Reqs: 11.1–11.7
"""

from __future__ import annotations

import pytest

from src.audit_utils import AuditModule
from src.config import (
    DataQualityIssue,
    EngineConfig,
    ExhibitRecord,
    ValuationAssumption,
    get_default_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path) -> EngineConfig:
    cfg = get_default_config()
    cfg.outputs_dir = tmp_path
    return cfg


def _sample_provenance() -> list[dict]:
    return [
        {
            "step": "fetch_submissions",
            "url": "https://data.sec.gov/submissions/CIK0001045810.json",
            "metadata": {
                "accession_number": "0001045810-25-000013",
                "filing_date": "2025-02-26",
                "report_period": "FY2025",
                "form_type": "10-K",
                "sec_url": "https://www.sec.gov/Archives/edgar/data/1045810/0001045810-25-000013.htm",
            },
        },
        {
            "step": "fetch_companyfacts",
            "url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
            "metadata": {
                "accession_number": "0001045810-24-000124",
                "filing_date": "2024-11-20",
                "report_period": "Q3FY2025",
                "form_type": "10-Q",
            },
        },
        {
            "step": "fetch_market_prices",
            "url": "https://query1.finance.yahoo.com/v8/finance/chart/NVDA",
            "metadata": {"tickers": "NVDA, AMD, AVGO"},
        },
    ]


def _sample_exhibits() -> list[ExhibitRecord]:
    return [
        ExhibitRecord(
            exhibit_id="EX-01",
            title="Revenue Segment Mix",
            source_caption="Source: NVIDIA SEC filings",
            data_source="SEC EDGAR XBRL",
            date_range="FY2020–FY2025",
            file_path="outputs/figures/revenue_segment_mix.png",
        ),
        ExhibitRecord(
            exhibit_id="EX-02",
            title="Margin Trends",
            source_caption="Source: NVIDIA SEC filings",
            data_source="SEC EDGAR XBRL",
            date_range="FY2020–FY2025",
            file_path="outputs/figures/margin_trends.png",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: generate_source_attribution
# ---------------------------------------------------------------------------

class TestGenerateSourceAttribution:

    def test_produces_file(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(_sample_provenance(), _sample_exhibits())
        assert (tmp_path / "source_attribution.md").exists()
        assert "Source Attribution" in content

    def test_includes_sec_filings(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(_sample_provenance(), _sample_exhibits())
        assert "SEC Filings" in content
        assert "0001045810-25-000013" in content

    def test_includes_market_data(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(_sample_provenance(), _sample_exhibits())
        assert "Market Data" in content

    def test_includes_exhibit_sources(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(_sample_provenance(), _sample_exhibits())
        assert "EX-01" in content
        assert "EX-02" in content

    def test_includes_llm_disclaimer(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(_sample_provenance(), [])
        assert "LLM" in content
        assert "not treated as sources of truth" in content

    def test_includes_assumptions_when_provided(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        assumptions = [
            ValuationAssumption(
                assumption_name="WACC",
                value=0.10,
                source="analyst_judgment",
                notes="Based on CAPM",
            )
        ]
        content = mod.generate_source_attribution([], [], assumptions=assumptions)
        assert "WACC" in content
        assert "analyst_judgment" in content

    def test_deduplicates_accessions(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        prov = _sample_provenance()
        # Duplicate the first entry
        prov.append(prov[0].copy())
        content = mod.generate_source_attribution(prov, [])
        # Should only appear once in the table
        assert content.count("0001045810-25-000013") == 1


# ---------------------------------------------------------------------------
# Tests: Peer Source Attribution (Req 25.6)
# ---------------------------------------------------------------------------

def _sample_peer_attribution() -> list[dict]:
    """Sample peer attribution data for testing."""
    return [
        {
            "ticker": "AMD",
            "peer_tier": "semi",
            "financial_data_date": "2025-12-27",
            "market_data_date": "2026-04-30",
            "staleness_days": 119,
            "status": "excluded",
            "reason": "Stale financial data (119d > 90d threshold)",
        },
        {
            "ticker": "AVGO",
            "peer_tier": "semi",
            "financial_data_date": "2026-02-01",
            "market_data_date": "2026-04-30",
            "staleness_days": 83,
            "status": "included",
            "reason": "All data valid and within staleness threshold",
        },
        {
            "ticker": "INTC",
            "peer_tier": "semi",
            "financial_data_date": "2026-03-28",
            "market_data_date": "2026-04-30",
            "staleness_days": 28,
            "status": "excluded",
            "reason": "Negative earnings (excluded from P/E)",
        },
        {
            "ticker": "TSM",
            "peer_tier": "infrastructure",
            "financial_data_date": "2026-03-31",
            "market_data_date": "2026-04-30",
            "staleness_days": 25,
            "status": "included",
            "reason": "All data valid and within staleness threshold",
        },
    ]


class TestPeerSourceAttribution:

    def test_peer_attribution_table_present(self, tmp_path):
        """Peer Source Attribution section appears when peer data is provided."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(
            _sample_provenance(), _sample_exhibits(),
            peer_attribution=_sample_peer_attribution(),
        )
        assert "Peer Source Attribution" in content

    def test_peer_attribution_columns(self, tmp_path):
        """Table contains all required columns per Req 25.6."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(
            _sample_provenance(), _sample_exhibits(),
            peer_attribution=_sample_peer_attribution(),
        )
        assert "Ticker" in content
        assert "Financial Data Date" in content
        assert "Market Data Date" in content
        assert "Staleness (Days)" in content
        assert "Status" in content
        assert "Reason" in content

    def test_peer_attribution_ticker_data(self, tmp_path):
        """Each peer ticker appears with its attribution data."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(
            _sample_provenance(), _sample_exhibits(),
            peer_attribution=_sample_peer_attribution(),
        )
        assert "AMD" in content
        assert "AVGO" in content
        assert "INTC" in content
        assert "TSM" in content

    def test_peer_attribution_staleness(self, tmp_path):
        """Staleness days are documented for each peer."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(
            _sample_provenance(), _sample_exhibits(),
            peer_attribution=_sample_peer_attribution(),
        )
        assert "119" in content  # AMD staleness
        assert "83" in content   # AVGO staleness

    def test_peer_attribution_inclusion_exclusion(self, tmp_path):
        """Inclusion/exclusion status and reason are documented."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(
            _sample_provenance(), _sample_exhibits(),
            peer_attribution=_sample_peer_attribution(),
        )
        assert "included" in content
        assert "excluded" in content
        assert "Stale financial data" in content

    def test_peer_attribution_summary(self, tmp_path):
        """Summary counts included and excluded peers."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(
            _sample_provenance(), _sample_exhibits(),
            peer_attribution=_sample_peer_attribution(),
        )
        assert "2 peers included" in content
        assert "2 peers excluded" in content

    def test_peer_attribution_from_provenance(self, tmp_path):
        """Peer attribution is reconstructed from provenance when not provided."""
        mod = AuditModule(_make_config(tmp_path))
        provenance = _sample_provenance() + [
            {
                "step": "fetch_peer_financials",
                "source_url": "yfinance",
                "ticker": "AVGO",
                "timestamp": "2026-04-30T02:54:12.707030+00:00",
                "source_date": "2026-02-01",
                "staleness_days": 83,
            },
            {
                "step": "fetch_peer_financials",
                "source_url": "yfinance",
                "ticker": "AMD",
                "timestamp": "2026-04-30T02:54:12.707030+00:00",
                "source_date": "2025-12-27",
                "staleness_days": 119,
            },
        ]
        content = mod.generate_source_attribution(provenance, _sample_exhibits())
        assert "Peer Source Attribution" in content
        assert "AVGO" in content
        assert "AMD" in content

    def test_peer_attribution_empty_graceful(self, tmp_path):
        """Graceful fallback when no peer data is available."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution([], [])
        assert "Peer Source Attribution" in content
        assert "No per-peer attribution data available" in content

    def test_peer_attribution_peer_tier(self, tmp_path):
        """Peer tier is documented for each peer."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_source_attribution(
            _sample_provenance(), _sample_exhibits(),
            peer_attribution=_sample_peer_attribution(),
        )
        assert "semi" in content
        assert "infrastructure" in content


# ---------------------------------------------------------------------------
# Tests: generate_data_dictionary
# ---------------------------------------------------------------------------

class TestGenerateDataDictionary:

    def test_produces_file(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        schema = [
            {
                "variable": "revenue",
                "source": "SEC EDGAR XBRL",
                "unit": "USD",
                "period": "FY2025",
                "source_available_date": "2025-02-26",
                "transformation": "none",
                "limitations": "",
            }
        ]
        content = mod.generate_data_dictionary(schema)
        assert (tmp_path / "data_dictionary.md").exists()
        assert "Data Dictionary" in content

    def test_includes_all_fields(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        schema = [
            {
                "variable": "gross_margin",
                "source": "Computed from XBRL",
                "unit": "ratio",
                "period": "FY2025",
                "source_available_date": "2025-02-26",
                "transformation": "gross_profit / revenue",
                "limitations": "Depends on XBRL tag availability",
            }
        ]
        content = mod.generate_data_dictionary(schema)
        assert "gross_margin" in content
        assert "ratio" in content
        assert "gross_profit / revenue" in content

    def test_empty_schema(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_data_dictionary([])
        assert "Data Dictionary" in content
        assert (tmp_path / "data_dictionary.md").exists()


# ---------------------------------------------------------------------------
# Tests: generate_limitations
# ---------------------------------------------------------------------------

class TestGenerateLimitations:

    def test_produces_file(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_limitations([], ["WACC is assumed constant"])
        assert (tmp_path / "limitations.md").exists()
        assert "Limitations" in content

    def test_includes_data_gaps(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        gaps = [
            DataQualityIssue(
                category="missing_tag",
                detail="SBC not available for FY2016",
                filing_or_source="10-K FY2016",
                severity="warning",
            )
        ]
        content = mod.generate_limitations(gaps, [])
        assert "missing_tag" in content
        assert "SBC" in content

    def test_includes_assumptions(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_limitations([], ["WACC is constant at 10%", "Terminal growth 3%"])
        assert "WACC is constant at 10%" in content
        assert "Terminal growth 3%" in content

    def test_includes_blockers(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_limitations(
            [], [], blockers=["PDF generation requires weasyprint"]
        )
        assert "weasyprint" in content

    def test_includes_model_limitations(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_limitations([], [])
        assert "Model Limitations" in content


# ---------------------------------------------------------------------------
# Tests: perform_self_audit
# ---------------------------------------------------------------------------

class TestPerformSelfAudit:

    def _full_report(self) -> str:
        return (
            "# NVDA Quantamental Report\n"
            "## Executive Summary\n"
            "NVIDIA recommendation: Hold.\n"
            "## Valuation\n"
            "DCF analysis with sensitivity table.\n"
            "## Risk Factors\n"
            "Export controls, customer concentration.\n"
            "## Recommendation\n"
            "Hold rating based on scorecard.\n"
            "## Point-in-Time Controls\n"
            "No lookahead bias. Staleness checks applied.\n"
            "## Narrative Drift\n"
            "TF-IDF analysis shows shifts.\n"
            "## Reverse-DCF\n"
            "Implied expectations grid.\n"
            "See source_attribution.md and prompt_log.md for LLM disclosure.\n"
            "Figure 1, Figure 2, Figure 3, Figure 4, Figure 5, Figure 6.\n"
        )

    def _full_attribution(self) -> str:
        return (
            "# Source Attribution\n"
            "## SEC Filings\n"
            "| Filing Date | Accession |\n"
            "| 2025-02-26 | 0001045810-25-000013 |\n"
        )

    def test_appends_to_model_audit(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        # Create existing model_audit.md
        (tmp_path / "model_audit.md").write_text("# Model Audit\n\nExisting content.\n")
        mod.perform_self_audit(self._full_report(), self._full_attribution())
        content = (tmp_path / "model_audit.md").read_text()
        assert "Existing content" in content
        assert "Self-Audit" in content

    def test_creates_model_audit_if_missing(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        mod.perform_self_audit(self._full_report(), self._full_attribution())
        assert (tmp_path / "model_audit.md").exists()

    def test_grades_are_conservative(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.perform_self_audit(self._full_report(), self._full_attribution())
        # Should never give A+ (skeptical)
        assert "A+" not in content

    def test_four_criteria_present(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.perform_self_audit(self._full_report(), self._full_attribution())
        assert "Completeness" in content
        assert "Novelty" in content
        assert "Readability" in content
        assert "Source Attribution" in content

    def test_overall_grade_present(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        content = mod.perform_self_audit(self._full_report(), self._full_attribution())
        assert "Overall Assessment" in content
        assert "Weighted Grade" in content

    def test_flags_missing_sections(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        # Minimal report missing many sections
        content = mod.perform_self_audit("Short report.", "No attribution.")
        assert "⚠️" in content


# ---------------------------------------------------------------------------
# Tests: validate_exhibit_attribution
# ---------------------------------------------------------------------------

class TestValidateExhibitAttribution:

    def test_no_violations_when_all_matched(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        report = "See EX-01 and EX-02 for details."
        attribution = "EX-01: Revenue chart. EX-02: Margin chart."
        violations = mod.validate_exhibit_attribution(report, attribution)
        assert violations == []

    def test_detects_missing_exhibit(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        report = "See EX-01 and EX-03 for details."
        attribution = "EX-01: Revenue chart."
        violations = mod.validate_exhibit_attribution(report, attribution)
        assert len(violations) == 1
        assert "EX-03" in violations[0]

    def test_detects_missing_figure_file(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        report = "![Chart](figures/revenue_segment_mix.png)"
        attribution = "No figures listed."
        violations = mod.validate_exhibit_attribution(report, attribution)
        assert len(violations) >= 1

    def test_empty_report_no_violations(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        violations = mod.validate_exhibit_attribution("", "")
        assert violations == []


# ---------------------------------------------------------------------------
# Tests: validate_report_date_filtering
# ---------------------------------------------------------------------------

class TestValidateReportDateFiltering:

    def test_no_violations_clean_context(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        context = {
            "metrics": [
                {"name": "revenue", "source_available_date": "2025-02-26"},
            ],
        }
        violations = mod.validate_report_date_filtering(context, "2026-04-25")
        assert violations == []

    def test_detects_future_data_in_list(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        context = {
            "metrics": [
                {"name": "revenue", "source_available_date": "2027-01-01"},
            ],
        }
        violations = mod.validate_report_date_filtering(context, "2026-04-25")
        assert len(violations) == 1
        assert "2027-01-01" in violations[0]

    def test_detects_future_data_in_dataframe(self, tmp_path):
        import pandas as pd
        mod = AuditModule(_make_config(tmp_path))
        df = pd.DataFrame({
            "source_available_date": ["2025-01-01", "2027-06-01"],
            "value": [100, 200],
        })
        context = {"metrics_df": df}
        violations = mod.validate_report_date_filtering(context, "2026-04-25")
        assert len(violations) == 1

    def test_empty_context_no_violations(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        violations = mod.validate_report_date_filtering({}, "2026-04-25")
        assert violations == []

    def test_detects_ml_lookahead(self, tmp_path):
        import pandas as pd
        mod = AuditModule(_make_config(tmp_path))
        matrix = pd.DataFrame({
            "feature_available_date": ["2025-01-01", "2027-01-01"],
            "target": [0.1, 0.2],
        })
        context = {"ml_results": {"feature_target_matrix": matrix}}
        violations = mod.validate_report_date_filtering(context, "2026-04-25")
        assert len(violations) == 1
        assert "feature_available_date" in violations[0]


# ---------------------------------------------------------------------------
# Tests: generate_prompt_log (Req 23.1–23.7)
# ---------------------------------------------------------------------------

# Shared prompt log entries matching the production entries from
# scripts/regenerate_outputs.py.  These are used by all prompt-log tests
# so that we exercise generate_prompt_log() directly rather than reading
# the on-disk output file.

_PROMPT_LOG_ENTRIES: list[dict] = [
    {
        "type": "architecture_design",
        "prompt": (
            "Design a quantamental analysis pipeline for NVIDIA using 10 years of SEC filings, "
            "market data, and peer financials."
        ),
        "model": "Claude Opus 4.6 (Anthropic) via Kiro",
        "purpose": "Define end-to-end pipeline architecture and module responsibilities",
        "output_used": "Pipeline architecture (ingest → parse → validate → metrics → segments → text → NLP → ML → valuation → charts → report → audit)",
        "verification": "Architecture reviewed against MIT 15.C51 assignment requirements and grading rubric",
        "date": "2026-04-30",
        "affected_sections": ["Scalability and Automation"],
        "material_claim": False,
    },
    {
        "type": "code_generation",
        "prompt": (
            "Implement src/xbrl_parser.py with fiscal-year selection algorithm. "
            "Parse NVIDIA companyfacts JSON and validate against published 10-K values."
        ),
        "model": "Claude Opus 4.6 (Anthropic) via Kiro",
        "purpose": "Fix XBRL parsing to correctly distinguish annual vs quarterly/YTD values",
        "output_used": "Multi-concept fallback logic with period-duration classification",
        "verification": (
            "Verified against 10-K filing accession 0001045810-25-000023 (FY2025): "
            "revenue = $130,497M ±1%"
        ),
        "date": "2026-04-30",
        "affected_sections": ["Fundamental Analysis — Last 10 Years of Public Filings"],
        "material_claim": True,
        "primary_source": "NVIDIA 10-K FY2025 (accession 0001045810-25-000023)",
    },
    {
        "type": "code_generation",
        "prompt": (
            "Implement src/data_validation.py with DataValidationGate class. "
            "Cross-check every core metric against published 10-K values."
        ),
        "model": "Claude Opus 4.6 (Anthropic) via Kiro",
        "purpose": "Implement hard data validation gate to prevent garbage-in-garbage-out",
        "output_used": "DataValidationGate with ValidatedMetric schema, tolerance checks, and blocking logic",
        "verification": (
            "Validated 12 core metrics against published 10-K values for FY2023 "
            "(accession 0001045810-23-000017), FY2024 (accession 0001045810-24-000029), "
            "FY2025 (accession 0001045810-25-000023)"
        ),
        "date": "2026-04-30",
        "affected_sections": ["Source Attribution and Audit", "Cover Page"],
        "material_claim": True,
        "primary_source": "tests/fixtures/nvda_published_values.json sourced from NVIDIA 10-K filings",
    },
    {
        "type": "report_drafting",
        "prompt": (
            "Generate the full quantamental research report for NVIDIA. Sections: Cover Page, "
            "Executive Summary, Company Overview, Historical Financial Analysis, Driver Analysis, "
            "ML/Quantamental Insights, Valuation, Risks, Catalysts, Scalability and Automation, "
            "Point-in-Time Controls."
        ),
        "model": "Claude Opus 4.6 (Anthropic) via Kiro",
        "purpose": "Draft complete equity research report with all required sections",
        "output_used": "Full report Markdown with Jinja2 templates; executive summary",
        "verification": (
            "Financial data in report verified against 10-K filings: FY2025 (accession "
            "0001045810-25-000023), FY2024 (accession 0001045810-24-000029)"
        ),
        "date": "2026-04-30",
        "affected_sections": [
            "Executive Summary",
            "Valuation and Recommendation",
            "Risks, Catalysts, and What Would Change the Rating",
        ],
        "material_claim": True,
        "primary_source": "SEC filings (accessions 0001045810-23-000017, 0001045810-24-000029, 0001045810-25-000023)",
    },
    {
        "type": "debugging",
        "prompt": (
            "Fix XBRL concept map for capex and total_debt. Build canonical validation truth "
            "table. Manually validate capex/debt from 10-K filings."
        ),
        "model": "Claude Opus 4.6 (Anthropic) via Kiro",
        "purpose": "A+ final repair pass — fix XBRL concept mapping, add canonical validation",
        "output_used": "Fixed XBRL concept map; manual validation values for capex and total_debt",
        "verification": (
            "Capex and total_debt manually validated against NVIDIA 10-K filings: "
            "FY2023 (accession 0001045810-23-000017), FY2024 (accession 0001045810-24-000029), "
            "FY2025 (accession 0001045810-25-000023)"
        ),
        "date": "2026-04-30",
        "affected_sections": ["Fundamental Analysis — Last 10 Years of Public Filings", "Valuation and Recommendation"],
        "material_claim": True,
        "primary_source": "NVIDIA 10-K filings (accessions 0001045810-23-000017, 0001045810-24-000029, 0001045810-25-000023)",
    },
    {
        "type": "verification",
        "prompt": (
            "Run full acceptance test suite. Verify all required output files exist, "
            "audit_status.json shows pass, no contradictions between output files."
        ),
        "model": "Claude Opus 4.6 (Anthropic) via Kiro",
        "purpose": "Final verification of all pipeline outputs and cross-file consistency",
        "output_used": "Acceptance test results confirming all checks pass",
        "verification": "Automated acceptance_tests.py script; audit consistency checker",
        "date": "2026-04-30",
        "affected_sections": [],
        "material_claim": False,
    },
    {
        "type": "code_generation",
        "prompt": (
            "Implement the final repair/generation pass for the NVDA quantamental engine. "
            "This is the post-mortem XBRL fix that addresses the catastrophic failure where "
            "FY2025 revenue was parsed as $26.974B instead of $130.497B. The fix includes: "
            "(1) Period-duration classification: annual (350-380d), quarterly (<100d), YTD (100-340d). "
            "(2) Annual fact selection: prefer 10-K annual-duration → frame-tagged → latest amendment. "
            "(3) Hard rejection of quarterly/YTD facts for annual metric selection. "
            "(4) Published-value validation gate with 1% tolerance on 12 core metrics. "
            "(5) DATA_BLOCKED status when any critical metric fails, preventing formal rating. "
            "(6) Regression test: FY2025 revenue must parse as ~$130.5B."
        ),
        "model": "Claude Opus 4.6 (Anthropic) via Kiro",
        "purpose": "Final repair/generation prompt — post-mortem XBRL fix preventing catastrophic revenue misparse",
        "output_used": "Complete pipeline rewrite: xbrl_parser.py, data_validation.py, config.py, run_pipeline.py, all test files",
        "verification": (
            "Regression test confirms FY2025 revenue = $130,497M ±1% against 10-K filing "
            "accession 0001045810-25-000023"
        ),
        "date": "2026-04-30",
        "affected_sections": [
            "Cover Page",
            "Executive Summary",
            "Fundamental Analysis — Last 10 Years of Public Filings",
            "Valuation and Recommendation",
            "Source Attribution and Audit",
        ],
        "material_claim": True,
        "primary_source": "NVIDIA 10-K FY2025 (accession 0001045810-25-000023): total revenue $130,497M",
    },
]

_COVERAGE_GAPS: list[str] = [
    "Early development prompts were not logged with the same granularity",
    "Some intermediate debugging iterations are summarized rather than logged verbatim",
]


class TestGeneratePromptLog:
    """Acceptance tests for generate_prompt_log() — Reqs 23.1–23.7."""

    # -- helpers --

    def _generate(self, tmp_path, **overrides):
        """Call generate_prompt_log with production-like entries."""
        mod = AuditModule(_make_config(tmp_path))
        kwargs = dict(
            entries=_PROMPT_LOG_ENTRIES,
            total_interactions=18,
            fraction_fully_logged=0.67,
            coverage_gaps=_COVERAGE_GAPS,
        )
        kwargs.update(overrides)
        return mod.generate_prompt_log(**kwargs)

    # -- Req 23.1: at least 5 structured entries --

    def test_at_least_five_structured_entries(self, tmp_path):
        """prompt_log.md contains at least 5 structured entries (### Entry headers)."""
        content = self._generate(tmp_path)
        import re
        entry_headers = re.findall(r"^### Entry \d+", content, re.MULTILINE)
        assert len(entry_headers) >= 5, (
            f"Expected ≥5 structured entries, found {len(entry_headers)}"
        )

    # -- Req 23.1: each entry has model name, purpose, and verification --

    def test_each_entry_has_model_purpose_verification(self, tmp_path):
        """Every entry has Model, Purpose, and Verification Method fields."""
        content = self._generate(tmp_path)
        import re
        # Split into individual entries
        entry_blocks = re.split(r"(?=^### Entry \d+)", content, flags=re.MULTILINE)
        entry_blocks = [b for b in entry_blocks if b.strip().startswith("### Entry")]
        assert len(entry_blocks) >= 5

        for block in entry_blocks:
            assert "**Model:**" in block, f"Missing Model in:\n{block[:200]}"
            assert "**Purpose:**" in block, f"Missing Purpose in:\n{block[:200]}"
            assert "**Verification Method:**" in block, (
                f"Missing Verification Method in:\n{block[:200]}"
            )

    # -- Req 23.2: each entry has a valid type classification --

    def test_each_entry_has_valid_type(self, tmp_path):
        """Every entry has a Type field with a valid classification."""
        content = self._generate(tmp_path)
        import re
        valid_types = {
            "code_generation", "report_drafting", "data_analysis",
            "debugging", "architecture_design", "verification",
        }
        type_matches = re.findall(r"\*\*Type:\*\*\s*`(\w+)`", content)
        assert len(type_matches) >= 5, (
            f"Expected ≥5 type annotations, found {len(type_matches)}"
        )
        for t in type_matches:
            assert t in valid_types, f"Invalid type '{t}'; expected one of {valid_types}"

    # -- Req 23.3: final repair prompt is included --

    def test_final_repair_prompt_included(self, tmp_path):
        """The final repair/generation prompt (post-mortem XBRL fix) is present."""
        content = self._generate(tmp_path)
        # The final repair entry should mention the post-mortem XBRL fix
        assert "post-mortem XBRL fix" in content.lower() or "xbrl fix" in content.lower(), (
            "Final repair prompt not found in prompt log"
        )
        # Should mention the catastrophic failure
        assert "$26.974B" in content or "26.974" in content or "$130.497B" in content or "130.497" in content, (
            "Final repair prompt should reference the revenue misparse values"
        )

    # -- Req 23.5, 23.6: incompleteness is disclosed --

    def test_incompleteness_disclosed(self, tmp_path):
        """Prompt log discloses total interactions, fraction logged, and coverage gaps."""
        content = self._generate(tmp_path)
        # Total interactions
        assert "18" in content, "Total interactions count (18) not found"
        # Fraction logged
        assert "67%" in content, "Fraction fully logged (67%) not found"
        # Coverage gaps section
        assert "Coverage" in content, "Coverage gaps section not found"
        assert "early development" in content.lower() or "not logged" in content.lower(), (
            "Coverage gap disclosure about early development prompts not found"
        )

    # -- Req 23.4, 23.7: LLM-Drafted Section Disclosure table --

    def test_llm_drafted_section_disclosure_table(self, tmp_path):
        """LLM-Drafted Section Disclosure table exists with verification info."""
        content = self._generate(tmp_path)
        assert "LLM-Drafted Section Disclosure" in content, (
            "Missing LLM-Drafted Section Disclosure section"
        )
        # Table should have headers
        assert "Report Section" in content
        assert "Verification Category" in content
        # Should reference at least one report section
        assert "Executive Summary" in content or "Valuation" in content

    # -- Req 23.7: cross-references to specific 10-K accession numbers --

    def test_cross_references_to_accession_numbers(self, tmp_path):
        """Prompt log cross-references specific 10-K accession numbers."""
        content = self._generate(tmp_path)
        # FY2025 accession
        assert "0001045810-25-000023" in content, (
            "Missing FY2025 10-K accession cross-reference"
        )
        # At least one more accession (FY2024 or FY2023)
        has_fy2024 = "0001045810-24-000029" in content
        has_fy2023 = "0001045810-23-000017" in content
        assert has_fy2024 or has_fy2023, (
            "Missing FY2024 or FY2023 10-K accession cross-reference"
        )

    # -- Req 23.1: prompt log writes to disk --

    def test_writes_to_disk(self, tmp_path):
        """generate_prompt_log() writes outputs/prompt_log.md to disk."""
        self._generate(tmp_path)
        path = tmp_path / "prompt_log.md"
        assert path.exists(), "prompt_log.md not written to disk"
        assert path.stat().st_size > 0, "prompt_log.md is empty"

    # -- Interaction summary table --

    def test_interaction_summary_by_type(self, tmp_path):
        """Prompt log includes an interaction summary table grouped by type."""
        content = self._generate(tmp_path)
        assert "Interaction Summary by Type" in content, (
            "Missing Interaction Summary by Type section"
        )
        # Should list code_generation (most common type in entries)
        assert "code_generation" in content

    # -- Verification statement --

    def test_verification_statement_present(self, tmp_path):
        """Prompt log includes a verification statement about how claims were verified."""
        content = self._generate(tmp_path)
        assert "Verification Statement" in content, (
            "Missing Verification Statement section"
        )
        assert "SEC filing" in content.lower() or "sec filing" in content.lower()
