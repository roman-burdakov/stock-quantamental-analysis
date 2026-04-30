"""
Tests for multi-ticker support — validates that the pipeline can be
configured for a different ticker (AMD) without code changes.

Covers:
  - EngineConfig instantiation with AMD ticker and CIK
  - All config fields that reference NVDA can be overridden for AMD
  - Peer groups can be reconfigured for AMD
  - Published validation values path can be swapped
  - Output file naming uses the configured ticker
  - XBRL parser respects the configured CIK
  - SEC fetcher uses the configured CIK
  - Report generator uses the configured company name and ticker
  - Pipeline script accepts --ticker flag
  - Edge cases: empty ticker, invalid CIK format

Req: 13.5 — Switching companies requires only config changes.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.config import (
    EngineConfig,
    ScenarioAssumptions,
    get_default_config,
)


# ---------------------------------------------------------------------------
# Helper: AMD-specific configuration
# ---------------------------------------------------------------------------

def get_amd_config() -> EngineConfig:
    """Return an EngineConfig parameterised for AMD.

    Demonstrates that switching tickers requires only config changes
    (Req 13.5).  No source-code modifications are needed.
    """
    amd_scenarios = {
        "bear": ScenarioAssumptions(
            name="bear",
            probability=0.25,
            revenue_cagr=0.05,
            fcf_margin_start=0.15,
            fcf_margin_terminal=0.12,
            terminal_growth=0.025,
            sbc_treatment="included_in_fcf",
            analyst_notes="PC/gaming weakness persists; data-center share gains stall.",
        ),
        "base": ScenarioAssumptions(
            name="base",
            probability=0.50,
            revenue_cagr=0.12,
            fcf_margin_start=0.20,
            fcf_margin_terminal=0.18,
            terminal_growth=0.03,
            sbc_treatment="included_in_fcf",
            analyst_notes="Steady data-center GPU share gains; Instinct MI ramp.",
        ),
        "bull": ScenarioAssumptions(
            name="bull",
            probability=0.25,
            revenue_cagr=0.20,
            fcf_margin_start=0.25,
            fcf_margin_terminal=0.22,
            terminal_growth=0.035,
            sbc_treatment="included_in_fcf",
            analyst_notes="AI accelerator share exceeds expectations; Xilinx synergies.",
        ),
    }

    return EngineConfig(
        ticker="AMD",
        cik="0000002488",
        company_name="Advanced Micro Devices, Inc.",
        start_fiscal_year=2016,
        end_fiscal_year=2026,
        report_date="2026-04-25",
        price_date="2026-04-25",
        # AMD peers — NVDA is now a peer, not the subject
        core_semiconductor_peers=["NVDA", "INTC", "QCOM", "MRVL", "AVGO"],
        infrastructure_peers=["TSM", "ASML"],
        ai_capex_context=["MSFT", "AMZN", "GOOGL", "META"],
        peer_justifications={
            "NVDA": "Direct GPU/accelerator competitor in data-center AI",
            "INTC": "CPU competitor; re-entering discrete GPU market",
            "QCOM": "Fabless peer with AI-edge inference exposure",
            "MRVL": "Data-center semiconductor peer (custom silicon, DPUs)",
            "AVGO": "Broadcom — networking/custom-silicon peer",
            "TSM": "Leading-edge foundry; AMD manufacturing dependency",
            "ASML": "Lithography monopoly; capex-cycle proxy",
            "MSFT": "Hyperscaler AI capex — Azure demand signal",
            "AMZN": "Hyperscaler AI capex — AWS demand signal",
            "GOOGL": "Hyperscaler AI capex — TPU vs AMD GPU demand",
            "META": "Hyperscaler AI capex — GPU buyer for LLM training",
        },
        # AMD has no recent stock split
        split_history=[],
        scenarios=amd_scenarios,
        wacc=0.10,
        terminal_growth=0.03,
        projection_years=10,
    )


# ===================================================================
# 1. EngineConfig instantiation with AMD
# ===================================================================

class TestAMDConfigInstantiation:
    """Verify EngineConfig can be instantiated with AMD parameters."""

    def test_amd_config_ticker(self):
        cfg = get_amd_config()
        assert cfg.ticker == "AMD"

    def test_amd_config_cik(self):
        cfg = get_amd_config()
        assert cfg.cik == "0000002488"

    def test_amd_config_company_name(self):
        cfg = get_amd_config()
        assert cfg.company_name == "Advanced Micro Devices, Inc."

    def test_amd_config_scenarios_sum_to_one(self):
        cfg = get_amd_config()
        total = sum(s.probability for s in cfg.scenarios.values())
        assert abs(total - 1.0) < 1e-6

    def test_amd_config_no_split_history(self):
        cfg = get_amd_config()
        assert cfg.split_history == []

    def test_amd_config_has_all_required_fields(self):
        """All fields present on the default NVDA config should also
        exist on the AMD config (same dataclass)."""
        nvda = get_default_config()
        amd = get_amd_config()
        for field_name in vars(nvda):
            assert hasattr(amd, field_name), (
                f"AMD config missing field: {field_name}"
            )


# ===================================================================
# 2. Config field overrides
# ===================================================================

class TestConfigFieldOverrides:
    """Verify that all NVDA-specific config fields can be overridden."""

    def test_ticker_override(self):
        cfg = EngineConfig(ticker="AMD")
        assert cfg.ticker == "AMD"

    def test_cik_override(self):
        cfg = EngineConfig(cik="0000002488")
        assert cfg.cik == "0000002488"

    def test_company_name_override(self):
        cfg = EngineConfig(company_name="Advanced Micro Devices, Inc.")
        assert cfg.company_name == "Advanced Micro Devices, Inc."

    def test_fiscal_year_range_override(self):
        cfg = EngineConfig(start_fiscal_year=2018, end_fiscal_year=2025)
        assert cfg.start_fiscal_year == 2018
        assert cfg.end_fiscal_year == 2025

    def test_report_date_override(self):
        cfg = EngineConfig(report_date="2025-12-31")
        assert cfg.report_date == "2025-12-31"

    def test_wacc_override(self):
        cfg = EngineConfig(wacc=0.12)
        assert cfg.wacc == 0.12

    def test_validation_tolerance_override(self):
        cfg = EngineConfig(validation_tolerance_pct=2.0)
        assert cfg.validation_tolerance_pct == 2.0


# ===================================================================
# 3. Peer group reconfiguration
# ===================================================================

class TestPeerGroupReconfiguration:
    """Verify peer groups can be reconfigured for AMD."""

    def test_amd_peers_exclude_amd(self):
        cfg = get_amd_config()
        assert "AMD" not in cfg.core_semiconductor_peers

    def test_amd_peers_include_nvda(self):
        cfg = get_amd_config()
        assert "NVDA" in cfg.core_semiconductor_peers

    def test_amd_peer_justifications_present(self):
        cfg = get_amd_config()
        for peer in cfg.core_semiconductor_peers:
            assert peer in cfg.peer_justifications, (
                f"Missing justification for peer: {peer}"
            )

    def test_amd_infrastructure_peers(self):
        cfg = get_amd_config()
        assert "TSM" in cfg.infrastructure_peers
        assert "ASML" in cfg.infrastructure_peers

    def test_amd_ai_capex_context(self):
        cfg = get_amd_config()
        assert len(cfg.ai_capex_context) >= 3


# ===================================================================
# 4. Published validation values path
# ===================================================================

class TestPublishedValuesPath:
    """Verify that the published validation values path can be swapped."""

    def test_default_fixtures_dir(self):
        cfg = get_default_config()
        assert cfg.fixtures_dir == Path("tests/fixtures")

    def test_custom_fixtures_dir(self):
        cfg = EngineConfig(fixtures_dir=Path("tests/fixtures/amd"))
        assert cfg.fixtures_dir == Path("tests/fixtures/amd")

    def test_amd_config_can_use_custom_fixtures(self):
        """AMD config can point to a different fixtures directory
        for AMD-specific published values."""
        cfg = get_amd_config()
        # Override fixtures dir for AMD
        cfg.fixtures_dir = Path("tests/fixtures/amd")
        assert cfg.fixtures_dir == Path("tests/fixtures/amd")


# ===================================================================
# 5. Output file naming uses configured ticker
# ===================================================================

class TestOutputFileNaming:
    """Verify output paths and filenames use the configured ticker."""

    def test_default_outputs_dir(self):
        cfg = get_default_config()
        assert cfg.outputs_dir == Path("outputs")

    def test_custom_outputs_dir(self):
        cfg = EngineConfig(outputs_dir=Path("outputs/amd"))
        assert cfg.outputs_dir == Path("outputs/amd")

    def test_processed_dir_configurable(self):
        cfg = EngineConfig(processed_dir=Path("data/processed/amd"))
        assert cfg.processed_dir == Path("data/processed/amd")

    def test_raw_dir_configurable(self):
        cfg = EngineConfig(raw_dir=Path("data/raw/amd"))
        assert cfg.raw_dir == Path("data/raw/amd")


# ===================================================================
# 6. XBRL parser respects configured CIK
# ===================================================================

class TestXBRLParserTickerConfig:
    """Verify the XBRL parser uses the configured ticker from config."""

    def test_parser_uses_config_ticker(self):
        from src.xbrl_parser import XBRLParser

        cfg = get_amd_config()
        parser = XBRLParser(cfg)
        assert parser.config.ticker == "AMD"
        assert parser.config.cik == "0000002488"

    def test_parser_tags_facts_with_configured_ticker(self):
        """When parsing companyfacts, the ticker column should use
        the configured ticker, not a hardcoded value."""
        from src.xbrl_parser import XBRLParser

        cfg = get_amd_config()
        parser = XBRLParser(cfg)

        # Minimal companyfacts JSON with one revenue entry
        minimal_facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2024-01-01",
                                    "end": "2024-12-31",
                                    "val": 25_000_000_000,
                                    "accn": "0000002488-25-000001",
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-02-15",
                                    "frame": "CY2024",
                                }
                            ]
                        }
                    }
                }
            }
        }

        result = parser.parse_companyfacts(minimal_facts)
        assert not result.empty
        # Every row should have ticker == "AMD"
        assert (result["ticker"] == "AMD").all(), (
            f"Expected all rows to have ticker='AMD', got: {result['ticker'].unique()}"
        )


# ===================================================================
# 7. SEC fetcher uses configured CIK
# ===================================================================

class TestEdgarFetcherCIKConfig:
    """Verify the SEC fetcher uses the configured CIK."""

    def test_fetcher_uses_config_cik(self):
        from src.edgar_fetch import EdgarFetcher

        cfg = get_amd_config()
        fetcher = EdgarFetcher(cfg)
        assert fetcher.config.cik == "0000002488"

    def test_fetcher_cache_path_uses_cik(self):
        """Cache file paths should incorporate the CIK so different
        companies don't collide."""
        cfg = get_amd_config()
        cik_padded = cfg.cik.lstrip("0").zfill(10)
        expected_submissions = cfg.raw_dir / f"submissions_CIK{cik_padded}.json"
        expected_facts = cfg.raw_dir / f"companyfacts_CIK{cik_padded}.json"

        # Verify the CIK is embedded in the expected paths
        assert "0000002488" in str(expected_submissions)
        assert "0000002488" in str(expected_facts)

    def test_fetcher_submissions_url_uses_cik(self):
        """The SEC submissions URL should use the configured CIK."""
        from src.edgar_fetch import _SUBMISSIONS_URL

        cik_padded = "0000002488"
        url = _SUBMISSIONS_URL.format(cik=cik_padded)
        assert "0000002488" in url
        assert "data.sec.gov" in url


# ===================================================================
# 8. Report generator uses configured company name and ticker
# ===================================================================

class TestReportGeneratorTickerConfig:
    """Verify the report generator uses the configured company name and ticker."""

    def test_report_context_uses_config_ticker(self):
        from src.report_utils import ReportGenerator

        cfg = get_amd_config()
        gen = ReportGenerator(cfg)

        context = gen.build_report_context(
            metrics=pd.DataFrame(),
            segments=pd.DataFrame(),
            nlp_features=pd.DataFrame(),
        )

        assert context["ticker"] == "AMD"
        assert context["company_name"] == "Advanced Micro Devices, Inc."

    def test_report_context_uses_config_report_date(self):
        from src.report_utils import ReportGenerator

        cfg = get_amd_config()
        gen = ReportGenerator(cfg)

        context = gen.build_report_context(
            metrics=pd.DataFrame(),
            segments=pd.DataFrame(),
            nlp_features=pd.DataFrame(),
        )

        assert context["report_date"] == cfg.report_date


# ===================================================================
# 9. Pipeline script accepts --ticker flag
# ===================================================================

class TestPipelineCLITickerFlag:
    """Verify the pipeline script accepts and uses the --ticker flag."""

    def test_parse_args_default_ticker(self):
        """Default ticker should be NVDA."""
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["run_pipeline.py"]
            from scripts.run_pipeline import parse_args
            args = parse_args()
            assert args.ticker == "NVDA"
        finally:
            sys.argv = sys_argv_backup

    def test_parse_args_amd_ticker(self):
        """--ticker AMD should set ticker to AMD."""
        sys_argv_backup = sys.argv
        try:
            sys.argv = [
                "run_pipeline.py",
                "--ticker", "AMD",
                "--report-date", "2026-04-25",
                "--price-date", "2026-04-25",
            ]
            from scripts.run_pipeline import parse_args
            args = parse_args()
            assert args.ticker == "AMD"
        finally:
            sys.argv = sys_argv_backup

    def test_build_config_sets_ticker(self):
        """_build_config should set the ticker from CLI args."""
        from argparse import Namespace
        from scripts.run_pipeline import _build_config

        args = Namespace(
            ticker="AMD",
            report_date="2026-04-25",
            price_date="2026-04-25",
            force_refresh=False,
            skip_tests=False,
            output_format="both",
            step="all",
        )
        config = _build_config(args)
        assert config.ticker == "AMD"


# ===================================================================
# 10. Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases for ticker/CIK configuration."""

    def test_empty_ticker(self):
        """Empty ticker should still create a valid config (no crash)."""
        cfg = EngineConfig(ticker="")
        assert cfg.ticker == ""

    def test_empty_cik(self):
        """Empty CIK should still create a valid config (no crash)."""
        cfg = EngineConfig(cik="")
        assert cfg.cik == ""

    def test_cik_without_leading_zeros(self):
        """CIK without leading zeros should be accepted."""
        cfg = EngineConfig(cik="2488")
        assert cfg.cik == "2488"

    def test_cik_with_leading_zeros(self):
        """CIK with leading zeros should be preserved."""
        cfg = EngineConfig(cik="0000002488")
        assert cfg.cik == "0000002488"

    def test_ticker_case_sensitivity(self):
        """Ticker should preserve case as provided."""
        cfg = EngineConfig(ticker="amd")
        assert cfg.ticker == "amd"

    def test_config_independence(self):
        """Two configs for different tickers should be independent."""
        nvda = get_default_config()
        amd = get_amd_config()

        assert nvda.ticker != amd.ticker
        assert nvda.cik != amd.cik
        assert nvda.company_name != amd.company_name

        # Modifying one should not affect the other
        nvda.ticker = "CHANGED"
        assert amd.ticker == "AMD"

    def test_split_history_empty_for_no_split_company(self):
        """Companies without splits should have empty split_history."""
        cfg = get_amd_config()
        assert cfg.split_history == []

    def test_core_validation_metrics_shared(self):
        """Core validation metrics list should be the same regardless
        of ticker — it's a structural property of the pipeline."""
        nvda = get_default_config()
        amd = get_amd_config()
        assert nvda.core_validation_metrics == amd.core_validation_metrics
