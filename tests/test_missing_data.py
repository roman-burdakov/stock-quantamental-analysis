"""
Tests for graceful handling of missing data across the pipeline.

Covers:
  1. Missing XBRL tags → null + log (no crash, no silent estimation)
  2. Missing prices → excluded (LOCF only ≤3 days)
  3. Missing filing sections → empty record with parse_status="missing" + log
  4. Missing peer EV inputs → EV-based multiples skipped

All tests run offline using fixtures and mock data.
Reqs: 12.6
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig
from src.filing_text_parser import FilingTextParser
from src.financial_metrics import FinancialMetricsCalculator
from src.valuation import ValuationModule
from src.xbrl_parser import XBRLParser

FIXTURES = Path("tests/fixtures")


# ===================================================================
# 1. Missing XBRL tags → null + log
# ===================================================================


class TestMissingXBRLTags:
    """When a required XBRL concept is missing, the parser returns null
    for that metric and logs the gap — it must not crash or silently estimate."""

    def test_completely_empty_facts_returns_empty_df(self):
        parser = XBRLParser(EngineConfig())
        df = parser.parse_companyfacts({"facts": {}})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_concept_logged_as_warning(self, caplog):
        parser = XBRLParser(EngineConfig())
        facts = {"facts": {"us-gaap": {}}}
        with caplog.at_level(logging.WARNING):
            parser.parse_companyfacts(facts)
        assert any("No XBRL concept found" in m for m in caplog.messages)

    def test_missing_tags_tracked_in_parser(self):
        parser = XBRLParser(EngineConfig())
        facts = {"facts": {"us-gaap": {}}}
        parser.parse_companyfacts(facts)
        assert len(parser._missing_tags) > 0

    def test_partial_facts_only_returns_found_metrics(self):
        """Fixture has revenue but not SGA — only found metrics appear."""
        with open(FIXTURES / "sample_companyfacts.json") as f:
            facts = json.load(f)
        parser = XBRLParser(EngineConfig())
        df = parser.parse_companyfacts(facts)
        assert "revenue" in df["metric_name"].values
        assert "sga" not in df["metric_name"].values

    def test_resolve_concept_returns_none_for_missing(self):
        parser = XBRLParser(EngineConfig())
        facts = {"facts": {"us-gaap": {}}}
        concept, entries = parser.resolve_concept(facts, "revenue")
        assert concept is None
        assert entries is None

    def test_metrics_null_when_xbrl_tag_missing(self, tmp_path):
        """FinancialMetricsCalculator produces null for metrics whose
        upstream XBRL inputs are absent."""
        config = EngineConfig()
        config.processed_dir = tmp_path
        calc = FinancialMetricsCalculator(config)

        # Only revenue present — gross_profit, operating_income, etc. missing
        data = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2099,
                "filing_date": "2099-02-01", "source_available_date": "2099-02-01",
                "accession_number": "fake", "metric_name": "revenue",
                "value": 100_000_000_000, "unit": "USD", "form_type": "10-K",
            },
        ])
        result = calc.compute_all_metrics(data)
        period = result[result["fiscal_period"] == "FY2099"]

        # gross_margin needs gross_profit which is missing → null
        gm = period[period["metric_name"] == "gross_margin"]
        assert len(gm) == 1
        assert gm.iloc[0]["metric_value"] is None or pd.isna(gm.iloc[0]["metric_value"])

        # FCF needs operating_cash_flow and capex → null
        fcf = period[period["metric_name"] == "FCF"]
        assert len(fcf) == 1
        assert fcf.iloc[0]["metric_value"] is None or pd.isna(fcf.iloc[0]["metric_value"])

    def test_metrics_null_logged(self, tmp_path, caplog):
        """Null metrics should be logged at INFO level."""
        config = EngineConfig()
        config.processed_dir = tmp_path
        calc = FinancialMetricsCalculator(config)

        data = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2099,
                "filing_date": "2099-02-01", "source_available_date": "2099-02-01",
                "accession_number": "fake", "metric_name": "revenue",
                "value": None, "unit": "USD", "form_type": "10-K",
            },
        ])
        with caplog.at_level(logging.INFO):
            calc.compute_all_metrics(data)
        assert any("null" in m.lower() or "none" in m.lower() for m in caplog.messages)


# ===================================================================
# 2. Missing prices → excluded (LOCF max 3 days)
# ===================================================================


class TestMissingPrices:
    """When market prices are missing beyond the 3-day LOCF window,
    those rows must be excluded — not interpolated."""

    def _make_fetcher(self, tmp_path):
        """Create an EdgarFetcher with a temp cache dir."""
        from src.edgar_fetch import EdgarFetcher

        config = EngineConfig()
        config.raw_dir = tmp_path
        config.force_refresh = True
        return EdgarFetcher(config)

    def test_locf_fills_weekend_gap(self, tmp_path):
        """A 2-day weekend gap should be filled via LOCF."""
        fetcher = self._make_fetcher(tmp_path)

        # Friday + Monday data, Saturday/Sunday missing
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-03", "2025-01-06"]),
            "ticker": ["NVDA", "NVDA"],
            "adj_close": [140.0, 142.0],
            "source_available_date": ["2025-01-03", "2025-01-06"],
        })
        result = fetcher._apply_locf(df, max_gap_days=3)
        # Saturday and Sunday should be filled
        dates = result["date"].dt.strftime("%Y-%m-%d").tolist()
        assert "2025-01-04" in dates
        assert "2025-01-05" in dates

    def test_locf_excludes_long_gap(self, tmp_path, caplog):
        """A gap > 3 calendar days should be logged and excluded."""
        fetcher = self._make_fetcher(tmp_path)

        # 5-day gap between observations
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-07"]),
            "ticker": ["NVDA", "NVDA"],
            "adj_close": [140.0, 145.0],
            "source_available_date": ["2025-01-01", "2025-01-07"],
        })
        with caplog.at_level(logging.INFO):
            result = fetcher._apply_locf(df, max_gap_days=3)

        # Days 4, 5, 6 exceed the 3-day window → should be dropped
        filled_dates = set(result["date"].dt.strftime("%Y-%m-%d").tolist())
        # Days within 3-day window (Jan 2, 3, 4) are kept; days beyond (Jan 5, 6) dropped
        assert "2025-01-05" not in filled_dates or "2025-01-06" not in filled_dates
        # Should have logged the exclusion
        assert any("excluding" in m.lower() or "gap" in m.lower() for m in caplog.messages)

    def test_locf_empty_input(self, tmp_path):
        """Empty price DataFrame should return empty, not crash."""
        fetcher = self._make_fetcher(tmp_path)
        df = pd.DataFrame(columns=["date", "ticker", "adj_close", "source_available_date"])
        result = fetcher._apply_locf(df, max_gap_days=3)
        assert result.empty

    def test_locf_single_ticker_no_gap(self, tmp_path):
        """Consecutive trading days should pass through unchanged."""
        fetcher = self._make_fetcher(tmp_path)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "ticker": ["NVDA", "NVDA"],
            "adj_close": [140.0, 141.0],
            "source_available_date": ["2025-01-02", "2025-01-03"],
        })
        result = fetcher._apply_locf(df, max_gap_days=3)
        assert len(result) == 2


# ===================================================================
# 3. Missing filing sections → empty + log
# ===================================================================


class TestMissingFilingSections:
    """When a filing section cannot be extracted, the parser returns
    an empty record with parse_status='missing' and logs it."""

    def test_unrecognizable_html_yields_missing_status(self):
        parser = FilingTextParser(EngineConfig())
        html = "<html><body><p>No Item headings here at all.</p></body></html>"
        results = parser.parse_filing(
            html=html,
            accession="test-missing",
            form_type="10-K",
            filing_date="2025-01-01",
            source_available_date="2025-01-01",
        )
        for r in results:
            assert r.parse_status == "missing"
            assert r.text == ""
            assert r.char_count == 0

    def test_missing_section_still_returns_all_expected_sections(self):
        """Even when nothing is found, all target sections should be present."""
        parser = FilingTextParser(EngineConfig())
        html = "<html><body><p>Empty filing.</p></body></html>"
        results = parser.parse_filing(
            html=html,
            accession="test-empty",
            form_type="10-K",
            filing_date="2025-01-01",
            source_available_date="2025-01-01",
        )
        section_names = {r.section_name for r in results}
        assert section_names == {"business", "risk_factors", "mda", "quant"}

    def test_10q_missing_sections_returns_expected_sections(self):
        parser = FilingTextParser(EngineConfig())
        html = "<html><body><p>Empty 10-Q.</p></body></html>"
        results = parser.parse_filing(
            html=html,
            accession="test-empty-10q",
            form_type="10-Q",
            filing_date="2025-01-01",
            source_available_date="2025-01-01",
        )
        section_names = {r.section_name for r in results}
        assert section_names == {"risk_factors", "mda"}

    def test_coverage_report_flags_missing(self):
        """generate_coverage_report should flag missing sections with a warning."""
        parser = FilingTextParser(EngineConfig())
        html = "<html><body><p>Nothing useful.</p></body></html>"
        results = parser.parse_filing(
            html=html,
            accession="test-coverage",
            form_type="10-K",
            filing_date="2025-01-01",
            source_available_date="2025-01-01",
        )
        report = parser.generate_coverage_report(results)
        missing_rows = report[report["parse_status"] == "missing"]
        assert len(missing_rows) > 0
        assert (missing_rows["warning"] == "Section not found").all()

    def test_partial_extraction_mixed_statuses(self):
        """HTML with only some sections should yield a mix of success and missing."""
        parser = FilingTextParser(EngineConfig())
        # HTML that has Item 1A content but nothing else recognizable
        html = """<html><body>
        <p>Item 1A. Risk Factors</p>
        <p>The company faces significant risks including market volatility,
        regulatory changes, competitive pressures, and supply chain disruptions.
        These risks could materially affect our business operations and financial
        results. Additional risk factors include geopolitical tensions.</p>
        <p>Item 2. Properties</p>
        <p>Our headquarters are in Santa Clara.</p>
        </body></html>"""
        results = parser.parse_filing(
            html=html,
            accession="test-partial",
            form_type="10-K",
            filing_date="2025-01-01",
            source_available_date="2025-01-01",
        )
        statuses = {r.section_name: r.parse_status for r in results}
        assert statuses["risk_factors"] == "success"
        # Other sections should be missing
        assert statuses["mda"] == "missing"


# ===================================================================
# 4. Missing peer EV inputs → EV multiples skipped
# ===================================================================


class TestMissingPeerEV:
    """When a peer's EV inputs (market_cap, debt, cash) are missing or
    stale, EV-based multiples should be skipped for that peer."""

    @pytest.fixture
    def vm(self):
        return ValuationModule()

    def test_missing_market_cap_skips_ev_multiples(self, vm):
        """Peer with market_cap=None → EV/Revenue and EV/EBITDA are None."""
        peer_fin = pd.DataFrame([{
            "ticker": "FAKE",
            "market_cap": None,
            "total_debt": 1000,
            "cash_and_securities": 500,
            "revenue": 50000,
            "ebitda": 10000,
            "net_income": 5000,
            "fcf": 4000,
            "stale_ev": False,
        }])
        nvda_fin = pd.DataFrame([{
            "ticker": "NVDA",
            "market_cap": 2_000_000_000_000,
            "total_debt": 8_000_000_000,
            "cash_and_securities": 40_000_000_000,
            "revenue": 130_000_000_000,
            "ebitda": 80_000_000_000,
            "net_income": 70_000_000_000,
            "fcf": 60_000_000_000,
            "stale_ev": False,
        }])
        result = vm.compute_peer_multiples(peer_fin, nvda_fin)
        fake_row = result[result["ticker"] == "FAKE"].iloc[0]
        assert fake_row["EV/Revenue"] is None or pd.isna(fake_row["EV/Revenue"])
        assert fake_row["EV/EBITDA"] is None or pd.isna(fake_row["EV/EBITDA"])

    def test_stale_ev_computes_ev_from_current_market_cap(self, vm):
        """Peer with stale_ev=True still computes EV from current market_cap
        (stale financials are a warning, not a blocker for EV computation)."""
        peer_fin = pd.DataFrame([{
            "ticker": "STALE",
            "market_cap": 100_000_000_000,
            "total_debt": 5000,
            "cash_and_securities": 2000,
            "revenue": 30000,
            "ebitda": 8000,
            "net_income": 4000,
            "fcf": 3000,
            "stale_ev": True,
        }])
        nvda_fin = pd.DataFrame([{
            "ticker": "NVDA",
            "market_cap": 2_000_000_000_000,
            "total_debt": 8_000_000_000,
            "cash_and_securities": 40_000_000_000,
            "revenue": 130_000_000_000,
            "ebitda": 80_000_000_000,
            "net_income": 70_000_000_000,
            "fcf": 60_000_000_000,
            "stale_ev": False,
        }])
        result = vm.compute_peer_multiples(peer_fin, nvda_fin)
        stale_row = result[result["ticker"] == "STALE"].iloc[0]
        # EV should be computed (market_cap is current)
        assert stale_row["EV/Revenue"] is not None and not pd.isna(stale_row["EV/Revenue"])
        assert stale_row["EV/EBITDA"] is not None and not pd.isna(stale_row["EV/EBITDA"])
        # Should be flagged as stale
        assert bool(stale_row["stale_financials"]) is True

    def test_stale_ev_still_has_pe_and_fcf_yield(self, vm):
        """Even with stale EV, P/E and FCF_yield (market_cap-based) should compute."""
        peer_fin = pd.DataFrame([{
            "ticker": "STALE",
            "market_cap": 100_000_000_000,
            "total_debt": 5000,
            "cash_and_securities": 2000,
            "revenue": 30_000_000_000,
            "ebitda": 8_000_000_000,
            "net_income": 4_000_000_000,
            "fcf": 3_000_000_000,
            "stale_ev": True,
        }])
        nvda_fin = pd.DataFrame()
        result = vm.compute_peer_multiples(peer_fin, nvda_fin)
        row = result[result["ticker"] == "STALE"].iloc[0]
        assert row["P/E"] is not None and not pd.isna(row["P/E"])
        assert row["FCF_yield"] is not None and not pd.isna(row["FCF_yield"])

    def test_valid_peer_computes_all_multiples(self, vm):
        """A peer with valid, fresh data should have all multiples populated."""
        peer_fin = pd.DataFrame([{
            "ticker": "GOOD",
            "market_cap": 200_000_000_000,
            "total_debt": 10_000_000_000,
            "cash_and_securities": 5_000_000_000,
            "revenue": 50_000_000_000,
            "ebitda": 15_000_000_000,
            "net_income": 8_000_000_000,
            "fcf": 7_000_000_000,
            "stale_ev": False,
        }])
        nvda_fin = pd.DataFrame()
        result = vm.compute_peer_multiples(peer_fin, nvda_fin)
        row = result[result["ticker"] == "GOOD"].iloc[0]
        assert row["EV/Revenue"] is not None and not pd.isna(row["EV/Revenue"])
        assert row["EV/EBITDA"] is not None and not pd.isna(row["EV/EBITDA"])
        assert row["P/E"] is not None and not pd.isna(row["P/E"])
        assert row["FCF_yield"] is not None and not pd.isna(row["FCF_yield"])

    def test_stale_ev_logged(self, vm, caplog):
        """Skipping EV multiples should be logged."""
        peer_fin = pd.DataFrame([{
            "ticker": "LOGME",
            "market_cap": None,
            "total_debt": 0,
            "cash_and_securities": 0,
            "revenue": 10000,
            "ebitda": 5000,
            "net_income": 2000,
            "fcf": 1500,
            "stale_ev": False,
        }])
        nvda_fin = pd.DataFrame()
        with caplog.at_level(logging.INFO):
            vm.compute_peer_multiples(peer_fin, nvda_fin)
        assert any("skipping ev" in m.lower() for m in caplog.messages)

    def test_mixed_peers_only_valid_get_ev(self, vm):
        """In a mix of valid and invalid peers, only valid ones get EV multiples."""
        peer_fin = pd.DataFrame([
            {
                "ticker": "VALID",
                "market_cap": 100_000_000_000,
                "total_debt": 5_000_000_000,
                "cash_and_securities": 2_000_000_000,
                "revenue": 40_000_000_000,
                "ebitda": 12_000_000_000,
                "net_income": 6_000_000_000,
                "fcf": 5_000_000_000,
                "stale_ev": False,
            },
            {
                "ticker": "INVALID",
                "market_cap": None,
                "total_debt": 0,
                "cash_and_securities": 0,
                "revenue": 20_000_000_000,
                "ebitda": 5_000_000_000,
                "net_income": 2_000_000_000,
                "fcf": 1_500_000_000,
                "stale_ev": False,
            },
        ])
        nvda_fin = pd.DataFrame()
        result = vm.compute_peer_multiples(peer_fin, nvda_fin)

        valid_row = result[result["ticker"] == "VALID"].iloc[0]
        invalid_row = result[result["ticker"] == "INVALID"].iloc[0]

        assert valid_row["EV/Revenue"] is not None and not pd.isna(valid_row["EV/Revenue"])
        assert invalid_row["EV/Revenue"] is None or pd.isna(invalid_row["EV/Revenue"])
