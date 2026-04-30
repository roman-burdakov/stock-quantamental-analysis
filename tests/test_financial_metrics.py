"""
Tests for src.financial_metrics — Financial Metrics Calculator.

Covers:
  - Verify calculations against manually computed values for ≥3 fiscal periods
  - Round-trip property: recomputing from same inputs produces identical results
  - Null handling: missing inputs → None, no crash

All tests run offline using fixtures in tests/fixtures/.
Reqs: 12.1
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import EngineConfig
from src.financial_metrics import FinancialMetricsCalculator
from src.xbrl_parser import XBRLParser


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures")


@pytest.fixture
def companyfacts() -> dict:
    with open(FIXTURES / "sample_companyfacts.json") as f:
        return json.load(f)


@pytest.fixture
def known_values() -> dict:
    with open(FIXTURES / "known_validation_values.json") as f:
        return json.load(f)


@pytest.fixture
def parsed_xbrl(companyfacts) -> pd.DataFrame:
    """Parse the sample fixture through XBRLParser."""
    parser = XBRLParser(EngineConfig())
    return parser.parse_companyfacts(companyfacts)


@pytest.fixture
def metrics_df(parsed_xbrl, tmp_path) -> pd.DataFrame:
    """Compute all metrics from parsed XBRL data, writing to tmp_path."""
    config = EngineConfig()
    config.processed_dir = tmp_path
    calc = FinancialMetricsCalculator(config)
    return calc.compute_all_metrics(parsed_xbrl)


def _get_metric(df: pd.DataFrame, period: str, name: str):
    """Extract a single metric_value from the metrics DataFrame."""
    row = df[(df["fiscal_period"] == period) & (df["metric_name"] == name)]
    if row.empty:
        return None
    return row.iloc[0]["metric_value"]


# ===================================================================
# 1. Verify calculations against manually computed values (≥3 periods)
# ===================================================================


class TestMetricCalculations:
    """Verify computed metrics match manually derived values from the fixture."""

    # -- FY2024 --

    def test_gross_margin_fy2024(self, metrics_df, known_values):
        """gross_margin = gross_profit / revenue for FY2024."""
        gm = _get_metric(metrics_df, "FY2024", "gross_margin")
        # From fixture: gross_profit=44301e9, revenue=60922e9
        expected = 44301000000 / 60922000000
        assert gm == pytest.approx(expected, rel=1e-6)

    def test_operating_margin_fy2024(self, metrics_df):
        """operating_margin = operating_income / revenue for FY2024."""
        om = _get_metric(metrics_df, "FY2024", "operating_margin")
        expected = 32972000000 / 60922000000
        assert om == pytest.approx(expected, rel=1e-6)

    def test_fcf_fy2024(self, metrics_df):
        """FCF = operating_cash_flow - capex for FY2024."""
        fcf = _get_metric(metrics_df, "FY2024", "FCF")
        expected = 28090000000 - 1069000000  # 27021e9
        assert fcf == pytest.approx(expected, rel=1e-6)

    def test_net_margin_fy2024(self, metrics_df):
        """net_margin = net_income / revenue for FY2024."""
        nm = _get_metric(metrics_df, "FY2024", "net_margin")
        expected = 29760000000 / 60922000000
        assert nm == pytest.approx(expected, rel=1e-6)

    # -- FY2025 --

    def test_gross_margin_fy2025(self, metrics_df):
        """gross_margin = gross_profit / revenue for FY2025."""
        gm = _get_metric(metrics_df, "FY2025", "gross_margin")
        expected = 101329000000 / 130497000000
        assert gm == pytest.approx(expected, rel=1e-6)

    def test_operating_margin_fy2025(self, metrics_df):
        """operating_margin = operating_income / revenue for FY2025."""
        om = _get_metric(metrics_df, "FY2025", "operating_margin")
        expected = 81450000000 / 130497000000
        assert om == pytest.approx(expected, rel=1e-6)

    def test_fcf_fy2025(self, metrics_df):
        """FCF = operating_cash_flow - capex for FY2025."""
        fcf = _get_metric(metrics_df, "FY2025", "FCF")
        expected = 64089000000 - 3233000000  # 60856e9
        assert fcf == pytest.approx(expected, rel=1e-6)

    def test_net_margin_fy2025(self, metrics_df):
        """net_margin = net_income / revenue for FY2025."""
        nm = _get_metric(metrics_df, "FY2025", "net_margin")
        expected = 72880000000 / 130497000000
        assert nm == pytest.approx(expected, rel=1e-6)

    # -- Q3 FY2025 (third period) --

    def test_gross_margin_q3_fy2025(self, metrics_df):
        """gross_margin = gross_profit / revenue for Q3 FY2025."""
        gm = _get_metric(metrics_df, "FY2025-Q3", "gross_margin")
        expected = 27266000000 / 35082000000
        assert gm == pytest.approx(expected, rel=1e-6)

    def test_operating_margin_q3_fy2025(self, metrics_df):
        """operating_margin = operating_income / revenue for Q3 FY2025."""
        om = _get_metric(metrics_df, "FY2025-Q3", "operating_margin")
        expected = 21869000000 / 35082000000
        assert om == pytest.approx(expected, rel=1e-6)

    # -- Cross-check against known_validation_values --

    def test_fcf_matches_known_fy2024(self, metrics_df, known_values):
        """FCF should match the known validation value for FY2024."""
        fcf = _get_metric(metrics_df, "FY2024", "FCF")
        expected_fcf = known_values["values"]["FY2024"]["fcf"]
        assert fcf == pytest.approx(expected_fcf, rel=0.02)

    def test_fcf_matches_known_fy2025(self, metrics_df, known_values):
        """FCF should match the known validation value for FY2025."""
        fcf = _get_metric(metrics_df, "FY2025", "FCF")
        expected_fcf = known_values["values"]["FY2025"]["fcf"]
        assert fcf == pytest.approx(expected_fcf, rel=0.02)

    def test_r_and_d_pct_revenue_fy2024(self, metrics_df):
        """R&D % revenue = r_and_d / revenue for FY2024."""
        rd_pct = _get_metric(metrics_df, "FY2024", "R&D_%_revenue")
        expected = 8675000000 / 60922000000
        assert rd_pct == pytest.approx(expected, rel=1e-6)

    def test_diluted_eps_fy2024(self, metrics_df):
        """Diluted EPS should pass through from parsed data for FY2024."""
        eps = _get_metric(metrics_df, "FY2024", "diluted_EPS")
        assert eps == pytest.approx(11.93, rel=1e-4)

    def test_diluted_eps_fy2025(self, metrics_df):
        """Diluted EPS should pass through from parsed data for FY2025."""
        eps = _get_metric(metrics_df, "FY2025", "diluted_EPS")
        assert eps == pytest.approx(2.94, rel=1e-4)


# ===================================================================
# 2. Round-trip property
# ===================================================================


class TestRoundTrip:
    """Recomputing from the same inputs produces identical results."""

    def test_recompute_produces_identical_results(self, parsed_xbrl, tmp_path):
        """Running compute_all_metrics twice on the same input yields the same output."""
        config = EngineConfig()
        config.processed_dir = tmp_path

        calc = FinancialMetricsCalculator(config)
        first = calc.compute_all_metrics(parsed_xbrl)
        second = calc.compute_all_metrics(parsed_xbrl)

        # Same shape
        assert first.shape == second.shape

        # Same columns
        assert list(first.columns) == list(second.columns)

        # Same values (compare row-by-row after sorting)
        first_sorted = first.sort_values(
            ["fiscal_period", "metric_name"]
        ).reset_index(drop=True)
        second_sorted = second.sort_values(
            ["fiscal_period", "metric_name"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(first_sorted, second_sorted)

    def test_recompute_with_fresh_calculator(self, parsed_xbrl, tmp_path):
        """A fresh calculator instance produces the same results."""
        config1 = EngineConfig()
        config1.processed_dir = tmp_path
        config2 = EngineConfig()
        config2.processed_dir = tmp_path

        first = FinancialMetricsCalculator(config1).compute_all_metrics(parsed_xbrl)
        second = FinancialMetricsCalculator(config2).compute_all_metrics(parsed_xbrl)

        first_sorted = first.sort_values(
            ["fiscal_period", "metric_name"]
        ).reset_index(drop=True)
        second_sorted = second.sort_values(
            ["fiscal_period", "metric_name"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(first_sorted, second_sorted)


# ===================================================================
# 3. Null handling
# ===================================================================


class TestNullHandling:
    """Missing inputs should produce None metrics, not crashes."""

    def test_missing_sga_produces_none(self, metrics_df):
        """SG&A is not in the fixture → SG&A_%_revenue should be null (None/NaN)."""
        sga_pct = _get_metric(metrics_df, "FY2024", "SG&A_%_revenue")
        assert sga_pct is None or pd.isna(sga_pct)

    def test_missing_equity_produces_none_roe(self, metrics_df):
        """shareholders_equity not in fixture → ROE should be null (None/NaN)."""
        roe = _get_metric(metrics_df, "FY2024", "ROE")
        assert roe is None or pd.isna(roe)

    def test_missing_total_assets_produces_none_roa(self, metrics_df):
        """total_assets not in fixture → ROA should be null (None/NaN)."""
        roa = _get_metric(metrics_df, "FY2024", "ROA")
        assert roa is None or pd.isna(roa)

    def test_empty_input_returns_empty_df(self, tmp_path):
        """Empty xbrl_data should return an empty DataFrame, not crash."""
        config = EngineConfig()
        config.processed_dir = tmp_path
        calc = FinancialMetricsCalculator(config)
        empty_df = pd.DataFrame(columns=[
            "ticker", "fiscal_period", "fiscal_year", "filing_date",
            "source_available_date", "accession_number", "metric_name",
            "value", "unit", "form_type",
        ])
        result = calc.compute_all_metrics(empty_df)
        assert result.empty

    def test_no_crash_with_all_none_values(self, tmp_path):
        """A period where all values are None should produce None metrics, not crash."""
        config = EngineConfig()
        config.processed_dir = tmp_path
        calc = FinancialMetricsCalculator(config)

        data = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2099,
                "filing_date": "2099-02-01", "source_available_date": "2099-02-01",
                "accession_number": "fake-accn", "metric_name": "revenue",
                "value": None, "unit": "USD", "form_type": "10-K",
            },
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2099,
                "filing_date": "2099-02-01", "source_available_date": "2099-02-01",
                "accession_number": "fake-accn", "metric_name": "gross_profit",
                "value": None, "unit": "USD", "form_type": "10-K",
            },
        ])
        result = calc.compute_all_metrics(data)
        assert not result.empty
        # All metric_values for this period should be None
        period_rows = result[result["fiscal_period"] == "FY2099"]
        assert period_rows["metric_value"].isna().all()

    def test_missing_capex_produces_none_fcf(self, tmp_path):
        """If capex is missing but operating_cash_flow exists, FCF should be None."""
        config = EngineConfig()
        config.processed_dir = tmp_path
        calc = FinancialMetricsCalculator(config)

        data = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2098,
                "filing_date": "2098-02-01", "source_available_date": "2098-02-01",
                "accession_number": "fake-accn", "metric_name": "operating_cash_flow",
                "value": 50000000000, "unit": "USD", "form_type": "10-K",
            },
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2098,
                "filing_date": "2098-02-01", "source_available_date": "2098-02-01",
                "accession_number": "fake-accn", "metric_name": "revenue",
                "value": 100000000000, "unit": "USD", "form_type": "10-K",
            },
        ])
        result = calc.compute_all_metrics(data)
        fcf = _get_metric(result, "FY2098", "FCF")
        assert fcf is None or pd.isna(fcf)

    def test_output_schema_columns(self, metrics_df):
        """Metrics output should contain all ComputedMetric schema columns."""
        required = {
            "ticker", "fiscal_period", "filing_date",
            "source_available_date", "metric_name", "metric_value",
            "source_accession", "unit",
        }
        assert required.issubset(set(metrics_df.columns))
