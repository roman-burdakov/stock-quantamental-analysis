"""
Tests for peer multiples filtering and quality gates.

Covers (Reqs 21.1–21.7):
  - NaN/negative EV row exclusion from EV-based charts
  - Negative P/E exclusion from primary P/E chart
  - Negative EBITDA exclusion from EV/EBITDA chart
  - Stale peer data exclusion
  - Peer tier separation (semi vs infrastructure vs context)
  - Limited sample warning
  - PeerRowStatus assignment for each exclusion reason

All tests run offline with inline test data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import get_default_config
from src.valuation import PeerRowStatus, ValuationModule


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vm() -> ValuationModule:
    """ValuationModule with default config."""
    return ValuationModule(get_default_config())


def _make_peer_row(
    ticker: str = "AMD",
    enterprise_value: float | None = 100_000.0,
    ev_revenue: float | None = 5.0,
    ev_ebitda: float | None = 20.0,
    pe: float | None = 30.0,
    fcf_yield: float | None = 0.03,
    market_cap: float | None = 120_000.0,
    source_date: str = "2026-04-01",
) -> dict:
    return {
        "ticker": ticker,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "EV/Revenue": ev_revenue,
        "EV/EBITDA": ev_ebitda,
        "P/E": pe,
        "FCF_yield": fcf_yield,
        "source_date": source_date,
    }


def _make_peer_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. NaN/negative EV row exclusion from EV-based charts (Req 21.2)
# ---------------------------------------------------------------------------

class TestEVExclusion:
    """Rows with NaN or negative EV are excluded from EV/Revenue and EV/EBITDA."""

    def test_nan_ev_excluded_from_ev_multiples(self, vm):
        """NaN enterprise_value → EV/Revenue and EV/EBITDA set to NaN/None."""
        df = _make_peer_df([
            _make_peer_row(ticker="AMD", enterprise_value=None, ev_revenue=5.0, ev_ebitda=20.0),
            _make_peer_row(ticker="AVGO", enterprise_value=200_000.0),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        amd = filtered[filtered["ticker"] == "AMD"].iloc[0]
        assert pd.isna(amd["EV/Revenue"])
        assert pd.isna(amd["EV/EBITDA"])
        assert amd["row_status"] == PeerRowStatus.MISSING_EV
        # AVGO should be unaffected
        avgo = filtered[filtered["ticker"] == "AVGO"].iloc[0]
        assert avgo["EV/Revenue"] == 5.0
        assert avgo["row_status"] == PeerRowStatus.USABLE

    def test_negative_ev_excluded_from_ev_multiples(self, vm):
        """Negative enterprise_value → EV/Revenue and EV/EBITDA set to NaN/None."""
        df = _make_peer_df([
            _make_peer_row(ticker="INTC", enterprise_value=-50_000.0, ev_revenue=3.0, ev_ebitda=15.0),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        intc = filtered[filtered["ticker"] == "INTC"].iloc[0]
        assert pd.isna(intc["EV/Revenue"])
        assert pd.isna(intc["EV/EBITDA"])
        assert intc["row_status"] == PeerRowStatus.NEGATIVE_EV

    def test_nan_ev_logged(self, vm):
        """NaN EV exclusion appears in the exclusion log."""
        df = _make_peer_df([
            _make_peer_row(ticker="AMD", enterprise_value=float("nan")),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        assert any("AMD" in entry and "missing" in entry.lower() for entry in log)

    def test_pe_and_fcf_yield_preserved_when_ev_excluded(self, vm):
        """P/E and FCF_yield remain when only EV is problematic."""
        df = _make_peer_df([
            _make_peer_row(ticker="AMD", enterprise_value=None, pe=25.0, fcf_yield=0.04),
        ])
        filtered, _ = vm.filter_peer_multiples(df)
        amd = filtered[filtered["ticker"] == "AMD"].iloc[0]
        assert amd["P/E"] == 25.0
        assert amd["FCF_yield"] == 0.04


# ---------------------------------------------------------------------------
# 2. Negative P/E exclusion from primary P/E chart (Req 21.3)
# ---------------------------------------------------------------------------

class TestNegativePEExclusion:
    """Rows with negative P/E are excluded from primary P/E chart."""

    def test_negative_pe_excluded(self, vm):
        """Negative P/E → P/E set to NaN/None, row_status updated."""
        df = _make_peer_df([
            _make_peer_row(ticker="INTC", pe=-12.0),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        intc = filtered[filtered["ticker"] == "INTC"].iloc[0]
        assert pd.isna(intc["P/E"])
        assert intc["row_status"] == PeerRowStatus.NEGATIVE_EARNINGS

    def test_negative_pe_logged(self, vm):
        """Negative P/E exclusion appears in the exclusion log."""
        df = _make_peer_df([
            _make_peer_row(ticker="INTC", pe=-5.0),
        ])
        _, log = vm.filter_peer_multiples(df)
        assert any("INTC" in entry and "P/E" in entry for entry in log)

    def test_positive_pe_not_excluded(self, vm):
        """Positive P/E → row remains usable."""
        df = _make_peer_df([
            _make_peer_row(ticker="AMD", pe=45.0),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        amd = filtered[filtered["ticker"] == "AMD"].iloc[0]
        assert amd["P/E"] == 45.0
        assert amd["row_status"] == PeerRowStatus.USABLE


# ---------------------------------------------------------------------------
# 3. Negative EBITDA exclusion from EV/EBITDA chart (Req 21.3)
# ---------------------------------------------------------------------------

class TestNegativeEBITDAExclusion:
    """Rows with negative EV/EBITDA are excluded from EV/EBITDA chart."""

    def test_negative_ev_ebitda_excluded(self, vm):
        """Negative EV/EBITDA → EV/EBITDA set to NaN/None."""
        df = _make_peer_df([
            _make_peer_row(ticker="INTC", ev_ebitda=-8.0),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        intc = filtered[filtered["ticker"] == "INTC"].iloc[0]
        assert pd.isna(intc["EV/EBITDA"])
        assert intc["row_status"] == PeerRowStatus.NEGATIVE_EBITDA

    def test_negative_ebitda_logged(self, vm):
        """Negative EBITDA exclusion appears in the exclusion log."""
        df = _make_peer_df([
            _make_peer_row(ticker="INTC", ev_ebitda=-3.0),
        ])
        _, log = vm.filter_peer_multiples(df)
        assert any("INTC" in entry and "EBITDA" in entry for entry in log)

    def test_ev_revenue_preserved_when_ebitda_negative(self, vm):
        """EV/Revenue remains valid when only EBITDA is negative."""
        df = _make_peer_df([
            _make_peer_row(ticker="INTC", ev_ebitda=-3.0, ev_revenue=4.0),
        ])
        filtered, _ = vm.filter_peer_multiples(df)
        intc = filtered[filtered["ticker"] == "INTC"].iloc[0]
        assert intc["EV/Revenue"] == 4.0


# ---------------------------------------------------------------------------
# 4. Stale peer data exclusion (Req 21.7)
# ---------------------------------------------------------------------------

class TestStalePeerExclusion:
    """Peers with source_date > staleness threshold are excluded."""

    def test_stale_peer_excluded(self, vm):
        """source_date > 90 days before report_date → all multiples nulled."""
        # report_date is 2026-04-25, so 2025-12-01 is ~145 days old
        df = _make_peer_df([
            _make_peer_row(ticker="QCOM", source_date="2025-12-01"),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        qcom = filtered[filtered["ticker"] == "QCOM"].iloc[0]
        assert qcom["row_status"] == PeerRowStatus.STALE_MARKET_DATA
        assert pd.isna(qcom["EV/Revenue"])
        assert pd.isna(qcom["EV/EBITDA"])
        assert pd.isna(qcom["P/E"])
        assert pd.isna(qcom["FCF_yield"])

    def test_stale_peer_logged(self, vm):
        """Stale exclusion appears in the exclusion log."""
        df = _make_peer_df([
            _make_peer_row(ticker="QCOM", source_date="2025-12-01"),
        ])
        _, log = vm.filter_peer_multiples(df)
        assert any("QCOM" in entry and "stale" in entry.lower() for entry in log)

    def test_fresh_peer_not_excluded(self, vm):
        """source_date within threshold → row remains usable."""
        # report_date is 2026-04-25, so 2026-03-01 is ~55 days old
        df = _make_peer_df([
            _make_peer_row(ticker="AMD", source_date="2026-03-01"),
        ])
        filtered, log = vm.filter_peer_multiples(df)
        amd = filtered[filtered["ticker"] == "AMD"].iloc[0]
        assert amd["row_status"] == PeerRowStatus.USABLE

    def test_no_source_date_column_no_crash(self, vm):
        """DataFrame without source_date column → no staleness filtering, no crash."""
        df = pd.DataFrame([{
            "ticker": "AMD",
            "market_cap": 100_000.0,
            "enterprise_value": 120_000.0,
            "EV/Revenue": 5.0,
            "EV/EBITDA": 20.0,
            "P/E": 30.0,
            "FCF_yield": 0.03,
        }])
        filtered, log = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.USABLE


# ---------------------------------------------------------------------------
# 5. Peer tier separation (Reqs 21.4, 21.5)
# ---------------------------------------------------------------------------

class TestPeerTierSeparation:
    """Semi, infrastructure, and context peers are correctly labeled."""

    def test_semi_peers_labeled(self, vm):
        """Core semiconductor peers get tier='semi'."""
        rows = [_make_peer_row(ticker=t) for t in ["AMD", "AVGO", "INTC", "QCOM", "MRVL"]]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        for _, row in filtered.iterrows():
            assert row["peer_tier"] == "semi", f"{row['ticker']} should be semi"

    def test_infrastructure_peers_labeled(self, vm):
        """Infrastructure peers get tier='infrastructure'."""
        rows = [_make_peer_row(ticker=t) for t in ["TSM", "ASML"]]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        for _, row in filtered.iterrows():
            assert row["peer_tier"] == "infrastructure", f"{row['ticker']} should be infrastructure"

    def test_context_peers_labeled(self, vm):
        """AI capex context peers get tier='context'."""
        rows = [_make_peer_row(ticker=t) for t in ["MSFT", "AMZN", "GOOGL", "META"]]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        for _, row in filtered.iterrows():
            assert row["peer_tier"] == "context", f"{row['ticker']} should be context"

    def test_unknown_ticker_defaults_to_context(self, vm):
        """Unknown ticker defaults to 'context' tier."""
        df = _make_peer_df([_make_peer_row(ticker="UNKNOWN_CO")])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["peer_tier"] == "context"

    def test_mixed_tiers_in_single_df(self, vm):
        """A mixed DataFrame correctly separates all three tiers."""
        rows = [
            _make_peer_row(ticker="AMD"),
            _make_peer_row(ticker="TSM"),
            _make_peer_row(ticker="MSFT"),
        ]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        tiers = dict(zip(filtered["ticker"], filtered["peer_tier"]))
        assert tiers["AMD"] == "semi"
        assert tiers["TSM"] == "infrastructure"
        assert tiers["MSFT"] == "context"


# ---------------------------------------------------------------------------
# 6. Limited sample warning (Req 21.6)
# ---------------------------------------------------------------------------

class TestLimitedSampleWarning:
    """Warning when fewer than 3 semi peers have valid EV multiples."""

    def test_warning_when_fewer_than_3_semi_valid(self, vm):
        """Only 2 semi peers with valid EV → 'limited peer sample'."""
        rows = [
            _make_peer_row(ticker="AMD", ev_revenue=5.0),
            _make_peer_row(ticker="AVGO", ev_revenue=8.0),
            _make_peer_row(ticker="INTC", enterprise_value=None, ev_revenue=None),
            _make_peer_row(ticker="QCOM", enterprise_value=None, ev_revenue=None),
            _make_peer_row(ticker="MRVL", enterprise_value=None, ev_revenue=None),
        ]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        warning = vm.check_limited_peer_sample(filtered)
        assert warning == "limited peer sample"

    def test_no_warning_when_3_or_more_semi_valid(self, vm):
        """3+ semi peers with valid EV → no warning."""
        rows = [
            _make_peer_row(ticker="AMD", ev_revenue=5.0),
            _make_peer_row(ticker="AVGO", ev_revenue=8.0),
            _make_peer_row(ticker="INTC", ev_revenue=4.0),
        ]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        warning = vm.check_limited_peer_sample(filtered)
        assert warning is None

    def test_warning_when_no_semi_peers(self, vm):
        """No semi peers at all → 'limited peer sample'."""
        rows = [
            _make_peer_row(ticker="MSFT", ev_revenue=10.0),
            _make_peer_row(ticker="AMZN", ev_revenue=8.0),
        ]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        warning = vm.check_limited_peer_sample(filtered)
        assert warning == "limited peer sample"

    def test_warning_when_empty_df(self, vm):
        """Empty DataFrame → 'limited peer sample'."""
        df = pd.DataFrame(columns=["ticker", "peer_tier", "EV/Revenue"])
        warning = vm.check_limited_peer_sample(df)
        assert warning == "limited peer sample"

    def test_context_peers_dont_count_for_semi_threshold(self, vm):
        """Context peers with valid EV don't satisfy the semi threshold."""
        rows = [
            _make_peer_row(ticker="AMD", ev_revenue=5.0),
            _make_peer_row(ticker="MSFT", ev_revenue=10.0),
            _make_peer_row(ticker="AMZN", ev_revenue=8.0),
            _make_peer_row(ticker="GOOGL", ev_revenue=12.0),
        ]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        warning = vm.check_limited_peer_sample(filtered)
        assert warning == "limited peer sample"


# ---------------------------------------------------------------------------
# 7. PeerRowStatus assignment for each exclusion reason (Req 21.1)
# ---------------------------------------------------------------------------

class TestPeerRowStatusAssignment:
    """Each exclusion reason maps to the correct PeerRowStatus constant."""

    def test_usable_status(self, vm):
        df = _make_peer_df([_make_peer_row(ticker="AMD")])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.USABLE

    def test_missing_ev_status(self, vm):
        df = _make_peer_df([_make_peer_row(ticker="AMD", enterprise_value=None)])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.MISSING_EV

    def test_negative_ev_status(self, vm):
        df = _make_peer_df([_make_peer_row(ticker="AMD", enterprise_value=-1000.0)])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.NEGATIVE_EV

    def test_negative_ebitda_status(self, vm):
        df = _make_peer_df([_make_peer_row(ticker="AMD", ev_ebitda=-5.0)])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.NEGATIVE_EBITDA

    def test_negative_earnings_status(self, vm):
        df = _make_peer_df([_make_peer_row(ticker="AMD", pe=-10.0, ev_ebitda=20.0)])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.NEGATIVE_EARNINGS

    def test_stale_market_data_status(self, vm):
        df = _make_peer_df([_make_peer_row(ticker="AMD", source_date="2025-01-01")])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.STALE_MARKET_DATA

    def test_stale_takes_priority_over_negative_ev(self, vm):
        """Staleness is checked first; stale rows don't get further EV checks."""
        df = _make_peer_df([
            _make_peer_row(ticker="AMD", enterprise_value=-1000.0, source_date="2025-01-01"),
        ])
        filtered, _ = vm.filter_peer_multiples(df)
        assert filtered.iloc[0]["row_status"] == PeerRowStatus.STALE_MARKET_DATA

    def test_all_rows_have_row_status(self, vm):
        """Every row in the output has a non-empty row_status."""
        rows = [
            _make_peer_row(ticker="AMD"),
            _make_peer_row(ticker="INTC", enterprise_value=None),
            _make_peer_row(ticker="TSM", pe=-5.0),
        ]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        for _, row in filtered.iterrows():
            assert row["row_status"] is not None
            assert row["row_status"] != ""

    def test_all_rows_have_peer_tier(self, vm):
        """Every row in the output has a peer_tier column."""
        rows = [
            _make_peer_row(ticker="AMD"),
            _make_peer_row(ticker="TSM"),
            _make_peer_row(ticker="MSFT"),
        ]
        df = _make_peer_df(rows)
        filtered, _ = vm.filter_peer_multiples(df)
        for _, row in filtered.iterrows():
            assert row["peer_tier"] in ("semi", "infrastructure", "context")
