"""
Tests for src/peer_freshness.py — Task 1.4 (Req 5.1–5.10).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.config import EngineConfig
from src.peer_freshness import (
    MarketPriceFreshnessGate,
    MarketPriceFreshnessReport,
    PeerFinancialsFreshnessGate,
    PeerFinancialsFreshnessReport,
    _calendar_days_between,
    _parse_date,
    _trading_days_between,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeFetcher:
    """Minimal stand-in for EdgarFetcher used in unit tests."""

    def __init__(
        self,
        market_prices_response: pd.DataFrame | None = None,
        peer_financials_response: pd.DataFrame | None = None,
        raise_on_market: Exception | None = None,
        raise_on_peers: Exception | None = None,
    ) -> None:
        self.market_calls = 0
        self.peer_calls = 0
        self._market_response = market_prices_response
        self._peers_response = peer_financials_response
        self._raise_on_market = raise_on_market
        self._raise_on_peers = raise_on_peers

    def fetch_market_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        self.market_calls += 1
        if self._raise_on_market:
            raise self._raise_on_market
        if self._market_response is not None:
            return self._market_response
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "source_available_date"])

    def fetch_peer_financials(self, tickers: list[str]) -> pd.DataFrame:
        self.peer_calls += 1
        if self._raise_on_peers:
            raise self._raise_on_peers
        if self._peers_response is not None:
            return self._peers_response
        return pd.DataFrame()


def _make_prices(per_ticker_dates: dict[str, str]) -> pd.DataFrame:
    """Build a market_prices DataFrame with a single row per ticker at
    the given date string."""
    rows: list[dict[str, Any]] = []
    for ticker, d in per_ticker_dates.items():
        rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": ticker,
                "adj_close": 100.0,
                "source_available_date": d,
            }
        )
    return pd.DataFrame(rows)


def _make_peer_financials(per_peer_dates: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, d in per_peer_dates.items():
        rows.append(
            {
                "ticker": ticker,
                "market_cap": 1e12,
                "total_debt": 0,
                "total_cash": 1e10,
                "revenue": 1e11,
                "ebitda": 5e10,
                "net_income": 4e10,
                "free_cash_flow": 3e10,
                "source_date": d,
                "source_available_date": d,
                "stale_ev": False,
                "retrieval_timestamp": d,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_parse_date_handles_string() -> None:
    assert _parse_date("2026-05-16") == date(2026, 5, 16)


def test_parse_date_handles_timestamp() -> None:
    assert _parse_date(pd.Timestamp("2026-05-16")) == date(2026, 5, 16)


def test_parse_date_handles_invalid() -> None:
    assert _parse_date("not-a-date") is None
    assert _parse_date(None) is None


def test_trading_days_between_weekdays() -> None:
    # Mon → Fri = 4 trading days between
    assert _trading_days_between(date(2026, 5, 11), date(2026, 5, 15)) == 4


def test_trading_days_between_skips_weekends() -> None:
    # Friday → Monday = 1 trading day between (skips weekend)
    assert _trading_days_between(date(2026, 5, 15), date(2026, 5, 18)) == 1


def test_trading_days_between_zero_when_reversed() -> None:
    assert _trading_days_between(date(2026, 5, 18), date(2026, 5, 11)) == 0


def test_calendar_days_between() -> None:
    assert _calendar_days_between(date(2026, 1, 1), date(2026, 1, 31)) == 30
    assert _calendar_days_between(date(2026, 5, 16), date(2026, 5, 16)) == 0


# ---------------------------------------------------------------------------
# MarketPriceFreshnessGate
# ---------------------------------------------------------------------------


def test_market_gate_fresh_data_no_refresh() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = MarketPriceFreshnessGate(cfg)
    fetcher = _FakeFetcher()

    # All tickers fresh: latest price = 1 trading day before report
    prices = _make_prices(
        {
            "NVDA": "2026-05-14",
            "^SOX": "2026-05-14",
            "^GSPC": "2026-05-14",
            "AMD": "2026-05-14",
        }
    )
    report = gate.evaluate_and_refresh(
        prices,
        report_date="2026-05-15",
        edgar_fetcher=fetcher,
        required_tickers=["NVDA", "^SOX", "^GSPC", "AMD"],
    )
    assert report.status == "fresh"
    assert report.tickers_excluded_stale == []
    assert report.nvda_stale is False
    assert fetcher.market_calls == 0  # no refresh triggered


def test_market_gate_stale_triggers_refresh() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = MarketPriceFreshnessGate(cfg)

    # Stale prices (10 trading days old)
    stale_prices = _make_prices({"NVDA": "2026-05-01", "^SOX": "2026-05-01"})
    # Refresh response: fresh
    fresh_prices = _make_prices(
        {"NVDA": "2026-05-14", "^SOX": "2026-05-14"}
    )
    fetcher = _FakeFetcher(market_prices_response=fresh_prices)

    report = gate.evaluate_and_refresh(
        stale_prices,
        report_date="2026-05-15",
        edgar_fetcher=fetcher,
        required_tickers=["NVDA", "^SOX"],
    )
    assert fetcher.market_calls == 1
    assert report.status == "refreshed"
    assert report.nvda_stale is False
    assert report.tickers_refreshed == ["NVDA", "^SOX"]


def test_market_gate_blocks_when_nvda_stale_after_refresh() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = MarketPriceFreshnessGate(cfg)

    # Stale + refresh returns equally stale data
    stale = _make_prices({"NVDA": "2026-05-01", "^SOX": "2026-05-14"})
    fetcher = _FakeFetcher(market_prices_response=stale)

    report = gate.evaluate_and_refresh(
        stale,
        report_date="2026-05-15",
        edgar_fetcher=fetcher,
        required_tickers=["NVDA", "^SOX"],
    )
    assert report.status == "blocked_nvda_stale"
    assert report.nvda_stale is True


def test_market_gate_partial_when_some_stale_others_fresh() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = MarketPriceFreshnessGate(cfg)

    # NVDA fresh, AMD stale, refresh returns only NVDA fresh
    initial = _make_prices(
        {"NVDA": "2026-05-14", "^SOX": "2026-05-14", "AMD": "2026-04-01"}
    )
    refresh_response = _make_prices(
        {"NVDA": "2026-05-14", "^SOX": "2026-05-14", "AMD": "2026-04-01"}
    )
    fetcher = _FakeFetcher(market_prices_response=refresh_response)

    report = gate.evaluate_and_refresh(
        initial,
        report_date="2026-05-15",
        edgar_fetcher=fetcher,
        required_tickers=["NVDA", "^SOX", "AMD"],
    )
    assert report.status == "partial"
    assert "AMD" in report.tickers_excluded_stale
    assert report.nvda_stale is False


def test_market_gate_no_data_returns_no_data_status() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = MarketPriceFreshnessGate(cfg)
    fetcher = _FakeFetcher()

    empty = pd.DataFrame(columns=["date", "ticker", "adj_close", "source_available_date"])
    report = gate.evaluate_and_refresh(
        empty,
        report_date="2026-05-15",
        edgar_fetcher=fetcher,
        required_tickers=["NVDA", "^SOX"],
    )
    assert report.status == "no_data"
    assert report.nvda_stale is True
    assert fetcher.market_calls == 0


def test_market_gate_refresh_failure_falls_back() -> None:
    """If the refresh raises, the gate must NOT crash; it falls back to
    reporting the existing-cache staleness."""
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = MarketPriceFreshnessGate(cfg)

    stale = _make_prices({"NVDA": "2026-05-01", "^SOX": "2026-05-14"})
    fetcher = _FakeFetcher(raise_on_market=ConnectionError("network down"))

    report = gate.evaluate_and_refresh(
        stale,
        report_date="2026-05-15",
        edgar_fetcher=fetcher,
        required_tickers=["NVDA", "^SOX"],
    )
    # Refresh attempted, failed → original staleness preserved
    assert report.status == "blocked_nvda_stale"
    assert "NVDA" in report.tickers_excluded_stale


def test_market_gate_write_report(tmp_path: Path) -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = tmp_path
    gate = MarketPriceFreshnessGate(cfg)
    fetcher = _FakeFetcher()
    prices = _make_prices({"NVDA": "2026-05-14", "^SOX": "2026-05-14"})
    report = gate.evaluate_and_refresh(
        prices,
        "2026-05-15",
        fetcher,
        required_tickers=["NVDA", "^SOX"],
    )
    out = gate.write_report(report)
    assert out.exists()
    with open(out) as f:
        loaded = json.load(f)
    assert loaded["status"] == "fresh"
    assert loaded["threshold_trading_days"] == 5


# ---------------------------------------------------------------------------
# PeerFinancialsFreshnessGate
# ---------------------------------------------------------------------------


def test_peer_fin_gate_fresh() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = PeerFinancialsFreshnessGate(cfg)
    fetcher = _FakeFetcher()
    fresh = _make_peer_financials(
        {"AMD": "2026-05-10", "AVGO": "2026-05-10", "INTC": "2026-05-10"}
    )
    report = gate.evaluate_and_refresh(fresh, "2026-05-15", fetcher)
    assert report.status == "fresh"
    assert report.peers_excluded_stale == []
    assert fetcher.peer_calls == 0


def test_peer_fin_gate_stale_triggers_refresh() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = PeerFinancialsFreshnessGate(cfg)
    # 60 days old → stale (default 30-day threshold)
    stale = _make_peer_financials({"AMD": "2026-03-15", "AVGO": "2026-03-15"})
    fresh_response = _make_peer_financials(
        {"AMD": "2026-05-10", "AVGO": "2026-05-10"}
    )
    fetcher = _FakeFetcher(peer_financials_response=fresh_response)
    report = gate.evaluate_and_refresh(stale, "2026-05-15", fetcher)
    assert fetcher.peer_calls == 1
    assert report.status == "refreshed"
    assert report.peers_excluded_stale == []


def test_peer_fin_gate_blocks_when_three_core_peers_stale() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = PeerFinancialsFreshnessGate(cfg)
    # 3 of the 5 core peers stale, refresh fails to fix them
    stale = _make_peer_financials(
        {
            "AMD": "2026-01-01",
            "AVGO": "2026-01-01",
            "INTC": "2026-01-01",
            "QCOM": "2026-05-10",
            "MRVL": "2026-05-10",
        }
    )
    fetcher = _FakeFetcher(peer_financials_response=stale)
    report = gate.evaluate_and_refresh(stale, "2026-05-15", fetcher)
    assert report.status == "stale_blocked"
    assert {"AMD", "AVGO", "INTC"}.issubset(set(report.peers_excluded_stale))


def test_peer_fin_gate_partial_when_two_excluded() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = PeerFinancialsFreshnessGate(cfg)
    # Only 2 of 5 core stale → partial, not blocked
    state = _make_peer_financials(
        {
            "AMD": "2026-01-01",
            "AVGO": "2026-01-01",
            "INTC": "2026-05-10",
            "QCOM": "2026-05-10",
            "MRVL": "2026-05-10",
        }
    )
    fetcher = _FakeFetcher(peer_financials_response=state)
    report = gate.evaluate_and_refresh(state, "2026-05-15", fetcher)
    assert report.status == "partial"
    assert {"AMD", "AVGO"}.issubset(set(report.peers_excluded_stale))


def test_peer_fin_gate_no_data() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = PeerFinancialsFreshnessGate(cfg)
    fetcher = _FakeFetcher()
    report = gate.evaluate_and_refresh(pd.DataFrame(), "2026-05-15", fetcher)
    assert report.status == "no_data"


def test_peer_fin_gate_refresh_exception_does_not_crash() -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = Path("/tmp/test_v2_freshness_outputs")
    gate = PeerFinancialsFreshnessGate(cfg)
    stale = _make_peer_financials({"AMD": "2026-01-01"})
    fetcher = _FakeFetcher(raise_on_peers=RuntimeError("oops"))
    # Should not crash
    report = gate.evaluate_and_refresh(stale, "2026-05-15", fetcher)
    assert "AMD" in report.peers_excluded_stale


def test_peer_fin_gate_write_report(tmp_path: Path) -> None:
    cfg = EngineConfig(report_date="2026-05-15")
    cfg.outputs_dir = tmp_path
    gate = PeerFinancialsFreshnessGate(cfg)
    fetcher = _FakeFetcher()
    fresh = _make_peer_financials({"AMD": "2026-05-10"})
    report = gate.evaluate_and_refresh(fresh, "2026-05-15", fetcher)
    out = gate.write_report(report)
    assert out.exists()
    with open(out) as f:
        loaded = json.load(f)
    assert loaded["threshold_days"] == 30
