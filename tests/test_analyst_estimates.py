"""
Tests for src/analyst_estimates.py — Task 1.4 (Req 2.1–2.7).

Most yfinance interactions are mocked. The module-level contract test
that ``AnalystEstimateFetcher`` does not expose a ``to_features()``
method is enforced separately by ``tests/test_v2_spec_consistency.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.analyst_estimates import (
    AnalystEstimateFetcher,
    AnalystOverlay,
    AnalystSnapshot,
)
from src.config import EngineConfig


# ---------------------------------------------------------------------------
# Fixtures: simulated yfinance Ticker
# ---------------------------------------------------------------------------


class _FakeYTicker:
    """Stand-in for yfinance.Ticker with the schema fields we read."""

    def __init__(
        self,
        earnings_estimate: pd.DataFrame | None = None,
        revenue_estimate: pd.DataFrame | None = None,
        eps_revisions: pd.DataFrame | None = None,
        recommendations_summary: pd.DataFrame | None = None,
        info: dict | None = None,
    ) -> None:
        self.earnings_estimate = earnings_estimate if earnings_estimate is not None else pd.DataFrame()
        self.revenue_estimate = revenue_estimate if revenue_estimate is not None else pd.DataFrame()
        self.eps_revisions = eps_revisions if eps_revisions is not None else pd.DataFrame()
        self.recommendations_summary = (
            recommendations_summary if recommendations_summary is not None else pd.DataFrame()
        )
        self.info = info or {}


def _earnings_estimate_df(curr_yr_avg: float = 28.50, n_analysts: int = 45) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numberOfAnalysts": [n_analysts, n_analysts, n_analysts, n_analysts],
            "avg": [3.20, 3.50, curr_yr_avg, 32.10],
            "low": [2.80, 3.10, curr_yr_avg - 2, 28.0],
            "high": [3.50, 3.80, curr_yr_avg + 2, 36.0],
            "yearAgoEps": [2.85, 3.10, 25.0, curr_yr_avg],
            "growth": [0.12, 0.13, 0.14, 0.13],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _revenue_estimate_df(
    curr_yr_avg: float = 200_000_000_000,
    year_ago: float = 130_000_000_000,
    next_yr_avg: float = 260_000_000_000,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numberOfAnalysts": [45, 45, 45, 45],
            "avg": [60_000_000_000, 65_000_000_000, curr_yr_avg, next_yr_avg],
            "low": [55_000_000_000, 60_000_000_000, 195_000_000_000, 250_000_000_000],
            "high": [65_000_000_000, 70_000_000_000, 210_000_000_000, 270_000_000_000],
            "yearAgoRevenue": [40_000_000_000, 45_000_000_000, year_ago, curr_yr_avg],
            "growth": [0.5, 0.44, 0.54, 0.30],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _eps_revisions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "upLast7days": [1, 0, 2, 1],
            "upLast30days": [3, 2, 5, 4],
            "downLast30days": [1, 1, 1, 0],
            "upLast60days": [4, 3, 7, 5],
            "downLast60days": [2, 2, 1, 1],
            "upLast90days": [5, 4, 9, 7],
            "downLast90days": [3, 2, 2, 2],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _recommendations_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [25, 23, 22, 20],
            "buy": [15, 16, 16, 18],
            "hold": [5, 6, 7, 7],
            "sell": [0, 0, 0, 0],
            "strongSell": [0, 0, 0, 0],
        }
    )


# ---------------------------------------------------------------------------
# Schema-extraction unit tests (mock yfinance directly via _populate_*)
# ---------------------------------------------------------------------------


def test_populate_estimates_extracts_eps_and_revenue() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(retrieval_date="2026-05-16", ticker="NVDA")
    yt = _FakeYTicker(
        earnings_estimate=_earnings_estimate_df(),
        revenue_estimate=_revenue_estimate_df(),
    )
    fetcher._populate_estimates(snapshot, yt)
    assert snapshot.eps_estimate_curr_yr == pytest.approx(28.50)
    assert snapshot.eps_estimate_curr_qtr == pytest.approx(3.20)
    assert snapshot.eps_estimate_next_yr == pytest.approx(32.10)
    assert snapshot.revenue_estimate_curr_yr == 200_000_000_000
    assert snapshot.revenue_estimate_next_yr == 260_000_000_000
    assert snapshot.n_analysts == 45
    # growth: (200B / 130B) - 1 ≈ 0.5384
    assert snapshot.revenue_growth_curr_yr == pytest.approx(200 / 130 - 1, abs=1e-6)
    # next-yr: (260 / 200) - 1 = 0.30
    assert snapshot.revenue_growth_next_yr == pytest.approx(0.30)


def test_populate_estimates_handles_empty_dataframes() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(retrieval_date="2026-05-16", ticker="NVDA")
    yt = _FakeYTicker()  # all empty
    fetcher._populate_estimates(snapshot, yt)
    # No fields populated, no exception raised
    assert snapshot.eps_estimate_curr_yr is None
    assert snapshot.revenue_estimate_curr_yr is None


def test_populate_revisions_computes_net() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(retrieval_date="2026-05-16", ticker="NVDA")
    yt = _FakeYTicker(eps_revisions=_eps_revisions_df())
    fetcher._populate_revisions(snapshot, yt)
    # 0y row: up30=5, down30=1 → net=4
    assert snapshot.eps_revision_30d == pytest.approx(4.0)
    assert snapshot.eps_revision_60d == pytest.approx(6.0)
    assert snapshot.eps_revision_90d == pytest.approx(7.0)


def test_populate_recommendation_from_summary() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(retrieval_date="2026-05-16", ticker="NVDA")
    yt = _FakeYTicker(recommendations_summary=_recommendations_summary_df())
    fetcher._populate_recommendation(snapshot, yt)
    # 0m: 25 SB, 15 B, 5 H = total 45; mean = (25 + 30 + 15) / 45 = 70/45 = 1.555...
    assert snapshot.recommendation_mean == pytest.approx(70 / 45, abs=1e-3)
    # -1m: 23 SB, 16 B, 6 H = total 45; mean = (23 + 32 + 18)/45 = 73/45
    # delta = current - old = 70/45 - 73/45 = -3/45
    assert snapshot.recommendation_delta_30d == pytest.approx(-3 / 45, abs=1e-3)


def test_populate_recommendation_falls_back_to_info() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(retrieval_date="2026-05-16", ticker="NVDA")
    yt = _FakeYTicker(info={"recommendationMean": 1.8})
    fetcher._populate_recommendation(snapshot, yt)
    assert snapshot.recommendation_mean == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# fetch_snapshot integration with fake yfinance via monkeypatch
# ---------------------------------------------------------------------------


def test_fetch_snapshot_succeeds_with_fake_yf(monkeypatch, tmp_path: Path) -> None:
    cfg = EngineConfig(force_refresh=True)
    cfg.raw_dir = tmp_path
    fetcher = AnalystEstimateFetcher(cfg)

    def _fake_ticker(symbol: str) -> _FakeYTicker:
        return _FakeYTicker(
            earnings_estimate=_earnings_estimate_df(),
            revenue_estimate=_revenue_estimate_df(),
            eps_revisions=_eps_revisions_df(),
            recommendations_summary=_recommendations_summary_df(),
        )

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _fake_ticker)

    snapshot = fetcher.fetch_snapshot(ticker="NVDA", retrieval_date="2026-05-16")
    assert snapshot.fetch_succeeded is True
    assert snapshot.eps_estimate_curr_yr == pytest.approx(28.50)
    assert snapshot.recommendation_mean is not None
    # Cache file should now exist
    cache = tmp_path / "analyst_estimates_20260516.json"
    assert cache.exists()


def test_fetch_snapshot_returns_failure_marker_on_exception(monkeypatch, tmp_path: Path) -> None:
    cfg = EngineConfig(force_refresh=True)
    cfg.raw_dir = tmp_path
    fetcher = AnalystEstimateFetcher(cfg)

    def _broken_ticker(symbol: str):
        raise ConnectionError("yfinance unreachable")

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _broken_ticker)

    snapshot = fetcher.fetch_snapshot(ticker="NVDA", retrieval_date="2026-05-16")
    assert snapshot.fetch_succeeded is False
    assert snapshot.fetch_error is not None
    assert "ConnectionError" in snapshot.fetch_error


def test_fetch_snapshot_uses_cache(monkeypatch, tmp_path: Path) -> None:
    cfg = EngineConfig(force_refresh=False)
    cfg.raw_dir = tmp_path
    fetcher = AnalystEstimateFetcher(cfg)

    # Pre-populate cache
    cached = AnalystSnapshot(
        retrieval_date="2026-05-16",
        ticker="NVDA",
        eps_estimate_curr_yr=99.99,
        fetch_succeeded=True,
    )
    cache_path = tmp_path / "analyst_estimates_20260516.json"
    with open(cache_path, "w") as f:
        json.dump(cached.to_dict(), f)

    # Monkeypatch yfinance to detect if it's called (should NOT be)
    called = {"yf": False}

    def _watch_ticker(symbol: str):
        called["yf"] = True
        return _FakeYTicker()

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _watch_ticker)

    snapshot = fetcher.fetch_snapshot(ticker="NVDA", retrieval_date="2026-05-16")
    assert snapshot.eps_estimate_curr_yr == 99.99
    assert called["yf"] is False, "Cache should have been used; yfinance not called"


# ---------------------------------------------------------------------------
# render_overlay_summary
# ---------------------------------------------------------------------------


def test_overlay_unavailable_when_snapshot_failed() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(retrieval_date="2026-05-16", ticker="NVDA", fetch_succeeded=False)
    overlay = fetcher.render_overlay_summary(snapshot, ml_prediction=0.40)
    assert overlay.direction == "unavailable"
    assert overlay.magnitude == "unavailable"


def test_overlay_unavailable_when_ml_prediction_none() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(
        retrieval_date="2026-05-16",
        ticker="NVDA",
        revenue_growth_curr_yr=0.40,
        fetch_succeeded=True,
    )
    overlay = fetcher.render_overlay_summary(snapshot, ml_prediction=None)
    assert overlay.direction == "unavailable"


def test_overlay_concur_when_difference_small() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(
        retrieval_date="2026-05-16",
        ticker="NVDA",
        revenue_growth_next_yr=0.30,
        fetch_succeeded=True,
    )
    overlay = fetcher.render_overlay_summary(snapshot, ml_prediction=0.31)
    assert overlay.direction == "concur"
    assert overlay.magnitude == "small"
    assert overlay.delta_pp == pytest.approx(1.0, abs=1e-3)


def test_overlay_ml_above_moderate() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(
        retrieval_date="2026-05-16",
        ticker="NVDA",
        revenue_growth_next_yr=0.20,
        fetch_succeeded=True,
    )
    overlay = fetcher.render_overlay_summary(snapshot, ml_prediction=0.23)
    assert overlay.direction == "ml_above"
    assert overlay.magnitude == "moderate"


def test_overlay_ml_below_large() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(
        retrieval_date="2026-05-16",
        ticker="NVDA",
        revenue_growth_next_yr=0.45,
        fetch_succeeded=True,
    )
    overlay = fetcher.render_overlay_summary(snapshot, ml_prediction=0.30)
    assert overlay.direction == "ml_below"
    assert overlay.magnitude == "large"
    assert overlay.delta_pp == pytest.approx(-15.0, abs=0.1)


def test_overlay_uses_next_year_growth_preferentially() -> None:
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    snapshot = AnalystSnapshot(
        retrieval_date="2026-05-16",
        ticker="NVDA",
        revenue_growth_curr_yr=0.50,
        revenue_growth_next_yr=0.30,
        fetch_succeeded=True,
    )
    overlay = fetcher.render_overlay_summary(snapshot, ml_prediction=0.30)
    # uses next_yr (0.30), not curr_yr (0.50)
    assert overlay.analyst_growth == 0.30
    assert overlay.direction == "concur"


# ---------------------------------------------------------------------------
# Contract: NO to_features method
# ---------------------------------------------------------------------------


def test_fetcher_has_no_to_features_method() -> None:
    """Critical Req 2 contract. Also enforced by test_v2_spec_consistency.py."""
    cfg = EngineConfig()
    fetcher = AnalystEstimateFetcher(cfg)
    assert not hasattr(fetcher, "to_features"), (
        "AnalystEstimateFetcher must NEVER expose to_features. "
        "Analyst data is a decision-time overlay only (Req 2)."
    )


def test_overlay_object_is_descriptive_not_numeric_modifier() -> None:
    """The AnalystOverlay carries comparator metadata only. The
    public API never returns a 'price adjustment' from analyst data."""
    overlay = AnalystOverlay(
        ml_growth=0.30,
        analyst_growth=0.28,
        delta_pp=2.0,
        direction="ml_above",
        magnitude="moderate",
        n_analysts=45,
        recommendation_mean=1.8,
    )
    d = overlay.to_dict()
    # No "price_adjustment" or "weight" or "target_price" fields
    forbidden = {"price_adjustment", "weight", "target_price", "adjusted_target"}
    assert not (forbidden & set(d.keys()))
