"""
Tests for src/market_features.py — Task 3.2 (Reqs 4.2-4.5).

The most important test here is ``test_no_lookahead_invariant``:
appending future-dated prices to the input must NOT change the output
for an as_of_date in the past. This is the strongest no-lookahead
guarantee available short of formal verification.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import math

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig
from src.market_features import (
    MarketFeatureBuilder,
    _isnan,
    _to_date,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _daily_prices(
    ticker: str,
    start: date,
    end: date,
    start_price: float = 100.0,
    daily_drift: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a daily-prices DataFrame with deterministic random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    # Geometric random walk
    returns = rng.normal(loc=daily_drift, scale=0.02, size=len(dates))
    prices = start_price * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "adj_close": prices,
            "source_available_date": dates.strftime("%Y-%m-%d"),
        }
    )


def _multi_ticker_prices(
    tickers: Iterable[str],
    start: date,
    end: date,
    seed_offset: int = 0,
) -> pd.DataFrame:
    frames = [
        _daily_prices(t, start, end, start_price=100.0 + 10.0 * i,
                      seed=42 + i + seed_offset)
        for i, t in enumerate(tickers)
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_to_date_handles_string() -> None:
    assert _to_date("2026-05-16") == date(2026, 5, 16)


def test_to_date_handles_datetime() -> None:
    from datetime import datetime
    assert _to_date(datetime(2026, 5, 16, 12, 0)) == date(2026, 5, 16)


def test_to_date_handles_pandas_timestamp() -> None:
    assert _to_date(pd.Timestamp("2026-05-16")) == date(2026, 5, 16)


def test_to_date_returns_none_on_invalid() -> None:
    assert _to_date("not a date") is None
    assert _to_date(None) is None


def test_isnan_helper() -> None:
    assert _isnan(float("nan")) is True
    assert _isnan(np.nan) is True
    assert _isnan(0.0) is False
    assert _isnan(1.5) is False
    assert _isnan(None) is False


# ---------------------------------------------------------------------------
# Builder unit tests
# ---------------------------------------------------------------------------


def test_compute_features_empty_prices_returns_empty_df() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    out = builder.compute_features(
        pd.DataFrame(columns=["date", "ticker", "adj_close"]),
        as_of_dates=["2026-05-16"],
    )
    # One row with as_of_date but all features None
    assert len(out) == 1
    assert out.iloc[0]["nvda_return_3m"] is None
    assert out.iloc[0]["nvda_return_12m"] is None


def test_compute_features_no_as_of_dates_returns_empty_df() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    prices = _daily_prices("NVDA", date(2024, 1, 1), date(2026, 1, 1))
    out = builder.compute_features(prices, as_of_dates=[])
    assert len(out) == 0


def test_trailing_return_basic() -> None:
    """Trailing return should equal end_price/start_price - 1."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    # Construct prices with known endpoints: 100 → 200 over 12 months
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-30", "2026-04-30"]),
            "ticker": ["NVDA", "NVDA"],
            "adj_close": [100.0, 200.0],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    # 200/100 - 1 = 1.0
    assert out.iloc[0]["nvda_return_12m"] == pytest.approx(1.0, abs=1e-6)


def test_excess_return_uses_sox() -> None:
    """Excess return should equal NVDA - SOX returns."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2025-04-30", "2026-04-30",
                "2025-04-30", "2026-04-30",
            ]),
            "ticker": ["NVDA", "NVDA", "^SOX", "^SOX"],
            "adj_close": [100.0, 200.0, 100.0, 150.0],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    # NVDA: +100%; SOX: +50%; excess = 50%
    assert out.iloc[0]["nvda_excess_return_12m_vs_sox"] == pytest.approx(0.5, abs=1e-6)


def test_excess_return_none_when_benchmark_missing() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = _daily_prices("NVDA", date(2024, 1, 1), date(2026, 5, 1))
    out = builder.compute_features(df, as_of_dates=["2026-05-01"])
    # SOX absent → excess return None
    assert out.iloc[0]["nvda_excess_return_12m_vs_sox"] is None


def test_volatility_returns_positive_value() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = _daily_prices("NVDA", date(2024, 1, 1), date(2026, 5, 1))
    out = builder.compute_features(df, as_of_dates=["2026-05-01"])
    vol = out.iloc[0]["nvda_volatility_60d"]
    assert vol is not None
    assert vol > 0
    # Annualized vol of a sigma=0.02 daily walk is ~0.32
    assert 0.1 < vol < 0.6


def test_volatility_none_with_insufficient_history() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    # Only 3 days of data
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-28", "2026-04-29", "2026-04-30"]),
            "ticker": ["NVDA"] * 3,
            "adj_close": [100.0, 101.0, 102.0],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    assert out.iloc[0]["nvda_volatility_60d"] is None


def test_beta_with_correlated_series() -> None:
    """Beta with NVDA = 2 * SOX should equal ~2.0."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)

    # Build SOX random walk; NVDA = SOX with 2x return
    rng = np.random.default_rng(1)
    dates = pd.bdate_range(start="2024-01-01", end="2026-05-01")
    sox_returns = rng.normal(0.0005, 0.012, len(dates))
    sox_prices = 100.0 * np.exp(np.cumsum(sox_returns))
    nvda_prices = 100.0 * np.exp(np.cumsum(2.0 * sox_returns))
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["NVDA"] * len(dates) + ["^SOX"] * len(dates),
        "adj_close": list(nvda_prices) + list(sox_prices),
    })
    out = builder.compute_features(df, as_of_dates=["2026-05-01"])
    beta = out.iloc[0]["nvda_beta_252d_vs_sox"]
    assert beta is not None
    # Should be close to 2.0
    assert 1.7 < beta < 2.3


def test_beta_none_when_insufficient_overlap() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    # Only 10 days of data
    df = pd.DataFrame(
        {
            "date": list(pd.date_range("2026-04-15", "2026-04-29")) * 2,
            "ticker": ["NVDA"] * 15 + ["^SOX"] * 15,
            "adj_close": [100.0 + i for i in range(15)] + [100.0 + i*0.5 for i in range(15)],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-29"])
    assert out.iloc[0]["nvda_beta_252d_vs_sox"] is None


def test_peer_spread_excludes_excluded_tickers() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = _multi_ticker_prices(
        ["NVDA", "AMD", "AVGO", "INTC"],
        start=date(2024, 1, 1),
        end=date(2026, 5, 1),
    )
    out_full = builder.compute_features(df, as_of_dates=["2026-05-01"])
    out_excluded = builder.compute_features(
        df, as_of_dates=["2026-05-01"], excluded_tickers=["INTC"]
    )
    # Excluded peer has no spread column
    assert "nvda_minus_AMD_return_12m" in out_full.columns
    assert "nvda_minus_INTC_return_12m" in out_full.columns
    assert "nvda_minus_AMD_return_12m" in out_excluded.columns
    assert "nvda_minus_INTC_return_12m" not in out_excluded.columns


def test_returns_use_only_prices_at_or_before_as_of() -> None:
    """If we add prices AFTER as_of, the feature row must not change."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)

    # Build a price series ending at 2025-12-31
    df = _daily_prices("NVDA", date(2024, 1, 1), date(2025, 12, 31))
    feats_pre = builder.compute_features(df, as_of_dates=["2025-12-31"])

    # Append future prices (2026-01 through 2026-05 with very different walk)
    future = _daily_prices(
        "NVDA", date(2026, 1, 1), date(2026, 5, 1),
        start_price=10000.0, daily_drift=0.05, seed=999,
    )
    df_with_future = pd.concat([df, future], ignore_index=True)
    feats_post = builder.compute_features(df_with_future, as_of_dates=["2025-12-31"])

    pd.testing.assert_frame_equal(feats_pre, feats_post)


# ---------------------------------------------------------------------------
# THE no-lookahead invariant test (Req 4.3 — Task 3.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_of_str", [
    "2018-12-31",
    "2020-06-30",
    "2022-09-30",
    "2024-12-31",
    "2025-12-31",
])
def test_no_lookahead_invariant_at_multiple_dates(as_of_str: str) -> None:
    """For multiple as_of_dates, verify that truncating prices to
    ``date <= as_of_date`` produces identical features to using the
    full price history. This is the strongest no-lookahead invariant."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = _multi_ticker_prices(
        ["NVDA", "^SOX", "^GSPC", "AMD", "AVGO"],
        start=date(2017, 1, 1),
        end=date(2026, 5, 1),
        seed_offset=7,
    )
    feats_full = builder.compute_features(df, as_of_dates=[as_of_str])
    df_trunc = df[pd.to_datetime(df["date"]) <= pd.Timestamp(as_of_str)]
    feats_trunc = builder.compute_features(df_trunc, as_of_dates=[as_of_str])
    pd.testing.assert_frame_equal(feats_full, feats_trunc)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compute_features_with_invalid_as_of_skips_silently() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = _daily_prices("NVDA", date(2024, 1, 1), date(2026, 5, 1))
    out = builder.compute_features(df, as_of_dates=["not a date", "2026-05-01"])
    # Invalid date is skipped; valid date produces row
    assert len(out) == 1
    assert pd.Timestamp(out.iloc[0]["as_of_date"]) == pd.Timestamp("2026-05-01")


def test_compute_features_handles_duplicate_ticker_dates() -> None:
    """If the price table has duplicate (ticker, date) rows, the
    builder must keep one and not crash."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-30", "2025-04-30", "2026-04-30"]),
            "ticker": ["NVDA", "NVDA", "NVDA"],
            "adj_close": [100.0, 99.0, 200.0],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    # Duplicates handled (keep last → 99.0 used as start)
    assert out.iloc[0]["nvda_return_12m"] is not None


def test_returns_handle_zero_start_price_safely() -> None:
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-30", "2026-04-30"]),
            "ticker": ["NVDA", "NVDA"],
            "adj_close": [0.0, 100.0],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    # Division by zero must produce None, not inf
    assert out.iloc[0]["nvda_return_12m"] is None


def test_returns_handle_nan_prices() -> None:
    """NaN adj_close values are dropped at indexing time. The trailing
    return then uses the last *non-NaN* price. With only one valid
    observation, start and end coincide and the return is 0.0."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-30", "2026-04-30"]),
            "ticker": ["NVDA", "NVDA"],
            "adj_close": [100.0, float("nan")],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    # NaN was dropped; only the 2025-04-30 price (100) survives. Both
    # start and end of the 12-month window resolve to that single price.
    assert out.iloc[0]["nvda_return_12m"] == pytest.approx(0.0, abs=1e-9)


def test_returns_none_when_all_prices_nan() -> None:
    """When every price is NaN, the ticker has no usable observations
    and trailing return must be None."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-30", "2026-04-30"]),
            "ticker": ["NVDA", "NVDA"],
            "adj_close": [float("nan"), float("nan")],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    assert out.iloc[0]["nvda_return_12m"] is None


def test_compute_features_accepts_index_aliases_for_sox() -> None:
    """If ^SOX appears as 'SOX' or '.SOX' in the cache, it should still
    resolve to the SOX benchmark."""
    cfg = EngineConfig()
    builder = MarketFeatureBuilder(cfg)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2025-04-30", "2026-04-30",
                "2025-04-30", "2026-04-30",
            ]),
            "ticker": ["NVDA", "NVDA", "SOX", "SOX"],  # alias without ^
            "adj_close": [100.0, 200.0, 100.0, 150.0],
        }
    )
    out = builder.compute_features(df, as_of_dates=["2026-04-30"])
    # Excess vs SOX should be 50% (NVDA +100% minus SOX +50%)
    assert out.iloc[0]["nvda_excess_return_12m_vs_sox"] == pytest.approx(0.5, abs=1e-6)
