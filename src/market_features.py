"""
v2 Market Feature Builder (Req 4.2-4.5, Task 3.1).

Computes price-derived ML market and peer-relative features from
``data/raw/market_prices.csv``. Every feature is strictly point-in-time:
for any ``as_of_date``, the feature uses only prices with
``date <= as_of_date``.

Why price-derived only (not statement-derived):
- Historical peer fundamentals (P/E, EV/Sales) require point-in-time
  XBRL companyfacts for every peer at every quarter. That's a sizable
  data-acquisition project beyond v2 scope.
- Price-derived features (return spreads, vol, beta) need only the
  daily-prices table that v1 already caches, plus the ^SOX/^GSPC
  indices added in Task 1.1.
- This restriction is documented in Req 4.4 and limitations.md.

Output columns (per as_of_date):
- ``nvda_return_3m``, ``nvda_return_12m``
- ``nvda_excess_return_3m_vs_sox``, ``nvda_excess_return_12m_vs_sox``
- ``nvda_excess_return_3m_vs_spx``, ``nvda_excess_return_12m_vs_spx``
- ``nvda_volatility_60d`` (annualized stdev of daily log returns)
- ``nvda_beta_252d_vs_sox`` (rolling-OLS slope)
- ``nvda_minus_<peer>_return_12m`` for each non-excluded peer

All windows are calendar-day windows (3m = 90 days, 12m = 365 days,
60d = 60 days, 252d = 252 days). Only trading-day prices populate the
windows; missing days are skipped via the existing LOCF in market_prices.

No-lookahead invariant:
    feats_full = builder.compute_features(prices_full, [as_of])
    prices_truncated = prices_full[prices_full.date <= as_of]
    feats_truncated = builder.compute_features(prices_truncated, [as_of])
    assert feats_full.equals(feats_truncated)

This is enforced by ``tests/test_market_features.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.config import EngineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window sizes in calendar days.
RETURN_WINDOW_3M_DAYS = 90
RETURN_WINDOW_12M_DAYS = 365
VOL_WINDOW_DAYS = 60
BETA_WINDOW_DAYS = 252

# Trading-day equivalents (used for rolling-window minimum sample sizes).
TRADING_DAYS_3M = 60
TRADING_DAYS_12M = 250
TRADING_DAYS_60D = 40
TRADING_DAYS_252D = 200

# Index tickers expected to be present in market_prices.csv.
SOX_TICKER = "^SOX"
SPX_TICKER = "^GSPC"

# yfinance returns ^SOX as a tradable symbol but some pipelines may have
# fetched it as ".SOX" or "SOX" historically. We accept any of these for
# robustness; the canonical name written by EdgarFetcher is ``^SOX``.
# IMPORTANT: ordered tuple, NOT set. Set iteration is hash-randomized
# in Python (PYTHONHASHSEED), which would produce non-deterministic
# alias resolution when multiple aliases are present in the cache.
# We always prefer the canonical name.
SOX_ALIASES = ("^SOX", ".SOX", "SOX")
SPX_ALIASES = ("^GSPC", ".GSPC", "GSPC", "^SPX", "SPX")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MarketFeatureRow:
    """One row of market features anchored on as_of_date.

    All fields are nullable: any missing input price (insufficient
    history, ticker excluded by freshness gate, NaN in source) yields
    None for that field. Downstream ML code should treat None as a
    feature gap and apply imputation/exclusion per its own policy.
    """

    as_of_date: date
    # NVDA-only return features
    nvda_return_3m: Optional[float] = None
    nvda_return_12m: Optional[float] = None
    # Excess returns vs benchmarks
    nvda_excess_return_3m_vs_sox: Optional[float] = None
    nvda_excess_return_12m_vs_sox: Optional[float] = None
    nvda_excess_return_3m_vs_spx: Optional[float] = None
    nvda_excess_return_12m_vs_spx: Optional[float] = None
    # Volatility / beta
    nvda_volatility_60d: Optional[float] = None
    nvda_beta_252d_vs_sox: Optional[float] = None
    # Per-peer return spreads (populated dynamically; serialized via
    # MarketFeatureBuilder.compute_features which adds extra columns)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class MarketFeatureBuilder:
    """Compute v2 ML market features from cached daily prices."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(
        self,
        prices: pd.DataFrame,
        as_of_dates: Iterable,
        excluded_tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compute market features for each ``as_of_date``.

        Parameters
        ----------
        prices
            Wide-or-long market-prices DataFrame. Required columns:
            ``date``, ``ticker``, ``adj_close``. Output of
            ``EdgarFetcher.fetch_market_prices()``.
        as_of_dates
            Iterable of date / datetime / str. One feature row per date.
            Each row uses only prices ≤ that date.
        excluded_tickers
            Tickers to skip when computing peer-spread columns. Typically
            from ``MarketPriceFreshnessReport.tickers_excluded_stale``.

        Returns
        -------
        pd.DataFrame
            One row per as_of_date. Columns include the
            ``MarketFeatureRow`` fields plus a ``nvda_minus_<peer>_return_12m``
            column for each non-excluded peer. Missing inputs produce
            ``None`` (not NaN) so the schema is JSON-friendly.
        """
        excluded = set(excluded_tickers or [])
        nvda = self.config.ticker
        peer_tickers = [
            p for p in self.config.core_semiconductor_peers
            if p not in excluded
        ]

        # Pre-build per-ticker price Series indexed by date (sorted ascending).
        price_by_ticker = self._index_prices_by_ticker(prices)

        nvda_prices = price_by_ticker.get(nvda)
        sox_prices = self._first_match(price_by_ticker, SOX_ALIASES)
        spx_prices = self._first_match(price_by_ticker, SPX_ALIASES)

        if nvda_prices is None or nvda_prices.empty:
            logger.warning(
                "No NVDA prices in market_prices.csv; market features unavailable."
            )

        rows: list[dict] = []
        for raw in as_of_dates:
            as_of = _to_date(raw)
            if as_of is None:
                continue
            row = self._compute_single_row(
                as_of=as_of,
                nvda_prices=nvda_prices,
                sox_prices=sox_prices,
                spx_prices=spx_prices,
                peer_prices={p: price_by_ticker.get(p) for p in peer_tickers},
            )
            rows.append(row)
        df = pd.DataFrame(rows)
        # Ensure deterministic column order even with no rows.
        return df

    # ------------------------------------------------------------------
    # Internal: per-row
    # ------------------------------------------------------------------

    def _compute_single_row(
        self,
        as_of: date,
        nvda_prices: Optional[pd.Series],
        sox_prices: Optional[pd.Series],
        spx_prices: Optional[pd.Series],
        peer_prices: dict[str, Optional[pd.Series]],
    ) -> dict:
        row: dict[str, object] = {"as_of_date": as_of}

        # --- Trailing returns (NVDA only) ---
        nvda_3m = self._trailing_return(nvda_prices, as_of, RETURN_WINDOW_3M_DAYS)
        nvda_12m = self._trailing_return(nvda_prices, as_of, RETURN_WINDOW_12M_DAYS)
        row["nvda_return_3m"] = nvda_3m
        row["nvda_return_12m"] = nvda_12m

        # --- Excess returns vs benchmarks ---
        sox_3m = self._trailing_return(sox_prices, as_of, RETURN_WINDOW_3M_DAYS)
        sox_12m = self._trailing_return(sox_prices, as_of, RETURN_WINDOW_12M_DAYS)
        spx_3m = self._trailing_return(spx_prices, as_of, RETURN_WINDOW_3M_DAYS)
        spx_12m = self._trailing_return(spx_prices, as_of, RETURN_WINDOW_12M_DAYS)
        row["nvda_excess_return_3m_vs_sox"] = self._safe_diff(nvda_3m, sox_3m)
        row["nvda_excess_return_12m_vs_sox"] = self._safe_diff(nvda_12m, sox_12m)
        row["nvda_excess_return_3m_vs_spx"] = self._safe_diff(nvda_3m, spx_3m)
        row["nvda_excess_return_12m_vs_spx"] = self._safe_diff(nvda_12m, spx_12m)

        # --- Volatility (annualized stdev of daily log returns) ---
        row["nvda_volatility_60d"] = self._annualized_volatility(
            nvda_prices, as_of, VOL_WINDOW_DAYS
        )

        # --- Rolling beta (NVDA vs SOX) ---
        row["nvda_beta_252d_vs_sox"] = self._rolling_beta(
            nvda_prices, sox_prices, as_of, BETA_WINDOW_DAYS
        )

        # --- Per-peer 12m return spread ---
        for peer, ps in peer_prices.items():
            peer_12m = self._trailing_return(ps, as_of, RETURN_WINDOW_12M_DAYS)
            row[f"nvda_minus_{peer}_return_12m"] = self._safe_diff(nvda_12m, peer_12m)

        return row

    # ------------------------------------------------------------------
    # Internal: feature primitives (all use prices <= as_of only)
    # ------------------------------------------------------------------

    @staticmethod
    def _trailing_return(
        prices: Optional[pd.Series],
        as_of: date,
        window_days: int,
    ) -> Optional[float]:
        """Total return from (as_of - window_days) to as_of.

        Uses the LAST available price ≤ as_of as the end value, and the
        LAST available price ≤ (as_of - window_days) as the start value.
        Returns None if either anchor is missing or zero.
        """
        if prices is None or prices.empty:
            return None
        end_ts = pd.Timestamp(as_of)
        start_ts = end_ts - pd.Timedelta(days=window_days)
        # Filter to prices <= as_of (strict no-lookahead).
        relevant = prices.loc[:end_ts]
        if relevant.empty:
            return None
        end_price = float(relevant.iloc[-1])
        # Start price = last price at or before start_ts.
        start_window = relevant.loc[:start_ts]
        if start_window.empty:
            return None
        start_price = float(start_window.iloc[-1])
        if start_price == 0 or _isnan(start_price) or _isnan(end_price):
            return None
        return round(end_price / start_price - 1.0, 6)

    @staticmethod
    def _annualized_volatility(
        prices: Optional[pd.Series],
        as_of: date,
        window_days: int,
    ) -> Optional[float]:
        """Annualized stdev of daily log returns over the trailing window.

        Annualization factor: sqrt(252). Requires at least 10 trading-day
        observations in the window; otherwise returns None.
        """
        if prices is None or prices.empty:
            return None
        end_ts = pd.Timestamp(as_of)
        start_ts = end_ts - pd.Timedelta(days=window_days)
        window = prices.loc[start_ts:end_ts]
        if len(window) < 10:
            return None
        # Log returns
        log_returns = np.log(window.values[1:] / window.values[:-1])
        log_returns = log_returns[~np.isnan(log_returns)]
        if len(log_returns) < 5:
            return None
        sigma = float(np.std(log_returns, ddof=1))
        if _isnan(sigma):
            return None
        return round(sigma * math.sqrt(252.0), 6)

    @staticmethod
    def _rolling_beta(
        nvda_prices: Optional[pd.Series],
        bench_prices: Optional[pd.Series],
        as_of: date,
        window_days: int,
    ) -> Optional[float]:
        """OLS slope of NVDA daily returns on benchmark daily returns
        over the trailing window. All data ≤ as_of.

        Beta = cov(NVDA, BENCH) / var(BENCH). Returns None if either
        series has insufficient overlap (< 30 observations) or var=0.
        """
        if (
            nvda_prices is None or nvda_prices.empty
            or bench_prices is None or bench_prices.empty
        ):
            return None
        end_ts = pd.Timestamp(as_of)
        start_ts = end_ts - pd.Timedelta(days=window_days)
        nvda_window = nvda_prices.loc[start_ts:end_ts]
        bench_window = bench_prices.loc[start_ts:end_ts]
        # Align on common dates.
        joined = pd.concat([nvda_window, bench_window], axis=1).dropna()
        if len(joined) < 30:
            return None
        nvda_returns = joined.iloc[:, 0].pct_change().dropna()
        bench_returns = joined.iloc[:, 1].pct_change().dropna()
        # Re-align after pct_change.
        common_idx = nvda_returns.index.intersection(bench_returns.index)
        if len(common_idx) < 30:
            return None
        nvda_returns = nvda_returns.loc[common_idx]
        bench_returns = bench_returns.loc[common_idx]
        bench_var = float(np.var(bench_returns, ddof=1))
        if bench_var == 0 or _isnan(bench_var):
            return None
        cov = float(np.cov(nvda_returns, bench_returns, ddof=1)[0, 1])
        if _isnan(cov):
            return None
        return round(cov / bench_var, 6)

    # ------------------------------------------------------------------
    # Internal: data shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _index_prices_by_ticker(prices: pd.DataFrame) -> dict[str, pd.Series]:
        """Return {ticker: Series indexed by date, values=adj_close}."""
        if prices is None or prices.empty:
            return {}
        required = {"date", "ticker", "adj_close"}
        if not required.issubset(set(prices.columns)):
            logger.warning(
                "market_prices missing required columns; expected %s, got %s",
                required, set(prices.columns),
            )
            return {}
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.dropna(subset=["adj_close"])
        df = df.sort_values(["ticker", "date"])
        out: dict[str, pd.Series] = {}
        for t, group in df.groupby("ticker"):
            s = pd.Series(
                group["adj_close"].astype(float).values,
                index=group["date"],
            )
            # Drop duplicate dates (keep last).
            s = s[~s.index.duplicated(keep="last")]
            out[str(t)] = s
        return out

    @staticmethod
    def _first_match(
        price_by_ticker: dict[str, pd.Series], aliases: tuple[str, ...]
    ) -> Optional[pd.Series]:
        """Return the first matching ticker series from a tuple of aliases.

        The tuple's order matters: canonical names should appear first so
        resolution is deterministic when multiple aliases are present.
        """
        for alias in aliases:
            if alias in price_by_ticker:
                return price_by_ticker[alias]
        return None

    @staticmethod
    def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """Return a - b, or None if either is missing."""
        if a is None or b is None:
            return None
        if _isnan(a) or _isnan(b):
            return None
        return round(a - b, 6)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _to_date(value) -> Optional[date]:
    """Best-effort parse to ``date``."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    s = str(value)[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _isnan(value) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False
