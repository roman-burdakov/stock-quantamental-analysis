"""
v2 Freshness Gates (Req 5).

Two independent freshness gates:

1. ``MarketPriceFreshnessGate`` — used by the v2 ML stage. Checks the
   most recent date for each required ticker in ``market_prices.csv``
   against the report date (in trading days). If NVDA itself is stale,
   the ML stage is blocked (data validation gate equivalent). Per-ticker
   exclusion is supported for non-NVDA tickers.

2. ``PeerFinancialsFreshnessGate`` — used by the v1 valuation stage
   for peer comparable multiples. Checks ``peer_financials.csv`` rows
   against the report date (in calendar days). Per-peer exclusion of
   stale rows; if ≥3 of 5 core peers are excluded, the peer-multiples
   exhibit is suppressed (formalising existing v1 behaviour).

Why two gates: ML peer/market-relative features are price-derived (return
spreads, volatility) and depend only on price freshness. Valuation
multiples (P/E, EV/Sales) are financial-statement-derived. The previous
single-gate design conflated the two; this module is the v2 fix.

Both gates have:
- ``evaluate_and_refresh()`` — the policy method that performs the
  staleness check and triggers a refresh via the EdgarFetcher when
  appropriate.
- A typed report dataclass capturing the outcome.
- Best-effort behaviour: a refresh failure does NOT crash the pipeline;
  it is logged and the gate falls back to the existing-cache state.

Usage from ``run_pipeline.py``:

    market_freshness_gate = MarketPriceFreshnessGate(config)
    market_freshness_report = market_freshness_gate.evaluate_and_refresh(
        state.market_prices, config.report_date, fetcher,
        required_tickers=[config.ticker, "^SOX", "^GSPC", *config.core_semiconductor_peers],
    )
    if market_freshness_report.status == "blocked_nvda_stale":
        # v2 ml stage cannot proceed
        ...

    peer_fin_gate = PeerFinancialsFreshnessGate(config)
    peer_fin_report = peer_fin_gate.evaluate_and_refresh(
        state.peer_financials, config.report_date, fetcher,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import EngineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketPriceFreshnessReport:
    """Outcome of the market-price freshness gate (Req 5.1–5.5)."""

    threshold_trading_days: int
    report_date: str
    tickers_checked: list[str]
    tickers_refreshed: list[str]
    tickers_excluded_stale: list[str]
    nvda_stale: bool
    per_ticker_age_days: dict[str, int]
    # status ∈ {"fresh", "refreshed", "partial", "blocked_nvda_stale", "no_data"}
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PeerFinancialsFreshnessReport:
    """Outcome of the peer-financials freshness gate (Req 5.6–5.10)."""

    threshold_days: int
    report_date: str
    peers_checked: list[str]
    peers_refreshed: list[str]
    peers_excluded_stale: list[str]
    per_peer_age_days: dict[str, int]
    # status ∈ {"fresh", "refreshed", "partial", "stale_blocked", "no_data"}
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(value: Any) -> Optional[date]:
    """Best-effort parse to a ``date`` from str/Timestamp/date inputs."""
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


def _trading_days_between(start: date, end: date) -> int:
    """Approximate trading-day count between two dates, accounting for
    weekends AND US Federal market holidays (NYSE/Nasdaq closures).

    Uses pandas USFederalHolidayCalendar via CustomBusinessDay, which
    captures: New Year's Day, MLK Day, Presidents Day, Good Friday is
    NOT a federal holiday but IS a market holiday — pandas does not
    cover it; this is a documented residual error.

    Returns 0 if start >= end.
    """
    if start >= end:
        return 0
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay

        cal = USFederalHolidayCalendar()
        bday = CustomBusinessDay(calendar=cal)
        rng = pd.date_range(start=start, end=end, freq=bday)
    except (ImportError, AttributeError):
        # Fall back to plain bdate_range if calendar import fails.
        rng = pd.bdate_range(start=start, end=end)
    # Subtract 1 because the range is inclusive on both ends; we want
    # the number of trading days *between* start and end (exclusive of
    # start, inclusive of end).
    return max(0, len(rng) - 1)


def _calendar_days_between(start: date, end: date) -> int:
    if start > end:
        return 0
    return (end - start).days


# ---------------------------------------------------------------------------
# Market-price freshness gate (for ML)
# ---------------------------------------------------------------------------

class MarketPriceFreshnessGate:
    """Gates ML market/peer-return features (Req 5.1–5.5).

    For every required ticker, computes the age (in trading days) between
    the latest cached price and the report date. If any ticker's age
    exceeds the configured threshold (default 5 trading days), the gate
    triggers a refresh via the supplied ``EdgarFetcher``. After refresh,
    per-ticker staleness is re-evaluated and stale tickers are recorded
    in ``tickers_excluded_stale``.

    Special case: if NVDA itself is stale even after refresh, the entire
    ML stage is blocked (status == "blocked_nvda_stale"). This is a hard
    gate — without fresh NVDA prices, return-based features are
    unreliable.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.threshold_trading_days = (
            config.mlconfig.market_price_max_staleness_trading_days
        )

    def _max_date_per_ticker(
        self, prices: pd.DataFrame
    ) -> dict[str, date]:
        """Return the latest available date for each ticker."""
        if prices.empty or "date" not in prices.columns or "ticker" not in prices.columns:
            return {}
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        latest = df.groupby("ticker")["date"].max()
        return {str(t): _parse_date(d) for t, d in latest.items() if _parse_date(d)}

    def _ages(
        self,
        prices: pd.DataFrame,
        report_date_obj: date,
        required_tickers: list[str],
    ) -> dict[str, int]:
        """Return per-ticker age in trading days vs report_date.

        Tickers absent from the price data get age = sentinel 99999.
        """
        latest = self._max_date_per_ticker(prices)
        ages: dict[str, int] = {}
        for t in required_tickers:
            d = latest.get(t)
            if d is None:
                ages[t] = 99999  # sentinel for missing
            else:
                ages[t] = _trading_days_between(d, report_date_obj)
        return ages

    def evaluate_and_refresh(
        self,
        market_prices: pd.DataFrame,
        report_date: str,
        edgar_fetcher,
        required_tickers: list[str],
    ) -> MarketPriceFreshnessReport:
        """Check freshness, refresh if needed, return outcome.

        Parameters
        ----------
        market_prices : pd.DataFrame
            Current cached prices. Columns: date, ticker, adj_close,
            source_available_date.
        report_date : str
            ISO date "YYYY-MM-DD" — the analysis cutoff.
        edgar_fetcher : EdgarFetcher
            Used to refresh prices if any ticker is stale. Must expose
            ``fetch_market_prices(tickers, start, end)``.
        required_tickers : list[str]
            All tickers that must be fresh for ML features. Typically:
            [NVDA, ^SOX, ^GSPC, *core_semiconductor_peers].
        """
        nvda = self.config.ticker
        report_date_obj = _parse_date(report_date) or date.today()
        threshold = self.threshold_trading_days

        if market_prices is None or market_prices.empty:
            logger.warning("Market prices empty; cannot evaluate freshness.")
            return MarketPriceFreshnessReport(
                threshold_trading_days=threshold,
                report_date=report_date,
                tickers_checked=list(required_tickers),
                tickers_refreshed=[],
                tickers_excluded_stale=list(required_tickers),
                nvda_stale=True,
                per_ticker_age_days={t: 99999 for t in required_tickers},
                status="no_data",
            )

        ages = self._ages(market_prices, report_date_obj, required_tickers)
        stale_tickers = [t for t, age in ages.items() if age > threshold]
        refreshed: list[str] = []

        if stale_tickers:
            logger.info(
                "MarketPriceFreshnessGate: %d/%d tickers stale (>%d trading days). "
                "Triggering refresh: %s",
                len(stale_tickers),
                len(required_tickers),
                threshold,
                stale_tickers,
            )
            try:
                # Force-refresh ALL required tickers (fetch_market_prices is
                # all-or-nothing per cache file). Use force_refresh on the
                # fetcher to bypass the cache.
                original_force = self.config.force_refresh
                self.config.force_refresh = True
                start = f"{self.config.start_fiscal_year}-01-01"
                try:
                    refreshed_prices = edgar_fetcher.fetch_market_prices(
                        list(required_tickers),
                        start=start,
                        end=report_date,
                    )
                finally:
                    # Always restore force_refresh, even if fetch raised.
                    self.config.force_refresh = original_force
                if refreshed_prices is not None and not refreshed_prices.empty:
                    market_prices = refreshed_prices
                    refreshed = list(required_tickers)
                else:
                    logger.warning("Refresh returned empty DataFrame.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Market-price refresh failed: %s", exc)

            # Re-evaluate ages after refresh
            ages = self._ages(market_prices, report_date_obj, required_tickers)
            stale_tickers = [t for t, age in ages.items() if age > threshold]

        nvda_stale = ages.get(nvda, 99999) > threshold

        if nvda_stale:
            status = "blocked_nvda_stale"
        elif not stale_tickers:
            status = "refreshed" if refreshed else "fresh"
        else:
            status = "partial"

        return MarketPriceFreshnessReport(
            threshold_trading_days=threshold,
            report_date=report_date,
            tickers_checked=list(required_tickers),
            tickers_refreshed=refreshed,
            tickers_excluded_stale=stale_tickers,
            nvda_stale=nvda_stale,
            per_ticker_age_days=ages,
            status=status,
        )

    def write_report(
        self,
        report: MarketPriceFreshnessReport,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Persist the freshness report as JSON for the audit step."""
        if output_path is None:
            output_path = self.config.outputs_dir / "market_price_freshness_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return output_path


# ---------------------------------------------------------------------------
# Peer-financials freshness gate (for valuation comps only)
# ---------------------------------------------------------------------------

class PeerFinancialsFreshnessGate:
    """Gates valuation peer multiples (Req 5.6–5.10).

    Per-peer staleness threshold is in calendar days (default 30, per
    MLConfig). Stale peers are excluded from peer-multiple comparisons.
    If ≥3 of 5 core peers are excluded, the peer-multiples exhibit is
    flagged ``stale_blocked`` and the report consumer should suppress it.

    This formalises the v1 informal staleness behaviour (90-day threshold
    in EngineConfig.peer_staleness_threshold_days). The v2 default of 30
    days is tighter; v1 90-day behaviour can be restored by setting
    ``MLConfig.peer_financials_max_staleness_days = 90``.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.threshold_days = (
            config.mlconfig.peer_financials_max_staleness_days
        )

    def _ages(
        self,
        peer_financials: pd.DataFrame,
        report_date_obj: date,
    ) -> dict[str, int]:
        if peer_financials is None or peer_financials.empty:
            return {}
        if "ticker" not in peer_financials.columns:
            return {}
        date_col = (
            "source_available_date"
            if "source_available_date" in peer_financials.columns
            else "source_date"
            if "source_date" in peer_financials.columns
            else None
        )
        if date_col is None:
            return {}
        ages: dict[str, int] = {}
        for _, row in peer_financials.iterrows():
            ticker = str(row["ticker"])
            d = _parse_date(row[date_col])
            if d is None:
                ages[ticker] = 99999
            else:
                ages[ticker] = _calendar_days_between(d, report_date_obj)
        return ages

    def evaluate_and_refresh(
        self,
        peer_financials: pd.DataFrame,
        report_date: str,
        edgar_fetcher,
    ) -> PeerFinancialsFreshnessReport:
        report_date_obj = _parse_date(report_date) or date.today()
        threshold = self.threshold_days
        core = list(self.config.core_semiconductor_peers)
        peers_checked = list(
            self.config.core_semiconductor_peers
            + self.config.infrastructure_peers
            + self.config.ai_capex_context
        )

        if peer_financials is None or peer_financials.empty:
            logger.warning("peer_financials empty; cannot evaluate freshness.")
            return PeerFinancialsFreshnessReport(
                threshold_days=threshold,
                report_date=report_date,
                peers_checked=peers_checked,
                peers_refreshed=[],
                peers_excluded_stale=peers_checked,
                per_peer_age_days={p: 99999 for p in peers_checked},
                status="no_data",
            )

        ages = self._ages(peer_financials, report_date_obj)
        stale = [p for p, age in ages.items() if age > threshold]
        refreshed: list[str] = []

        if stale:
            logger.info(
                "PeerFinancialsFreshnessGate: %d peers stale (>%d days). "
                "Triggering refresh: %s",
                len(stale),
                threshold,
                stale,
            )
            try:
                original_force = self.config.force_refresh
                self.config.force_refresh = True
                try:
                    refreshed_df = edgar_fetcher.fetch_peer_financials(peers_checked)
                finally:
                    # Always restore force_refresh, even if fetch raised.
                    self.config.force_refresh = original_force
                if refreshed_df is not None and not refreshed_df.empty:
                    peer_financials = refreshed_df
                    refreshed = peers_checked
            except Exception as exc:  # noqa: BLE001
                logger.warning("Peer-financials refresh failed: %s", exc)

            ages = self._ages(peer_financials, report_date_obj)
            stale = [p for p, age in ages.items() if age > threshold]

        excluded = sorted(stale)
        # Block exhibit if ≥3 of 5 core peers excluded.
        core_excluded = [p for p in excluded if p in core]
        if len(core_excluded) >= 3:
            status = "stale_blocked"
        elif not excluded:
            status = "refreshed" if refreshed else "fresh"
        else:
            status = "partial"

        return PeerFinancialsFreshnessReport(
            threshold_days=threshold,
            report_date=report_date,
            peers_checked=peers_checked,
            peers_refreshed=refreshed,
            peers_excluded_stale=excluded,
            per_peer_age_days=ages,
            status=status,
        )

    def write_report(
        self,
        report: PeerFinancialsFreshnessReport,
        output_path: Optional[Path] = None,
    ) -> Path:
        if output_path is None:
            output_path = (
                self.config.outputs_dir / "peer_financials_freshness_report.json"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return output_path
