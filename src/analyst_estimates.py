"""
v2 Analyst Estimate Fetcher (Req 2).

Decision-time overlay only. Analyst estimates are NEVER ML features.

Why this module exists at all:
- The professor's feedback explicitly named "analyst-estimate revisions"
  as one of four feature groups. Ignoring this would lose grade points.
- yfinance only provides a current snapshot of analyst estimates, not
  historical revisions. Treating today's snapshot as a historical
  feature is mathematically degenerate (only one observation across
  all training rows = no usable variation) and statistically
  lookahead-by-construction (today's analyst data was not available
  to a 2018-era decision).
- The corrected design treats analyst data as a side-by-side comparator
  reported alongside the ML prediction. The ML model is trained only
  on fundamentals + market + NLP features. The analyst comparator
  informs the report narrative ("ML and analysts agree" or "ML diverges
  from analyst consensus by Xpp") but does NOT arithmetically modify
  the target price.

Design contract (enforced by tests/test_v2_spec_consistency.py):
- ``AnalystEstimateFetcher`` exposes ``fetch_snapshot()`` and
  ``render_overlay_summary()``, but **NEVER** a ``to_features()`` method.
- ``AnalystSnapshot`` is consumed by ``MLDecisionEngine.compare_to_analyst_overlay()``
  (Milestone 5) as a comparator, not by any ML training/prediction code.

Failure semantics:
- yfinance fetch failures are non-blocking. ``fetch_succeeded=False``
  in the snapshot lets downstream code skip the overlay without crashing
  the pipeline.

Caching:
- One snapshot per retrieval date. File:
  ``data/raw/analyst_estimates_<YYYYMMDD>.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from src.config import EngineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalystSnapshot:
    """Current analyst-consensus snapshot (decision-time overlay only).

    Field semantics:
    - ``revenue_growth_curr_yr`` is computed when prior-year revenue is
      available (current revenue estimate / prior revenue − 1). Otherwise
      it is None.
    - ``recommendation_mean`` follows the standard scale: 1.0 = Strong
      Buy, 5.0 = Sell. yfinance returns this directly.
    - ``recommendation_delta_30d`` is current mean minus 30-day-ago mean
      where available. yfinance does not always return this; None when
      missing.
    - ``fetch_succeeded=False`` indicates a network or schema error;
      downstream code should skip the overlay rather than crash.
    """

    retrieval_date: str  # ISO date "YYYY-MM-DD"
    ticker: str
    # Revenue / EPS estimates
    eps_estimate_curr_qtr: Optional[float] = None
    eps_estimate_curr_yr: Optional[float] = None
    eps_estimate_next_yr: Optional[float] = None
    revenue_estimate_curr_yr: Optional[float] = None
    revenue_estimate_next_yr: Optional[float] = None
    # Derived growth (where prior-year revenue is available)
    revenue_growth_curr_yr: Optional[float] = None
    revenue_growth_next_yr: Optional[float] = None
    # Revisions
    eps_revision_30d: Optional[float] = None
    eps_revision_60d: Optional[float] = None
    eps_revision_90d: Optional[float] = None
    # Sentiment / dispersion
    n_analysts: Optional[int] = None
    recommendation_mean: Optional[float] = None
    recommendation_delta_30d: Optional[float] = None
    # Fetch status
    fetch_succeeded: bool = False
    fetch_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalystOverlay:
    """Result of comparing the ML prediction to the analyst snapshot.

    This is what ``MLDecisionEngine.compare_to_analyst_overlay()`` (in
    Milestone 5) will consume. It is purely descriptive; no number here
    feeds the target-price math.
    """

    ml_growth: Optional[float]
    analyst_growth: Optional[float]
    delta_pp: Optional[float]  # ML - analyst, percentage points
    direction: str  # "ml_above" | "ml_below" | "concur" | "unavailable"
    magnitude: str  # "large" (>5pp) | "moderate" (2-5pp) | "small" (<2pp) | "unavailable"
    n_analysts: Optional[int]
    recommendation_mean: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class AnalystEstimateFetcher:
    """Fetch yfinance analyst-consensus snapshot (Req 2.1–2.7).

    Public API:
    - ``fetch_snapshot(ticker, retrieval_date)``  → ``AnalystSnapshot``
    - ``render_overlay_summary(snapshot, ml_prediction)`` → ``AnalystOverlay``

    There is intentionally NO ``to_features()`` method. Adding one would
    invite reintroduction of the rejected analyst-augmented-ML design;
    the consistency test in ``tests/test_v2_spec_consistency.py`` will
    fail any commit that adds it.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, retrieval_date: str) -> Path:
        date_compact = retrieval_date.replace("-", "")
        return self.config.raw_dir / f"analyst_estimates_{date_compact}.json"

    def _load_cached(self, path: Path) -> Optional[AnalystSnapshot]:
        if not path.exists() or not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return AnalystSnapshot(**payload)
        except (json.JSONDecodeError, TypeError, OSError) as e:
            logger.warning("Failed to load cached analyst snapshot %s: %s", path, e)
            return None

    def _save_cache(self, snapshot: AnalystSnapshot, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_snapshot(
        self,
        ticker: Optional[str] = None,
        retrieval_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> AnalystSnapshot:
        """Fetch (or load cached) analyst snapshot.

        Parameters
        ----------
        ticker : str, optional
            Defaults to ``config.ticker``.
        retrieval_date : str, optional
            ISO date "YYYY-MM-DD". Defaults to today's UTC date. Only
            controls the cache key: the actual data returned by yfinance
            is always "current as of now" since yfinance has no
            historical retrieval support.
        use_cache : bool
            If True, load from cache when available.
        """
        ticker = ticker or self.config.ticker
        retrieval_date = retrieval_date or date.today().isoformat()
        cache_path = self._cache_path(retrieval_date)

        if use_cache and not self.config.force_refresh:
            cached = self._load_cached(cache_path)
            if cached is not None:
                logger.info("Using cached analyst snapshot: %s", cache_path)
                return cached

        snapshot = AnalystSnapshot(retrieval_date=retrieval_date, ticker=ticker)
        try:
            import yfinance as yf  # local import to keep module-level light
        except ImportError as e:
            snapshot.fetch_error = f"yfinance not installed: {e}"
            logger.warning("yfinance not available; analyst overlay disabled.")
            return snapshot

        try:
            yt = yf.Ticker(ticker)
            self._populate_estimates(snapshot, yt)
            self._populate_revisions(snapshot, yt)
            self._populate_recommendation(snapshot, yt)
            snapshot.fetch_succeeded = (
                snapshot.eps_estimate_curr_yr is not None
                or snapshot.revenue_estimate_curr_yr is not None
                or snapshot.recommendation_mean is not None
            )
            if not snapshot.fetch_succeeded and snapshot.fetch_error is None:
                snapshot.fetch_error = "yfinance returned no usable analyst data"
        except Exception as exc:  # noqa: BLE001
            # Defensive: yfinance schema changes have historically broken
            # callers. We never crash the pipeline on this overlay.
            snapshot.fetch_error = f"{type(exc).__name__}: {exc}"
            snapshot.fetch_succeeded = False
            logger.warning("Analyst snapshot fetch failed: %s", snapshot.fetch_error)

        try:
            self._save_cache(snapshot, cache_path)
        except OSError as e:
            logger.warning("Failed to cache analyst snapshot: %s", e)

        return snapshot

    # ------------------------------------------------------------------
    # yfinance schema-specific extractors (defensive)
    # ------------------------------------------------------------------

    def _populate_estimates(self, snapshot: AnalystSnapshot, yt) -> None:
        """Pull EPS / revenue estimates. Tolerates schema variations."""
        # yfinance.Ticker.earnings_estimate is a DataFrame indexed by
        # period codes ("0q", "+1q", "0y", "+1y") with columns including
        # numberOfAnalysts, avg, low, high, yearAgoEps, growth.
        try:
            ee = getattr(yt, "earnings_estimate", None)
            if ee is not None and not ee.empty:
                snapshot.eps_estimate_curr_qtr = self._safe_get(ee, "0q", "avg")
                snapshot.eps_estimate_curr_yr = self._safe_get(ee, "0y", "avg")
                snapshot.eps_estimate_next_yr = self._safe_get(ee, "+1y", "avg")
                # n_analysts: prefer current-year row
                n = self._safe_get(ee, "0y", "numberOfAnalysts")
                if n is not None:
                    try:
                        snapshot.n_analysts = int(n)
                    except (TypeError, ValueError):
                        pass
        except Exception as e:  # noqa: BLE001
            logger.debug("earnings_estimate extraction failed: %s", e)

        # yfinance.Ticker.revenue_estimate has the same schema with an
        # additional yearAgoRevenue column for growth derivation.
        try:
            re_df = getattr(yt, "revenue_estimate", None)
            if re_df is not None and not re_df.empty:
                snapshot.revenue_estimate_curr_yr = self._safe_get(re_df, "0y", "avg")
                snapshot.revenue_estimate_next_yr = self._safe_get(re_df, "+1y", "avg")
                # Derive growth using yearAgoRevenue when available
                ya_curr = self._safe_get(re_df, "0y", "yearAgoRevenue")
                if (
                    snapshot.revenue_estimate_curr_yr is not None
                    and ya_curr
                    and ya_curr != 0
                ):
                    snapshot.revenue_growth_curr_yr = float(
                        snapshot.revenue_estimate_curr_yr
                    ) / float(ya_curr) - 1.0
                # next-yr growth = next_yr / curr_yr − 1 if curr_yr available
                if (
                    snapshot.revenue_estimate_next_yr is not None
                    and snapshot.revenue_estimate_curr_yr is not None
                    and float(snapshot.revenue_estimate_curr_yr) != 0.0
                ):
                    snapshot.revenue_growth_next_yr = float(
                        snapshot.revenue_estimate_next_yr
                    ) / float(snapshot.revenue_estimate_curr_yr) - 1.0
        except Exception as e:  # noqa: BLE001
            logger.debug("revenue_estimate extraction failed: %s", e)

    def _populate_revisions(self, snapshot: AnalystSnapshot, yt) -> None:
        """Pull EPS revision counts. yfinance schema is sparse; we record
        net revisions (up-down) for current-year row when present."""
        try:
            er = getattr(yt, "eps_revisions", None)
            if er is None or er.empty:
                return
            # Net revisions = upLast{N}days - downLast{N}days for "0y"
            for days, attr in (
                (30, "eps_revision_30d"),
                (60, "eps_revision_60d"),
                (90, "eps_revision_90d"),
            ):
                up = self._safe_get(er, "0y", f"upLast{days}days")
                down = self._safe_get(er, "0y", f"downLast{days}days")
                if up is not None or down is not None:
                    net = (up or 0) - (down or 0)
                    setattr(snapshot, attr, float(net))
        except Exception as e:  # noqa: BLE001
            logger.debug("eps_revisions extraction failed: %s", e)

    def _populate_recommendation(self, snapshot: AnalystSnapshot, yt) -> None:
        """Compute mean recommendation 1-5 from buy/hold/sell counts."""
        try:
            rs = getattr(yt, "recommendations_summary", None)
            if rs is None or rs.empty:
                # Fall back to .info["recommendationMean"]
                info = getattr(yt, "info", None) or {}
                rm = info.get("recommendationMean")
                if rm is not None:
                    try:
                        snapshot.recommendation_mean = float(rm)
                    except (TypeError, ValueError):
                        pass
                return

            # recommendations_summary columns: strongBuy, buy, hold, sell, strongSell
            # period column ('0m', '-1m', '-2m', '-3m'). Use 0m for current.
            row = self._select_period_row(rs, "0m")
            if row is None:
                return
            sb = float(row.get("strongBuy", 0) or 0)
            b = float(row.get("buy", 0) or 0)
            h = float(row.get("hold", 0) or 0)
            s = float(row.get("sell", 0) or 0)
            ss = float(row.get("strongSell", 0) or 0)
            total = sb + b + h + s + ss
            if total > 0:
                # Score: SB=1, B=2, H=3, S=4, SS=5
                snapshot.recommendation_mean = (
                    1 * sb + 2 * b + 3 * h + 4 * s + 5 * ss
                ) / total
                # If we have an older period row, compute 30-day delta
                row_old = self._select_period_row(rs, "-1m")
                if row_old is not None:
                    sb_o = float(row_old.get("strongBuy", 0) or 0)
                    b_o = float(row_old.get("buy", 0) or 0)
                    h_o = float(row_old.get("hold", 0) or 0)
                    s_o = float(row_old.get("sell", 0) or 0)
                    ss_o = float(row_old.get("strongSell", 0) or 0)
                    total_o = sb_o + b_o + h_o + s_o + ss_o
                    if total_o > 0:
                        old_mean = (
                            1 * sb_o + 2 * b_o + 3 * h_o + 4 * s_o + 5 * ss_o
                        ) / total_o
                        snapshot.recommendation_delta_30d = (
                            snapshot.recommendation_mean - old_mean
                        )
        except Exception as e:  # noqa: BLE001
            logger.debug("recommendations_summary extraction failed: %s", e)

    # ------------------------------------------------------------------
    # Helper accessors for yfinance DataFrames
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_get(df, period_label: str, col: str):
        """Get df.loc[period_label, col] safely. Tolerates missing
        index labels and missing columns. Coerces numpy scalars
        (int64/float64) to Python primitives so the snapshot is
        JSON-serializable.
        """
        val = None
        try:
            if period_label in df.index and col in df.columns:
                val = df.at[period_label, col]
        except Exception:  # noqa: BLE001
            val = None
        if val is None:
            # Some yfinance versions use a "period" column instead of index
            try:
                if "period" in df.columns and col in df.columns:
                    row = df[df["period"] == period_label]
                    if not row.empty:
                        val = row.iloc[0][col]
            except Exception:  # noqa: BLE001
                pass
        if val is None:
            return None
        if _isnan(val):
            return None
        # Coerce numpy scalars to Python primitives.
        try:
            if hasattr(val, "item"):
                return val.item()
        except Exception:  # noqa: BLE001
            pass
        return val

    @staticmethod
    def _select_period_row(df, period_label: str):
        """Pick a row by period label, supporting both index and column
        layouts. Coerces numpy scalars to Python primitives in the
        returned dict."""
        row_dict = None
        try:
            if period_label in df.index:
                row = df.loc[period_label]
                if hasattr(row, "to_dict"):
                    row_dict = row.to_dict()
        except Exception:  # noqa: BLE001
            pass
        if row_dict is None:
            try:
                if "period" in df.columns:
                    match = df[df["period"] == period_label]
                    if not match.empty:
                        row_dict = match.iloc[0].to_dict()
            except Exception:  # noqa: BLE001
                pass
        if row_dict is None:
            return None
        coerced: dict[str, Any] = {}
        for k, v in row_dict.items():
            if v is None or _isnan(v):
                coerced[k] = None
            elif hasattr(v, "item"):
                try:
                    coerced[k] = v.item()
                except Exception:  # noqa: BLE001
                    coerced[k] = v
            else:
                coerced[k] = v
        return coerced

    # ------------------------------------------------------------------
    # Overlay / comparator (NOT a feature transformation)
    # ------------------------------------------------------------------

    def render_overlay_summary(
        self,
        snapshot: AnalystSnapshot,
        ml_prediction: Optional[float],
    ) -> AnalystOverlay:
        """Compare the ML revenue-growth prediction to analyst consensus.

        Returns a descriptive ``AnalystOverlay`` for the report
        comparator. **No number returned here ever modifies a target
        price.** The ML adjustment math operates on ML predictions only.
        """
        analyst = (
            snapshot.revenue_growth_next_yr
            if snapshot.revenue_growth_next_yr is not None
            else snapshot.revenue_growth_curr_yr
        )
        # Guard against None AND NaN — a NaN from yfinance would silently
        # propagate through arithmetic and produce nonsensical output
        # (delta_pp=NaN, direction='ml_below' from sign comparison).
        ml_invalid = ml_prediction is None or _isnan(ml_prediction)
        analyst_invalid = analyst is None or _isnan(analyst)
        if ml_invalid or analyst_invalid or not snapshot.fetch_succeeded:
            return AnalystOverlay(
                ml_growth=None if ml_invalid else ml_prediction,
                analyst_growth=None if analyst_invalid else analyst,
                delta_pp=None,
                direction="unavailable",
                magnitude="unavailable",
                n_analysts=snapshot.n_analysts,
                recommendation_mean=snapshot.recommendation_mean,
            )

        delta = (ml_prediction - analyst) * 100.0  # in percentage points
        if abs(delta) < 2.0:
            magnitude = "small"
            direction = "concur"
        elif abs(delta) < 5.0:
            magnitude = "moderate"
            direction = "ml_above" if delta > 0 else "ml_below"
        else:
            magnitude = "large"
            direction = "ml_above" if delta > 0 else "ml_below"
        return AnalystOverlay(
            ml_growth=ml_prediction,
            analyst_growth=analyst,
            delta_pp=delta,
            direction=direction,
            magnitude=magnitude,
            n_analysts=snapshot.n_analysts,
            recommendation_mean=snapshot.recommendation_mean,
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _isnan(value: Any) -> bool:
    try:
        import math

        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False
