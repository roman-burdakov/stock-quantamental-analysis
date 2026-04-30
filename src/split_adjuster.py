"""
NVDA Quantamental Engine — Split Adjustment Utility.

Applies stock-split adjustments to per-share metrics so that all values
are on a consistent post-split basis.  Raw filing values are never
overwritten; both ``raw_value`` and ``adjusted_value`` are stored.

Reqs: 3.1, 3.2
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from src.config import EngineConfig

logger = logging.getLogger(__name__)

# Per-share metrics that require split adjustment.
# - diluted_eps: divide by split ratio for pre-split facts
# - diluted_shares: multiply by split ratio for pre-split facts
PER_SHARE_METRICS = {"diluted_eps", "diluted_shares"}


class SplitAdjuster:
    """Apply stock-split adjustments to per-share financial facts.

    The adjuster reads split events from ``config.split_history`` and,
    for each per-share metric whose ``period_end`` falls before the split
    date, computes an ``adjusted_value`` on a post-split basis.

    Rules:
    - ``raw_value`` is always set to the original filing value and is
      **never** overwritten.
    - ``adjusted_value`` is the split-adjusted value.
    - ``adjustment_factor`` is the cumulative split ratio applied
      (e.g. 10.0 for a 10:1 split on pre-split facts, 1.0 for
      post-split facts).
    - ``adjustment_basis`` is ``"split_adjusted_to_current"`` for
      pre-split facts and ``"as_reported"`` for post-split facts.
    - ``validation_basis`` indicates which basis the published reference
      value uses for comparison.

    For **diluted_shares** (pre-split): adjusted = raw × ratio
    For **diluted_eps** (pre-split):    adjusted = raw / ratio
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self._splits = self._parse_split_history()

    def _parse_split_history(self) -> list[dict]:
        """Parse and validate split events from config.

        Returns a list of dicts with keys: ``date`` (:class:`date`),
        ``ratio`` (int/float), ``description`` (str).
        """
        splits: list[dict] = []
        for entry in self.config.split_history:
            try:
                split_date = date.fromisoformat(entry["date"])
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid split entry %s: %s", entry, exc)
                continue
            ratio = entry.get("ratio", 1)
            if ratio <= 1:
                logger.warning("Skipping split with ratio <= 1: %s", entry)
                continue
            splits.append({
                "date": split_date,
                "ratio": ratio,
                "description": entry.get("description", ""),
            })
        # Sort chronologically (earliest first)
        splits.sort(key=lambda s: s["date"])
        return splits

    def compute_adjustment_factor(self, period_end: Optional[str]) -> float:
        """Compute the cumulative split adjustment factor for a given period end.

        For each split event where ``period_end < split_date``, the
        factor is multiplied by the split ratio.  If ``period_end`` is
        on or after all split dates, the factor is 1.0 (no adjustment).

        Parameters
        ----------
        period_end:
            ISO date string for the fact's reporting period end.

        Returns
        -------
        float
            Cumulative adjustment factor (≥ 1.0).
        """
        if not period_end or not self._splits:
            return 1.0

        try:
            pe_date = date.fromisoformat(period_end)
        except (ValueError, TypeError):
            return 1.0

        factor = 1.0
        for split in self._splits:
            if pe_date < split["date"]:
                factor *= split["ratio"]
        return factor

    def adjust_value(
        self,
        metric_name: str,
        raw_value: Optional[float],
        adjustment_factor: float,
    ) -> Optional[float]:
        """Compute the split-adjusted value for a per-share metric.

        - **diluted_shares**: ``raw_value * adjustment_factor``
          (more shares after split)
        - **diluted_eps**: ``raw_value / adjustment_factor``
          (lower EPS after split)

        Non-per-share metrics or a factor of 1.0 return ``raw_value``
        unchanged.

        Parameters
        ----------
        metric_name:
            The metric identifier (e.g. ``"diluted_eps"``).
        raw_value:
            The original value from the filing.
        adjustment_factor:
            The cumulative split ratio from :meth:`compute_adjustment_factor`.

        Returns
        -------
        float | None
            The adjusted value, or ``None`` if ``raw_value`` is ``None``.
        """
        if raw_value is None:
            return None
        if adjustment_factor == 1.0:
            return raw_value
        if metric_name not in PER_SHARE_METRICS:
            return raw_value

        if metric_name == "diluted_shares":
            return raw_value * adjustment_factor
        elif metric_name == "diluted_eps":
            return raw_value / adjustment_factor
        return raw_value

    def apply_split_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply split adjustments to a parsed facts DataFrame.

        For every row:
        - ``raw_value`` is set to the original ``value`` (never overwritten).
        - For per-share metrics with ``period_end`` before a split date,
          ``adjusted_value`` is computed and ``adjustment_basis`` is set
          to ``"split_adjusted_to_current"``.
        - For post-split or non-per-share metrics, ``adjusted_value``
          equals ``raw_value`` and ``adjustment_basis`` is
          ``"as_reported"``.
        - ``validation_basis`` is set to indicate which basis the
          published reference uses.

        Parameters
        ----------
        df:
            DataFrame from :meth:`XBRLParser.parse_companyfacts`.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with split-adjustment columns populated.
        """
        if df.empty or not self._splits:
            # Even with no splits, ensure columns are populated
            if not df.empty:
                df = df.copy()
                df["raw_value"] = df["value"]
                df["adjusted_value"] = df["value"]
                df["adjustment_factor"] = 1.0
                df["adjustment_basis"] = "as_reported"
                df["validation_basis"] = "as_reported"
            return df

        df = df.copy()

        # Always preserve the raw value
        df["raw_value"] = df["value"]

        # Compute adjustment factor per row (based on period_end vs split dates)
        date_factors = df["period_end"].apply(self.compute_adjustment_factor)

        # Compute adjusted values — only per-share metrics get a non-1.0 factor
        adjusted_values: list[Optional[float]] = []
        actual_factors: list[float] = []
        adjustment_bases: list[str] = []
        validation_bases: list[str] = []

        for idx, row in df.iterrows():
            metric = row["metric_name"]
            raw = row["value"]
            date_factor = date_factors[idx]

            # Only per-share metrics get the split factor applied
            if metric in PER_SHARE_METRICS:
                factor = date_factor
            else:
                factor = 1.0

            adj = self.adjust_value(metric, raw, factor)
            adjusted_values.append(adj)
            actual_factors.append(factor)

            if metric in PER_SHARE_METRICS and factor != 1.0:
                adjustment_bases.append("split_adjusted_to_current")
                # Published references for pre-split years may be on
                # either basis.  The FY2025 10-K restates prior years
                # on post-split basis, so we note that.
                validation_bases.append("post_split_restated")
            else:
                adjustment_bases.append("as_reported")
                validation_bases.append("as_reported")

        df["adjusted_value"] = adjusted_values
        df["adjustment_factor"] = actual_factors
        df["adjustment_basis"] = adjustment_bases
        df["validation_basis"] = validation_bases

        # Log summary
        pre_split_count = (df["adjustment_factor"] != 1.0).sum()
        if pre_split_count > 0:
            logger.info(
                "Split adjustment applied to %d pre-split per-share facts",
                pre_split_count,
            )

        return df
