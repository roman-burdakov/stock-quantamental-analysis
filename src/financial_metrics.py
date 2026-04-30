"""
NVDA Quantamental Engine — Financial Metrics Calculator.

Computes ratios, growth rates, TTM, FCF, inflection points from
parsed XBRL data. Outputs ComputedMetric rows with source_available_date.

Requirements: 6.1–6.7
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ComputedMetric, EngineConfig

logger = logging.getLogger(__name__)


class FinancialMetricsCalculator:
    """Compute financial ratios, growth rates, TTM, and inflection flags."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all_metrics(self, xbrl_data: pd.DataFrame) -> pd.DataFrame:
        """Compute per-fiscal-period metrics from parsed XBRL data.

        Parameters
        ----------
        xbrl_data:
            Output of ``XBRLParser.parse_companyfacts()`` with columns:
            ticker, fiscal_period, fiscal_year, filing_date,
            source_available_date, accession_number, metric_name, value,
            unit, form_type.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, fiscal_period, filing_date,
            source_available_date, metric_name, metric_value,
            source_accession, unit.
        """
        if xbrl_data.empty:
            logger.warning("Empty xbrl_data passed to compute_all_metrics")
            return self._empty_output()

        results: list[dict] = []

        # Group by fiscal period (fiscal_year + fiscal_period)
        groups = xbrl_data.groupby(["fiscal_year", "fiscal_period"])

        for (fy, fp), group in groups:
            period_label = f"FY{fy}" if fp == "FY" else f"FY{fy}-{fp}"
            lookup = self._build_lookup(group)

            # Derive filing_date and source_available_date from the group
            filing_date = group["filing_date"].iloc[0]
            source_available_date = group["source_available_date"].iloc[0]
            accession = group["accession_number"].iloc[0]

            # --- Core line items ---
            revenue = lookup.get("revenue")
            cogs = lookup.get("cogs")
            gross_profit = lookup.get("gross_profit")
            operating_income = lookup.get("operating_income")
            net_income = lookup.get("net_income")
            total_assets = lookup.get("total_assets")
            shareholders_equity = lookup.get("shareholders_equity")
            operating_cash_flow = lookup.get("operating_cash_flow")
            capex = lookup.get("capex")
            r_and_d = lookup.get("r_and_d")
            sga = lookup.get("sga")
            current_assets = lookup.get("current_assets")
            current_liabilities = lookup.get("current_liabilities")
            total_debt = lookup.get("total_debt")
            diluted_eps = lookup.get("diluted_eps")

            base = dict(
                ticker=self.config.ticker,
                fiscal_period=period_label,
                filing_date=filing_date,
                source_available_date=source_available_date,
                source_accession=accession,
            )

            # --- Margin ratios ---
            results.append(self._metric(
                base, "gross_margin",
                self._safe_div(gross_profit, revenue), "ratio",
            ))
            results.append(self._metric(
                base, "operating_margin",
                self._safe_div(operating_income, revenue), "ratio",
            ))
            results.append(self._metric(
                base, "net_margin",
                self._safe_div(net_income, revenue), "ratio",
            ))

            # --- Return ratios ---
            results.append(self._metric(
                base, "ROE",
                self._safe_div(net_income, shareholders_equity), "ratio",
            ))
            results.append(self._metric(
                base, "ROA",
                self._safe_div(net_income, total_assets), "ratio",
            ))

            # --- FCF ---
            fcf = self._safe_sub(operating_cash_flow, capex)
            results.append(self._metric(base, "FCF", fcf, "USD"))
            results.append(self._metric(
                base, "FCF_margin",
                self._safe_div(fcf, revenue), "ratio",
            ))

            # --- Leverage / liquidity ---
            results.append(self._metric(
                base, "debt_to_equity",
                self._safe_div(total_debt, shareholders_equity), "ratio",
            ))
            results.append(self._metric(
                base, "current_ratio",
                self._safe_div(current_assets, current_liabilities), "ratio",
            ))

            # --- Expense ratios ---
            results.append(self._metric(
                base, "R&D_%_revenue",
                self._safe_div(r_and_d, revenue), "ratio",
            ))
            results.append(self._metric(
                base, "SG&A_%_revenue",
                self._safe_div(sga, revenue), "ratio",
            ))

            # --- Per-share ---
            results.append(self._metric(
                base, "diluted_EPS", diluted_eps, "USD/shares",
            ))

            # --- Raw line items (needed by pipeline for valuation inputs) ---
            results.append(self._metric(base, "revenue", revenue, "USD"))
            results.append(self._metric(base, "net_income", net_income, "USD"))
            results.append(self._metric(base, "operating_income", operating_income, "USD"))
            results.append(self._metric(base, "operating_cash_flow", operating_cash_flow, "USD"))
            results.append(self._metric(base, "capex", capex, "USD"))
            results.append(self._metric(base, "cash_and_securities", lookup.get("cash_and_securities"), "USD"))
            results.append(self._metric(base, "total_debt", total_debt, "USD"))
            results.append(self._metric(base, "long_term_debt", lookup.get("long_term_debt"), "USD"))
            results.append(self._metric(base, "short_term_debt", lookup.get("short_term_debt"), "USD"))
            results.append(self._metric(base, "diluted_shares", lookup.get("diluted_shares"), "shares"))

        metrics_df = pd.DataFrame(results)

        # --- YoY growth (requires multiple periods) ---
        growth_rows = self._compute_yoy_growth(xbrl_data, metrics_df)
        if growth_rows:
            metrics_df = pd.concat(
                [metrics_df, pd.DataFrame(growth_rows)], ignore_index=True,
            )

        # --- Segment mix (percentage of revenue per segment) ---
        segment_rows = self._compute_segment_mix(xbrl_data)
        if segment_rows:
            metrics_df = pd.concat(
                [metrics_df, pd.DataFrame(segment_rows)], ignore_index=True,
            )

        # Save to CSV
        output_path = Path(self.config.processed_dir) / "nvda_metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
        logger.info("Saved metrics to %s (%d rows)", output_path, len(metrics_df))

        return metrics_df

    def compute_ttm(
        self, quarterly: pd.DataFrame, metric: str,
    ) -> pd.DataFrame:
        """Compute trailing-twelve-month values from quarterly data.

        Parameters
        ----------
        quarterly:
            DataFrame with columns: fiscal_period, fiscal_year,
            metric_name, value (or metric_value). Must contain quarterly
            rows (fiscal_period != "FY").
        metric:
            The metric_name to aggregate (e.g. "revenue").

        Returns
        -------
        pd.DataFrame
            Columns: fiscal_period, ttm_value, periods_used.
        """
        # Determine value column
        val_col = "metric_value" if "metric_value" in quarterly.columns else "value"

        # Filter to the target metric and quarterly rows only
        mask = quarterly["metric_name"] == metric
        if "fiscal_period" in quarterly.columns:
            # Exclude annual-only rows (exact "FY" prefix without quarter suffix)
            # Keep rows like "FY2025-Q1" but exclude "FY2025" (annual)
            q_mask = quarterly["fiscal_period"].str.contains(r"Q\d", na=False, regex=True)
            mask = mask & q_mask
        df = quarterly.loc[mask].copy()

        if df.empty:
            logger.warning("No quarterly data for TTM computation of '%s'", metric)
            return pd.DataFrame(columns=["fiscal_period", "ttm_value", "periods_used"])

        # Sort chronologically
        df = df.sort_values(["fiscal_year", "fiscal_period"]).reset_index(drop=True)

        results: list[dict] = []
        for i in range(len(df)):
            # Take up to 4 trailing quarters ending at index i
            start = max(0, i - 3)
            window = df.iloc[start: i + 1]
            n_periods = len(window)
            ttm_val = window[val_col].sum() if n_periods == 4 else None
            if ttm_val is not None and n_periods < 4:
                ttm_val = None  # Need exactly 4 quarters
            results.append({
                "fiscal_period": df.iloc[i]["fiscal_period"],
                "ttm_value": ttm_val,
                "periods_used": n_periods,
            })

        return pd.DataFrame(results)

    def flag_inflection_points(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Flag metrics where YoY change exceeds 2σ from historical mean.

        Parameters
        ----------
        metrics:
            Output of ``compute_all_metrics`` with columns:
            fiscal_period, metric_name, metric_value.

        Returns
        -------
        pd.DataFrame
            Same as input with added ``inflection_flag`` column (bool).
        """
        if metrics.empty:
            return metrics.assign(inflection_flag=False)

        df = metrics.copy()
        df["inflection_flag"] = False

        # Only flag ratio/growth metrics (not absolute USD values)
        flaggable = df["unit"].isin(["ratio", "ratio_yoy"])

        for metric_name, grp in df.loc[flaggable].groupby("metric_name"):
            if len(grp) < 3:
                continue  # Need enough history for meaningful σ

            values = grp["metric_value"].dropna()
            if len(values) < 3:
                continue

            mean = values.mean()
            std = values.std()
            if std == 0 or np.isnan(std):
                continue

            threshold = 2 * std
            for idx in grp.index:
                val = df.loc[idx, "metric_value"]
                if val is not None and not np.isnan(val):
                    if abs(val - mean) > threshold:
                        df.loc[idx, "inflection_flag"] = True

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty DataFrame with the ComputedMetric schema."""
        return pd.DataFrame(columns=[
            "ticker", "fiscal_period", "filing_date",
            "source_available_date", "metric_name", "metric_value",
            "source_accession", "unit",
        ])

    @staticmethod
    def _build_lookup(group: pd.DataFrame) -> dict[str, Optional[float]]:
        """Build metric_name → value dict from a period group."""
        lookup: dict[str, Optional[float]] = {}
        for _, row in group.iterrows():
            name = row["metric_name"]
            val = row["value"]
            if name not in lookup or (val is not None and lookup[name] is None):
                lookup[name] = val
        return lookup

    @staticmethod
    def _safe_div(
        numerator: Optional[float], denominator: Optional[float],
    ) -> Optional[float]:
        """Divide with null propagation; log on missing input."""
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            logger.warning("Division by zero avoided")
            return None
        return numerator / denominator

    @staticmethod
    def _safe_sub(
        a: Optional[float], b: Optional[float],
    ) -> Optional[float]:
        """Subtract with null propagation."""
        if a is None or b is None:
            return None
        return a - b

    @staticmethod
    def _metric(
        base: dict, name: str, value: Optional[float], unit: str,
    ) -> dict:
        """Build a single metric row dict."""
        if value is None:
            logger.info("Metric '%s' is null for period %s", name, base.get("fiscal_period"))
        return {**base, "metric_name": name, "metric_value": value, "unit": unit}

    def _compute_yoy_growth(
        self, xbrl_data: pd.DataFrame, metrics_df: pd.DataFrame,
    ) -> list[dict]:
        """Compute revenue_growth_YoY and OI_growth_YoY across annual periods."""
        growth_rows: list[dict] = []

        for raw_metric, growth_name in [
            ("revenue", "revenue_growth_YoY"),
            ("operating_income", "OI_growth_YoY"),
        ]:
            # Use annual (FY) data only
            annual = xbrl_data[
                (xbrl_data["metric_name"] == raw_metric)
                & (xbrl_data["fiscal_period"] == "FY")
            ].sort_values("fiscal_year").reset_index(drop=True)

            if len(annual) < 2:
                continue

            for i in range(1, len(annual)):
                curr = annual.iloc[i]
                prev = annual.iloc[i - 1]
                curr_val = curr["value"]
                prev_val = prev["value"]

                growth = self._safe_div(
                    self._safe_sub(curr_val, prev_val), prev_val,
                )

                period_label = f"FY{curr['fiscal_year']}"
                growth_rows.append({
                    "ticker": self.config.ticker,
                    "fiscal_period": period_label,
                    "filing_date": curr["filing_date"],
                    "source_available_date": curr["source_available_date"],
                    "source_accession": curr["accession_number"],
                    "metric_name": growth_name,
                    "metric_value": growth,
                    "unit": "ratio_yoy",
                })

        return growth_rows

    def _compute_segment_mix(self, xbrl_data: pd.DataFrame) -> list[dict]:
        """Compute segment revenue as % of total revenue per period."""
        rows: list[dict] = []

        seg = xbrl_data[xbrl_data["metric_name"] == "segment_revenue"]
        rev = xbrl_data[xbrl_data["metric_name"] == "revenue"]

        if seg.empty or rev.empty:
            logger.info("Segment mix: no segment_revenue or revenue data available")
            return rows

        for (fy, fp), seg_group in seg.groupby(["fiscal_year", "fiscal_period"]):
            rev_match = rev[
                (rev["fiscal_year"] == fy) & (rev["fiscal_period"] == fp)
            ]
            if rev_match.empty:
                continue
            total_rev = rev_match.iloc[0]["value"]
            if total_rev is None or total_rev == 0:
                continue

            period_label = f"FY{fy}" if fp == "FY" else f"FY{fy}-{fp}"
            for _, seg_row in seg_group.iterrows():
                seg_val = seg_row["value"]
                pct = self._safe_div(seg_val, total_rev)
                rows.append({
                    "ticker": self.config.ticker,
                    "fiscal_period": period_label,
                    "filing_date": seg_row["filing_date"],
                    "source_available_date": seg_row["source_available_date"],
                    "source_accession": seg_row["accession_number"],
                    "metric_name": "segment_mix",
                    "metric_value": pct,
                    "unit": "ratio",
                })

        return rows
