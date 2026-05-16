"""
v2 Quarterly Panel Builder (Reqs 1, 6, 8 — Milestone 4).

Assembles the dense quarterly feature/target matrix that the v2 ML
decision engine will train on. Combines four data streams:

1. Quarterly fundamentals (from ``data/processed/nvda_metrics.csv``),
   with mixed-frequency handling: flow concepts are stored as YTD-
   cumulative in NVIDIA's 10-Q filings and must be converted to
   per-period quarterly values via YTD-difference (Req 1.4).
2. NLP features (from ``data/processed/nvda_nlp_features.csv``), joined
   by filing_date.
3. Market features (from ``MarketFeatureBuilder.compute_features()``),
   joined by feature_available_date.
4. Targets (Req 6): primary = next-quarter YoY revenue growth;
   secondary = next-FY annual revenue growth; tertiary = forward-12m
   excess return direction vs SPX.

Key design points:

- The panel skeleton is one row per quarterly fiscal_period of the
  ticker, indexed by fiscal_period and feature_available_date.
- All flow features are derived per-period (gross_margin, operating_margin,
  fcf_margin) so they're directly usable.
- Per-period revenue is derived from YTD via subtraction within FY:
  Q1=YTD_Q1, Q2=YTD_Q2-YTD_Q1, Q3=YTD_Q3-YTD_Q2, Q4=annual-YTD_Q3.
- Validation: sum(Q1..Q4) must equal annual within 1% tolerance per Req 1.4.
  Mismatches set ``feature_imputed_<col>=True``.
- Targets respect Req 6 horizon-aware embargo:
  primary horizon=1Q, secondary horizon=4Q, tertiary horizon=4Q.
- Strict no-lookahead: feature_available_date < target_available_date
  for every row. Validated by ``validate_no_lookahead_panel()``.

Output is written to ``data/processed/ml_quarterly_panel.csv`` with
provenance columns (accession numbers, source dates, imputation flags).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.config import EngineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — concept categorisation for mixed-frequency handling
# ---------------------------------------------------------------------------

# Flow concepts: reported as YTD in 10-Q filings; must be derived
# per-period via subtraction.
FLOW_CONCEPTS = {
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "capex",
    "FCF",
    "r_and_d",
    "R&D_expense",
}

# Instant concepts: balance-sheet items at period end. Use as-is.
INSTANT_CONCEPTS = {
    "total_debt",
    "long_term_debt",
    "short_term_debt",
    "cash_and_securities",
    "diluted_shares",
}

# Ratio / level concepts: already correctly defined per-period in metrics
# (computed from period-specific numerator/denominator). Use as-is.
RATIO_CONCEPTS = {
    "gross_margin",
    "operating_margin",
    "net_margin",
    "FCF_margin",
    "R&D_%_revenue",
    "SG&A_%_revenue",
    "ROA",
    "ROE",
    "current_ratio",
    "debt_to_equity",
}


# ---------------------------------------------------------------------------
# Period parsing helpers
# ---------------------------------------------------------------------------

QUARTER_RE = re.compile(r"^FY(\d{4})-Q([1-4])$")
ANNUAL_RE = re.compile(r"^FY(\d{4})$")


def _parse_quarterly_period(period: str) -> Optional[tuple[int, int]]:
    """Return (fy, q) tuple or None."""
    m = QUARTER_RE.match(period)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _parse_annual_period(period: str) -> Optional[int]:
    m = ANNUAL_RE.match(period)
    return int(m.group(1)) if m else None


def _quarter_to_int(period: str) -> Optional[int]:
    """Map FY2024-Q1 → 2024*4+0 = 8096; FY2024 (annual) → fy*4+3.

    Returns a totally-ordered integer that respects (FY, Q) ordering:
    FY2024-Q1 < FY2024-Q2 < FY2024-Q3 < FY2024-Q4 (= FY2024 annual)
    < FY2025-Q1 etc.
    """
    pq = _parse_quarterly_period(period)
    if pq:
        fy, q = pq
        return fy * 4 + (q - 1)
    pa = _parse_annual_period(period)
    if pa:
        return pa * 4 + 3  # treat annual = Q4
    return None


def _next_quarter_label(period: str) -> Optional[str]:
    """Return the fiscal_period label for the quarter after this one.

    FY2024-Q1 → FY2024-Q2; FY2024-Q3 → FY2024-Q4 (mapped to FY2024 annual);
    FY2024-Q4 / FY2024 → FY2025-Q1.
    """
    pq = _parse_quarterly_period(period)
    if pq:
        fy, q = pq
        if q < 4:
            return f"FY{fy}-Q{q + 1}"
        return f"FY{fy + 1}-Q1"
    pa = _parse_annual_period(period)
    if pa:
        return f"FY{pa + 1}-Q1"
    return None


def _previous_quarter_label(period: str, n_back: int = 1) -> Optional[str]:
    """Return the fiscal_period label n quarters before this one."""
    pq = _parse_quarterly_period(period)
    if not pq:
        return None
    fy, q = pq
    total = fy * 4 + (q - 1) - n_back
    if total < 0:
        return None
    fy_back = total // 4
    q_back = (total % 4) + 1
    return f"FY{fy_back}-Q{q_back}"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class QuarterlyPanelBuilder:
    """Build the v2 quarterly feature/target panel."""

    YTD_TOLERANCE_PCT = 1.0  # Req 1.4: 1% sum-to-annual tolerance

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

    def _ticker_mask(self, metrics: pd.DataFrame) -> pd.Series:
        """Return a boolean Series filtering metrics rows to self.config.ticker.

        If the ``ticker`` column is missing (older v1 format), returns an
        all-True mask so that downstream filters apply only by metric_name
        and fiscal_period.
        """
        if "ticker" in metrics.columns:
            return metrics["ticker"] == self.config.ticker
        return pd.Series(True, index=metrics.index)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_panel(
        self,
        metrics: pd.DataFrame,
        nlp: Optional[pd.DataFrame] = None,
        market: Optional[pd.DataFrame] = None,
        prices: Optional[pd.DataFrame] = None,
        save_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Assemble the quarterly panel.

        Parameters
        ----------
        metrics
            Long-format metric DataFrame from ``financial_metrics``.
            Required columns: fiscal_period, filing_date,
            source_available_date, source_accession, metric_name,
            metric_value.
        nlp
            Long-format NLP feature DataFrame from ``nlp_features``.
            Required columns: filing_date, source_accession, section,
            feature_type, feature_name, value. May be None.
        market
            Wide-format market features (one row per as_of_date) from
            ``MarketFeatureBuilder.compute_features()``. May be None.
        prices
            Wide-format prices for the tertiary forward-return target.
            May be None (target_3 will be None).
        save_path
            Override default output path.
            Defaults to ``data/processed/ml_quarterly_panel.csv``.

        Returns
        -------
        pd.DataFrame
            One row per quarterly fiscal_period, with columns:
            - feature_period, feature_available_date, source_accession
            - <flow>_quarterly columns (per-period derived)
            - <ratio> columns (gross_margin, operating_margin, ...)
            - revenue_growth_QoQ, revenue_growth_YoY (computed from
              standalone quarterly values)
            - rd_intensity, capex_intensity, fcf_margin (derived)
            - operating_leverage (delta_op_inc / delta_revenue)
            - feature_imputed_<col> flags for YTD-diff failures
            - NLP, market columns
            - target_period, target_available_date,
              target_rev_growth_quarterly_YoY (primary target),
              target_rev_growth_annual_FY (secondary),
              target_excess_return_12m_vs_spx_direction (tertiary)
        """
        # 1. Build skeleton (one row per quarterly period)
        skeleton = self._build_skeleton(metrics)
        if skeleton.empty:
            logger.warning("Empty quarterly skeleton; cannot build panel.")
            return skeleton

        # 2. Attach fundamentals (per-period derived from YTD where needed)
        fundamentals = self._attach_fundamentals(skeleton, metrics)

        # 3. Attach NLP features
        if nlp is not None and not nlp.empty:
            with_nlp = self._attach_nlp(fundamentals, nlp)
        else:
            with_nlp = fundamentals

        # 4. Attach market features
        if market is not None and not market.empty:
            with_market = self._attach_market(with_nlp, market)
        else:
            with_market = with_nlp

        # 5. Attach targets
        with_targets = self._attach_targets(with_market, metrics, prices)

        # 6. Per-row no-lookahead flag — downstream walk-forward should
        #    drop rows where this is True (typically caused by upstream
        #    fiscal-year mislabelling in metrics).
        with_targets = self._flag_no_lookahead_violations(with_targets)

        # 7. Persist
        out_path = save_path or (
            self.config.processed_dir / "ml_quarterly_panel.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with_targets.to_csv(out_path, index=False)
        logger.info(
            "ml_quarterly_panel.csv written: %d rows × %d cols",
            len(with_targets),
            len(with_targets.columns),
        )
        return with_targets

    def _flag_no_lookahead_violations(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add a ``no_lookahead_violation`` boolean column.

        True iff the row has a populated primary target AND
        ``feature_available_date >= target_available_date``. These rows
        come from upstream data quality issues (typically fiscal-year
        mislabelling in the metrics file) and should be dropped by the
        walk-forward training loop.
        """
        df = panel.copy()
        flags: list[bool] = []
        for _, row in df.iterrows():
            tgt_avail = row.get("target_available_date")
            tgt_val = row.get("target_rev_growth_quarterly_YoY")
            if (
                tgt_avail is None
                or pd.isna(tgt_val)
                or row.get("feature_available_date") is None
            ):
                flags.append(False)
                continue
            try:
                if pd.Timestamp(row["feature_available_date"]) >= pd.Timestamp(tgt_avail):
                    flags.append(True)
                else:
                    flags.append(False)
            except (TypeError, ValueError):
                flags.append(False)
        df["no_lookahead_violation"] = flags
        n_violations = sum(flags)
        if n_violations > 0:
            logger.warning(
                "Flagged %d no-lookahead violation(s) in quarterly panel; "
                "these rows will be excluded from walk-forward training.",
                n_violations,
            )
        return df

    # ------------------------------------------------------------------
    # Step 1: skeleton
    # ------------------------------------------------------------------

    def _build_skeleton(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Build one row per quarterly fiscal_period for self.config.ticker.

        Uses revenue rows as the canonical period skeleton because
        every quarterly filing reports revenue.
        """
        if metrics.empty:
            return pd.DataFrame()
        df = metrics[
            self._ticker_mask(metrics)
            & metrics["metric_name"].eq("revenue")
            & metrics["fiscal_period"].str.match(r"^FY\d{4}-Q[1-4]$", na=False)
        ].copy()
        if df.empty:
            return pd.DataFrame()
        # One row per quarterly period; if duplicates exist, keep the
        # latest filing (highest source_available_date).
        df = df.sort_values(
            ["fiscal_period", "source_available_date"]
        ).drop_duplicates("fiscal_period", keep="last")
        skeleton = df[
            ["fiscal_period", "filing_date", "source_available_date",
             "source_accession"]
        ].rename(columns={
            "fiscal_period": "feature_period",
            "filing_date": "feature_filing_date",
            "source_available_date": "feature_available_date",
            "source_accession": "feature_accession",
        }).copy()
        skeleton = skeleton.sort_values("feature_period").reset_index(drop=True)
        return skeleton

    # ------------------------------------------------------------------
    # Step 2: fundamentals + YTD derivation
    # ------------------------------------------------------------------

    def _attach_fundamentals(
        self, skeleton: pd.DataFrame, metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach per-period fundamental features.

        For ratio concepts: pivot directly. For flow concepts: derive
        per-period values from YTD-cumulative quarterly rows via
        subtraction. Validate sum(Q1..Q4) ≈ annual.
        """
        df = skeleton.copy()
        m = metrics[self._ticker_mask(metrics)].copy()

        # 2a. Ratios (use as-is for quarterly periods).
        # Dedup deterministically: sort by source_available_date so when
        # multiple filings report the same fiscal_period × metric (an
        # amendment / restatement), the latest filing wins.
        m_sorted = m.sort_values(
            ["fiscal_period", "source_available_date"], kind="mergesort"
        )
        for ratio in RATIO_CONCEPTS:
            ratio_col = ratio.lower()  # gross_margin, operating_margin, ...
            ratio_data = (
                m_sorted[m_sorted["metric_name"] == ratio]
                .drop_duplicates("fiscal_period", keep="last")
                .set_index("fiscal_period")["metric_value"]
                .to_dict()
            )
            df[ratio_col] = df["feature_period"].map(ratio_data)

        # 2b. Flow concepts: derive per-period from YTD via subtraction
        for flow in FLOW_CONCEPTS:
            std_col = f"{flow.lower()}_quarterly"
            imputed_col = f"feature_imputed_{std_col}"
            quarterly_values, imputed_flags = self._derive_quarterly_from_ytd(
                m, flow
            )
            df[std_col] = df["feature_period"].map(quarterly_values)
            df[imputed_col] = df["feature_period"].map(imputed_flags).fillna(False)

        # 2c. Compute quarterly-YoY revenue growth (canonical primary feature)
        rev_q = df.set_index("feature_period")["revenue_quarterly"].to_dict()
        df["revenue_growth_QoQ"] = df["feature_period"].map(
            lambda p: self._yoy_growth(rev_q, p, n_back=1)
        )
        df["revenue_growth_YoY"] = df["feature_period"].map(
            lambda p: self._yoy_growth(rev_q, p, n_back=4)
        )

        # 2d. Intensity ratios on standalone quarterly values
        # rd_intensity = R&D / revenue (quarterly); capex_intensity = capex / revenue
        df["rd_intensity"] = self._safe_div(
            df.get("r_and_d_quarterly"), df["revenue_quarterly"]
        )
        df["capex_intensity"] = self._safe_div(
            df.get("capex_quarterly"), df["revenue_quarterly"]
        )
        # operating_leverage = delta_operating_income / delta_revenue (YoY)
        df["operating_leverage"] = self._operating_leverage(df)

        return df

    def _derive_quarterly_from_ytd(
        self, metrics: pd.DataFrame, flow_concept: str
    ) -> tuple[dict[str, Optional[float]], dict[str, bool]]:
        """Derive standalone quarterly values from YTD-cumulative XBRL data.

        Rule (Req 1.4):
            Q1 = YTD_Q1
            Q2 = YTD_Q2 - YTD_Q1
            Q3 = YTD_Q3 - YTD_Q2
            Q4 = annual - YTD_Q3

        Validation: sum(Q1..Q4) must equal annual within 1%. Mismatches
        flag the entire FY's quarterly values as imputed.
        """
        flow_rows = metrics[metrics["metric_name"] == flow_concept]
        if flow_rows.empty:
            return {}, {}

        # Dedup deterministically: when the same fiscal_period appears
        # multiple times (amendment / restatement), keep the latest filing
        # by source_available_date.
        flow_rows = flow_rows.sort_values(
            ["fiscal_period", "source_available_date"], kind="mergesort"
        ).drop_duplicates("fiscal_period", keep="last")

        # Build YTD lookup keyed by fiscal_period
        ytd_by_period = (
            flow_rows.set_index("fiscal_period")["metric_value"].to_dict()
        )

        # Group by FY: collect quarterly + annual values
        fy_quarters: dict[int, dict[str, float]] = {}
        for period, value in ytd_by_period.items():
            pq = _parse_quarterly_period(period)
            if pq:
                fy, q = pq
                fy_quarters.setdefault(fy, {})[f"q{q}"] = float(value)
                continue
            pa = _parse_annual_period(period)
            if pa is not None:
                fy_quarters.setdefault(pa, {})["annual"] = float(value)

        derived: dict[str, Optional[float]] = {}
        imputed: dict[str, bool] = {}

        for fy, vals in fy_quarters.items():
            q1 = vals.get("q1")
            q2_ytd = vals.get("q2")
            q3_ytd = vals.get("q3")
            q4_ytd = vals.get("q4")  # rare: full-year YTD reported as Q4
            annual = vals.get("annual")

            # Per-period derivation via subtraction
            d_q1 = q1 if q1 is not None else None
            d_q2 = q2_ytd - q1 if (q2_ytd is not None and q1 is not None) else None
            d_q3 = (
                q3_ytd - q2_ytd if (q3_ytd is not None and q2_ytd is not None)
                else None
            )
            # Q4 preference: annual − YTD_Q3 (most common). Fall back to
            # explicit Q4 row if present and YTD_Q3 absent.
            if annual is not None and q3_ytd is not None:
                d_q4 = annual - q3_ytd
            elif q4_ytd is not None and q3_ytd is not None:
                d_q4 = q4_ytd - q3_ytd
            else:
                d_q4 = None

            # Sum-to-annual validation
            std_quarters = [d_q1, d_q2, d_q3, d_q4]
            fy_imputed = False
            if annual is not None and all(q is not None for q in std_quarters):
                summed = sum(std_quarters)
                tolerance = abs(annual) * (self.YTD_TOLERANCE_PCT / 100.0)
                if abs(summed - annual) > tolerance:
                    fy_imputed = True
                    logger.info(
                        "%s FY%d sum-to-annual mismatch: sum=%.2f annual=%.2f "
                        "(diff=%.2f%%) → flagged as feature_imputed",
                        flow_concept, fy, summed, annual,
                        100 * abs(summed - annual) / abs(annual) if annual else 0,
                    )

            for i, val in enumerate(std_quarters, start=1):
                period_label = f"FY{fy}-Q{i}"
                derived[period_label] = val
                imputed[period_label] = fy_imputed

        return derived, imputed

    @staticmethod
    def _yoy_growth(
        period_to_value: dict[str, float], period: str, n_back: int
    ) -> Optional[float]:
        """Compute (current - prior) / prior where prior is n quarters ago."""
        cur = period_to_value.get(period)
        prior_period = _previous_quarter_label(period, n_back=n_back)
        if prior_period is None:
            return None
        prior = period_to_value.get(prior_period)
        if cur is None or prior is None or prior == 0:
            return None
        try:
            return round((cur - prior) / abs(prior), 6)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def _safe_div(num, den):
        """Element-wise safe division for pandas Series (or scalars)."""
        if num is None:
            return None
        if isinstance(num, pd.Series) and isinstance(den, pd.Series):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = num / den
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            return ratio.round(6)
        # Scalar fallback
        try:
            if den == 0 or den is None:
                return None
            return round(num / den, 6)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _operating_leverage(panel: pd.DataFrame) -> pd.Series:
        """Operating leverage = ΔOp Income / ΔRevenue (YoY).

        Computed within the panel by joining the period 4 quarters ago.
        Returns NaN where either quarter is missing.
        """
        oi = panel.get("operating_income_quarterly")
        rev = panel.get("revenue_quarterly")
        if oi is None or rev is None:
            return pd.Series([None] * len(panel), index=panel.index)
        df = pd.DataFrame({
            "feature_period": panel["feature_period"],
            "oi": oi,
            "rev": rev,
        })
        df["prior_period"] = df["feature_period"].apply(
            lambda p: _previous_quarter_label(p, n_back=4)
        )
        prior_map = df.set_index("feature_period")[["oi", "rev"]].to_dict("index")
        result = []
        for _, row in df.iterrows():
            pp = row["prior_period"]
            prior = prior_map.get(pp)
            if prior is None or row["oi"] is None or row["rev"] is None:
                result.append(None)
                continue
            d_oi = row["oi"] - prior["oi"] if prior["oi"] is not None else None
            d_rev = row["rev"] - prior["rev"] if prior["rev"] is not None else None
            if d_oi is None or d_rev is None or d_rev == 0:
                result.append(None)
            else:
                result.append(round(d_oi / d_rev, 6))
        return pd.Series(result, index=panel.index)

    # ------------------------------------------------------------------
    # Step 3: NLP attachment
    # ------------------------------------------------------------------

    def _attach_nlp(self, panel: pd.DataFrame, nlp: pd.DataFrame) -> pd.DataFrame:
        """Join NLP features by source_accession (preferred) or filing_date.

        Pivots the long-format nlp DataFrame into wide columns named
        ``nlp_<section>_<feature_name>``. NaN fills for missing values
        (Req 3.5: NO LOCF for NLP).

        Deterministic dedup: when the same (accession, section,
        feature_name) appears multiple times in the NLP feature CSV
        (e.g., due to multiple computation passes), the LAST row by
        filing_date wins. We sort explicitly before pivoting so
        ``aggfunc='last'`` is reproducible across runs.
        """
        if nlp.empty or "source_accession" not in nlp.columns:
            return panel
        # Pivot to one column per (section, feature_name)
        pivot = nlp.copy()
        pivot["col"] = (
            "nlp_" + pivot["section"].astype(str)
            + "_" + pivot["feature_name"].astype(str)
        )
        # Sort deterministically before pivot so aggfunc='last' is
        # reproducible. Use filing_date as the tie-breaker (latest wins);
        # fall back to row index for stability.
        sort_cols = [c for c in ["source_accession", "col", "filing_date"]
                     if c in pivot.columns]
        pivot = pivot.sort_values(sort_cols, kind="mergesort")
        wide = pivot.pivot_table(
            index="source_accession",
            columns="col",
            values="value",
            aggfunc="last",
        ).reset_index()
        out = panel.merge(
            wide,
            left_on="feature_accession",
            right_on="source_accession",
            how="left",
        )
        if "source_accession" in out.columns and out is not panel:
            out = out.drop(columns=["source_accession"])
        return out

    # ------------------------------------------------------------------
    # Step 4: market attachment
    # ------------------------------------------------------------------

    def _attach_market(
        self, panel: pd.DataFrame, market: pd.DataFrame
    ) -> pd.DataFrame:
        """Join market features by feature_available_date == as_of_date."""
        if market.empty or "as_of_date" not in market.columns:
            return panel
        mkt = market.copy()
        mkt["as_of_date"] = pd.to_datetime(mkt["as_of_date"]).dt.strftime("%Y-%m-%d")
        return panel.merge(
            mkt, left_on="feature_available_date", right_on="as_of_date",
            how="left",
        ).drop(columns=["as_of_date"], errors="ignore")

    # ------------------------------------------------------------------
    # Step 5: targets
    # ------------------------------------------------------------------

    def _attach_targets(
        self,
        panel: pd.DataFrame,
        metrics: pd.DataFrame,
        prices: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Attach the three v2 ML targets (Req 6).

        - Primary: rev_growth_quarterly_YoY at quarter t+1
        - Secondary: rev_growth_annual_FY for the next full FY
        - Tertiary: excess_return_12m_vs_spx_direction
        """
        df = panel.copy()

        # --- Primary: next-quarter YoY revenue growth ---
        rev_q = df.set_index("feature_period")["revenue_quarterly"].to_dict()
        next_q_label_map = {
            p: _next_quarter_label(p) for p in df["feature_period"]
        }
        next_filing_date_map = (
            df.set_index("feature_period")["feature_available_date"].to_dict()
        )

        primary_vals: list[Optional[float]] = []
        target_periods: list[Optional[str]] = []
        target_avail_dates: list[Optional[str]] = []
        for _, row in df.iterrows():
            cur_period = row["feature_period"]
            next_period = next_q_label_map.get(cur_period)
            if next_period is None or next_period not in rev_q:
                primary_vals.append(None)
                target_periods.append(None)
                target_avail_dates.append(None)
                continue
            # YoY at next quarter: revenue[next] vs revenue[next - 4q]
            yoy = self._yoy_growth(rev_q, next_period, n_back=4)
            primary_vals.append(yoy)
            target_periods.append(next_period)
            # target_available_date = filing_date of the next quarter
            target_avail_dates.append(next_filing_date_map.get(next_period))

        df["target_period"] = target_periods
        df["target_available_date"] = target_avail_dates
        df["target_rev_growth_quarterly_YoY"] = primary_vals

        # --- Secondary: next-FY annual revenue growth ---
        annual_revenue = (
            metrics[
                self._ticker_mask(metrics)
                & (metrics["metric_name"] == "revenue")
                & metrics["fiscal_period"].str.match(r"^FY\d{4}$", na=False)
            ]
            .sort_values("fiscal_period")
            .drop_duplicates("fiscal_period", keep="last")
        )
        rev_annual_map = annual_revenue.set_index("fiscal_period")[
            "metric_value"
        ].to_dict()
        annual_avail_map = annual_revenue.set_index("fiscal_period")[
            "source_available_date"
        ].to_dict()

        sec_vals: list[Optional[float]] = []
        sec_avail: list[Optional[str]] = []
        sec_periods: list[Optional[str]] = []
        for _, row in df.iterrows():
            cur_period = row["feature_period"]
            pq = _parse_quarterly_period(cur_period)
            if not pq:
                sec_vals.append(None)
                sec_avail.append(None)
                sec_periods.append(None)
                continue
            cur_fy, _ = pq
            next_fy_label = f"FY{cur_fy + 1}"
            cur_fy_label = f"FY{cur_fy}"
            cur_annual = rev_annual_map.get(cur_fy_label)
            next_annual = rev_annual_map.get(next_fy_label)
            if cur_annual is None or next_annual is None or cur_annual == 0:
                sec_vals.append(None)
            else:
                sec_vals.append(round((next_annual - cur_annual) / cur_annual, 6))
            sec_periods.append(next_fy_label)
            sec_avail.append(annual_avail_map.get(next_fy_label))
        df["target_secondary_period"] = sec_periods
        df["target_secondary_available_date"] = sec_avail
        df["target_rev_growth_annual_FY"] = sec_vals

        # --- Tertiary: forward 12m excess return direction vs SPX ---
        if prices is not None and not prices.empty:
            df["target_excess_return_12m_vs_spx_direction"] = (
                self._compute_excess_return_direction(df, prices)
            )
        else:
            df["target_excess_return_12m_vs_spx_direction"] = None

        return df

    def _compute_excess_return_direction(
        self, panel: pd.DataFrame, prices: pd.DataFrame
    ) -> list[Optional[int]]:
        """Forward 12m return direction: 1 if NVDA forward 12m return >
        SPX forward 12m return, else 0. None if either is unavailable.

        Note: this target uses prices AFTER feature_available_date —
        it is a forward-looking target used in walk-forward as the
        ground truth, not a feature.
        """
        if prices.empty or "ticker" not in prices.columns:
            return [None] * len(panel)
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        nvda = df[df["ticker"] == self.config.ticker].set_index("date")["adj_close"]
        # Ordered tuple (canonical first) for deterministic alias resolution.
        spx_aliases = ("^GSPC", "GSPC", ".GSPC", "^SPX", "SPX")
        spx_ticker = next((t for t in spx_aliases if t in df["ticker"].unique()), None)
        spx = (
            df[df["ticker"] == spx_ticker].set_index("date")["adj_close"]
            if spx_ticker
            else None
        )
        if nvda.empty or spx is None or spx.empty:
            return [None] * len(panel)
        nvda = nvda.sort_index()
        spx = spx.sort_index()

        result: list[Optional[int]] = []
        for _, row in panel.iterrows():
            avail_date = pd.Timestamp(row["feature_available_date"])
            future_date = avail_date + pd.Timedelta(days=365)
            n_now = self._last_le(nvda, avail_date)
            n_future = self._first_ge(nvda, future_date)
            s_now = self._last_le(spx, avail_date)
            s_future = self._first_ge(spx, future_date)
            if (
                n_now is None or n_future is None or s_now is None or s_future is None
                or n_now == 0 or s_now == 0
            ):
                result.append(None)
                continue
            nvda_ret = n_future / n_now - 1
            spx_ret = s_future / s_now - 1
            result.append(1 if nvda_ret > spx_ret else 0)
        return result

    @staticmethod
    def _last_le(s: pd.Series, ts: pd.Timestamp) -> Optional[float]:
        sub = s.loc[:ts]
        if sub.empty:
            return None
        return float(sub.iloc[-1])

    @staticmethod
    def _first_ge(s: pd.Series, ts: pd.Timestamp) -> Optional[float]:
        sub = s.loc[ts:]
        if sub.empty:
            return None
        return float(sub.iloc[0])

    # ------------------------------------------------------------------
    # No-lookahead validation
    # ------------------------------------------------------------------

    def validate_no_lookahead_panel(self, panel: pd.DataFrame) -> dict:
        """Verify feature_available_date < target_available_date for every
        row that has a populated target.

        Returns a dict with:
        - ``ok``: bool — overall pass
        - ``violations``: list of dicts describing offending rows
        - ``rows_checked``: int
        """
        if panel.empty or "feature_available_date" not in panel.columns:
            return {"ok": True, "violations": [], "rows_checked": 0}

        violations: list[dict] = []
        for _, row in panel.iterrows():
            tgt_avail = row.get("target_available_date")
            tgt_val = row.get("target_rev_growth_quarterly_YoY")
            if tgt_avail is None or pd.isna(tgt_val):
                continue
            if pd.Timestamp(row["feature_available_date"]) >= pd.Timestamp(tgt_avail):
                violations.append({
                    "feature_period": row["feature_period"],
                    "feature_available_date": str(row["feature_available_date"]),
                    "target_available_date": str(tgt_avail),
                })
        return {
            "ok": len(violations) == 0,
            "violations": violations,
            "rows_checked": int(panel["target_rev_growth_quarterly_YoY"].notna().sum()),
        }
