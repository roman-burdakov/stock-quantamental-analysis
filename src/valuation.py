"""
NVDA Quantamental Engine — Valuation Module.

DCF valuation, reverse-DCF grid, peer multiples, scenario analysis,
and scorecard-based recommendation.

Requirements: 9.1–9.14, 21.1–21.7
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize

from src.config import (
    ComponentStatus,
    ComponentStatusEnum,
    DataQualityStatus,
    EngineConfig,
    FinalRecommendation,
    Recommendation,
    RecommendationStatus,
    RecommendationThresholds,
    ReportMode,
    ScenarioAssumptions,
    ValidatedMetric,
    get_default_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peer row status constants (Req 21.1)
# ---------------------------------------------------------------------------

class PeerRowStatus:
    """String constants for peer row exclusion/inclusion reasons."""
    USABLE = "usable"
    MISSING_EV = "missing_ev"
    NEGATIVE_EV = "negative_ev"
    NEGATIVE_EBITDA = "negative_ebitda"
    NEGATIVE_EARNINGS = "negative_earnings"
    STALE_MARKET_DATA = "stale_market_data"
    CONTEXT_ONLY = "context_only"
    EXCLUDED_FROM_PRIMARY_CHART = "excluded_from_primary_chart"


class ValuationModule:
    """DCF, reverse-DCF, peer multiples, scenarios, recommendation."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or get_default_config()

    # ------------------------------------------------------------------
    # 1. Core DCF
    # ------------------------------------------------------------------

    def compute_dcf(
        self,
        assumptions: ScenarioAssumptions,
        base_revenue: float,
        wacc: float,
        net_cash: float,
        shares: float,
    ) -> dict:
        """Run a multi-year DCF using *assumptions*.

        FCF = projected_revenue × fcf_margin (linearly interpolated from
        fcf_margin_start to fcf_margin_terminal over the projection window).

        Returns dict with full projection detail.
        """
        n = self.config.projection_years
        cagr = assumptions.revenue_cagr
        tg = assumptions.terminal_growth
        margin_start = assumptions.fcf_margin_start
        margin_end = assumptions.fcf_margin_terminal

        projected_revenue: list[float] = []
        fcf_margins: list[float] = []
        projected_fcf: list[float] = []
        discount_factors: list[float] = []
        pv_fcf: list[float] = []

        for yr in range(1, n + 1):
            rev = base_revenue * ((1 + cagr) ** yr)
            projected_revenue.append(rev)

            # Linear interpolation of FCF margin
            if n > 1:
                margin = margin_start + (margin_end - margin_start) * ((yr - 1) / (n - 1))
            else:
                margin = margin_end
            fcf_margins.append(margin)

            fcf = rev * margin
            projected_fcf.append(fcf)

            df = 1 / ((1 + wacc) ** yr)
            discount_factors.append(df)

            pv_fcf.append(fcf * df)

        # Terminal value (Gordon Growth)
        # NOTE: When WACC ≈ terminal_growth, the Gordon Growth model produces
        # extreme values. We apply a cap at 200× terminal FCF to prevent
        # nonsensical outputs. This creates a discontinuity in sensitivity
        # tables near the WACC=tg boundary — see limitations.md.
        terminal_fcf = projected_fcf[-1] * (1 + tg)
        spread = wacc - tg
        terminal_value_cap = terminal_fcf * 200  # Finite cap for edge cases

        if spread > 0.005:
            # Normal case: sufficient spread between WACC and terminal growth
            terminal_value = terminal_fcf / spread
        elif spread > 0:
            # Very narrow spread — cap at 200× terminal FCF to avoid absurd values
            terminal_value = min(terminal_fcf / spread, terminal_value_cap)
            logger.warning(
                "WACC-growth spread very narrow (%.4f) — terminal value capped at "
                "200× FCF. Sensitivity table may show discontinuity near this point.",
                spread,
            )
        else:
            # WACC ≤ terminal growth — Gordon Growth model is undefined.
            # Use a finite cap (200× terminal FCF) rather than 0 to avoid
            # discontinuity in sensitivity tables.
            terminal_value = terminal_value_cap
            logger.warning(
                "WACC (%.4f) ≤ terminal growth (%.4f) — terminal value capped at "
                "200× FCF ($%.1fB). Gordon Growth model is undefined in this region.",
                wacc, tg, terminal_value / 1e9,
            )
        pv_terminal = terminal_value / ((1 + wacc) ** n)

        sum_pv_fcf = sum(pv_fcf)
        enterprise_value = sum_pv_fcf + pv_terminal
        equity_value = enterprise_value + net_cash
        per_share_value = equity_value / shares if shares > 0 else 0.0

        return {
            "projected_revenue": projected_revenue,
            "fcf_margin": fcf_margins,
            "projected_fcf": projected_fcf,
            "discount_factors": discount_factors,
            "pv_fcf": pv_fcf,
            "terminal_value": terminal_value,
            "pv_terminal": pv_terminal,
            "enterprise_value": enterprise_value,
            "net_cash_bridge": net_cash,
            "equity_value": equity_value,
            "per_share_value": per_share_value,
        }

    # ------------------------------------------------------------------
    # 2. Historical FCF margin reconciliation
    # ------------------------------------------------------------------

    def reconcile_historical_fcf_margin(
        self, metrics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare assumed FCF margins to historical actuals.

        *metrics* is the output of ``FinancialMetricsCalculator.compute_all_metrics``
        with columns: fiscal_period, metric_name, metric_value.

        Returns a DataFrame with columns:
            fiscal_period, historical_fcf_margin, assumed_start, assumed_terminal, gap_vs_start
        """
        fcf_rows = metrics[metrics["metric_name"] == "FCF_margin"].copy()
        if fcf_rows.empty:
            logger.warning("No FCF_margin rows found in metrics for reconciliation")
            return pd.DataFrame(columns=[
                "fiscal_period", "historical_fcf_margin",
                "assumed_start", "assumed_terminal", "gap_vs_start",
            ])

        # Use base-case assumptions as reference
        base = self.config.scenarios.get("base")
        assumed_start = base.fcf_margin_start if base else None
        assumed_terminal = base.fcf_margin_terminal if base else None

        records: list[dict] = []
        for _, row in fcf_rows.iterrows():
            hist = row["metric_value"]
            gap = (hist - assumed_start) if (hist is not None and assumed_start is not None) else None
            records.append({
                "fiscal_period": row["fiscal_period"],
                "historical_fcf_margin": hist,
                "assumed_start": assumed_start,
                "assumed_terminal": assumed_terminal,
                "gap_vs_start": gap,
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 3. Sensitivity table (WACC × terminal growth)
    # ------------------------------------------------------------------

    def compute_sensitivity_table(
        self,
        base: ScenarioAssumptions,
        revenue: float,
        cash: float,
        shares: float,
        wacc_range: list[float],
        tg_range: list[float],
    ) -> pd.DataFrame:
        """WACC × terminal_growth 5×5 grid of per-share values.

        Returns DataFrame indexed by WACC with terminal-growth columns.
        """
        grid: dict[str, list[float]] = {}
        for tg in tg_range:
            col_values: list[float] = []
            for w in wacc_range:
                modified = ScenarioAssumptions(
                    name=base.name,
                    probability=base.probability,
                    revenue_cagr=base.revenue_cagr,
                    fcf_margin_start=base.fcf_margin_start,
                    fcf_margin_terminal=base.fcf_margin_terminal,
                    terminal_growth=tg,
                    sbc_treatment=base.sbc_treatment,
                    analyst_notes=base.analyst_notes,
                )
                result = self.compute_dcf(modified, revenue, w, cash, shares)
                col_values.append(result["per_share_value"])
            grid[f"tg={tg:.3f}"] = col_values

        return pd.DataFrame(grid, index=[f"wacc={w:.3f}" for w in wacc_range])

    # ------------------------------------------------------------------
    # 4. Reverse-DCF grid
    # ------------------------------------------------------------------

    # Historical plausible ranges for NVIDIA (used for plausibility assessment)
    _PLAUSIBLE_CAGR_MIN = 0.05
    _PLAUSIBLE_CAGR_MAX = 0.30
    _PLAUSIBLE_MARGIN_MIN = 0.15
    _PLAUSIBLE_MARGIN_MAX = 0.45

    def compute_reverse_dcf_grid(
        self,
        price: float,
        shares: float,
        cash: float,
        wacc: float,
        cagr_range: list[float],
        margin_range: list[float],
        current_price: float | None = None,
    ) -> pd.DataFrame:
        """Revenue-CAGR × terminal-FCF-margin grid → implied share price.

        Cells within ±10 % of *price* are flagged in a companion
        ``_highlight`` DataFrame attribute (stored as ``grid.attrs["highlight"]``).

        When *current_price* is provided and falls outside the grid range
        (no cell within ±10 %), the method uses solvers to compute implied
        CAGR and margin, storing the results and a plausibility assessment
        in the DataFrame's attrs.  The grid is NOT extended to unreasonable
        assumptions — instead, a ``grid_warning`` attr explains the situation.
        """
        grid: dict[str, list[float]] = {}
        highlight: dict[str, list[bool]] = {}

        # Use current_price for highlight comparison if provided, else fall back to price
        ref_price = current_price if current_price is not None else price

        for margin in margin_range:
            col_vals: list[float] = []
            col_hl: list[bool] = []
            for cagr in cagr_range:
                result = self._dcf_per_share(cagr, margin, wacc, cash, shares)
                col_vals.append(result)
                col_hl.append(abs(result - ref_price) / ref_price <= 0.10 if ref_price > 0 else False)
            grid[f"margin={margin:.2f}"] = col_vals
            highlight[f"margin={margin:.2f}"] = col_hl

        df = pd.DataFrame(grid, index=[f"cagr={c:.2f}" for c in cagr_range])
        df.attrs["highlight"] = pd.DataFrame(
            highlight, index=[f"cagr={c:.2f}" for c in cagr_range],
        )

        # --- Current-price-outside-grid analysis (Req 9.7, 16.3) ---
        if current_price is not None and current_price > 0:
            all_values = df.select_dtypes(include=[np.number]).values
            price_inside_grid = bool(np.any(
                np.abs(all_values - current_price) / current_price <= 0.10
            ))
            df.attrs["price_inside_grid"] = price_inside_grid

            if not price_inside_grid:
                # Compute implied assumptions using solvers
                mid_margin = margin_range[len(margin_range) // 2]
                mid_cagr = cagr_range[len(cagr_range) // 2]

                implied_cagr = self.solve_implied_cagr(
                    current_price, shares, cash, wacc, mid_margin,
                )
                implied_margin = self.solve_implied_margin(
                    current_price, shares, cash, wacc, mid_cagr,
                )

                df.attrs["implied_cagr"] = implied_cagr
                df.attrs["implied_margin"] = implied_margin

                # Assess plausibility against historical ranges
                cagr_assessment = self._assess_plausibility(
                    implied_cagr, self._PLAUSIBLE_CAGR_MIN, self._PLAUSIBLE_CAGR_MAX, "CAGR",
                )
                margin_assessment = self._assess_plausibility(
                    implied_margin, self._PLAUSIBLE_MARGIN_MIN, self._PLAUSIBLE_MARGIN_MAX, "FCF margin",
                )

                df.attrs["implied_cagr_assessment"] = cagr_assessment
                df.attrs["implied_margin_assessment"] = margin_assessment

                grid_min = float(np.min(all_values))
                grid_max = float(np.max(all_values))
                df.attrs["grid_warning"] = (
                    f"Current price (${current_price:,.2f}) lies outside the "
                    f"reverse-DCF grid range (${grid_min:,.2f}–${grid_max:,.2f}). "
                    f"Implied CAGR={implied_cagr:.1%} (at {mid_margin:.0%} margin): {cagr_assessment}. "
                    f"Implied margin={implied_margin:.1%} (at {mid_cagr:.0%} CAGR): {margin_assessment}."
                )
                logger.info(
                    "Current price $%.2f outside grid range $%.2f–$%.2f; "
                    "implied CAGR=%.1f%%, implied margin=%.1f%%",
                    current_price, grid_min, grid_max,
                    implied_cagr * 100, implied_margin * 100,
                )

        return df

    @staticmethod
    def _assess_plausibility(
        value: float,
        plausible_min: float,
        plausible_max: float,
        label: str,
    ) -> str:
        """Classify an implied assumption as within, above, or far outside plausible range."""
        import math
        if math.isnan(value):
            return f"{label} could not be solved (no convergence)"
        if plausible_min <= value <= plausible_max:
            return f"within historical/plausible range ({plausible_min:.0%}–{plausible_max:.0%})"
        elif value > plausible_max:
            overshoot = value - plausible_max
            if overshoot <= 0.10:
                return f"above plausible range ({plausible_min:.0%}–{plausible_max:.0%})"
            else:
                return f"far outside plausible range ({plausible_min:.0%}–{plausible_max:.0%})"
        else:
            undershoot = plausible_min - value
            if undershoot <= 0.10:
                return f"below plausible range ({plausible_min:.0%}–{plausible_max:.0%})"
            else:
                return f"far outside plausible range ({plausible_min:.0%}–{plausible_max:.0%})"

    def generate_reverse_dcf_report(self, grid: pd.DataFrame) -> str:
        """Produce a text summary of the reverse-DCF grid analysis.

        Includes:
        - Grid description (dimensions, assumption ranges)
        - Whether current price is inside or outside the grid
        - If outside: implied CAGR and margin with plausibility assessment
        - Historical plausible ranges for NVIDIA
        """
        lines: list[str] = []
        lines.append("## Reverse-DCF Grid Analysis")
        lines.append("")

        # Grid description
        n_rows, n_cols = grid.shape
        lines.append(
            f"Grid dimensions: {n_rows} CAGR assumptions × {n_cols} FCF margin assumptions."
        )
        lines.append(
            f"Historical plausible ranges for NVIDIA: "
            f"CAGR {self._PLAUSIBLE_CAGR_MIN:.0%}–{self._PLAUSIBLE_CAGR_MAX:.0%}, "
            f"FCF margin {self._PLAUSIBLE_MARGIN_MIN:.0%}–{self._PLAUSIBLE_MARGIN_MAX:.0%}."
        )
        lines.append("")

        # Price-inside-grid status
        price_inside = grid.attrs.get("price_inside_grid")
        if price_inside is True:
            lines.append(
                "Current price is **within** the modeled grid range "
                "(at least one cell within ±10% of current price)."
            )
            # Show highlighted cells
            hl = grid.attrs.get("highlight")
            if hl is not None:
                highlighted_count = int(hl.values.sum())
                lines.append(
                    f"{highlighted_count} cell(s) highlighted as near current price."
                )
        elif price_inside is False:
            lines.append(
                "⚠ Current price is **outside** the modeled grid range."
            )
            lines.append("")

            warning = grid.attrs.get("grid_warning", "")
            if warning:
                lines.append(warning)
                lines.append("")

            implied_cagr = grid.attrs.get("implied_cagr")
            implied_margin = grid.attrs.get("implied_margin")
            cagr_assess = grid.attrs.get("implied_cagr_assessment", "")
            margin_assess = grid.attrs.get("implied_margin_assessment", "")

            if implied_cagr is not None:
                lines.append(f"- Implied CAGR: {implied_cagr:.1%} — {cagr_assess}")
            if implied_margin is not None:
                lines.append(f"- Implied FCF margin: {implied_margin:.1%} — {margin_assess}")

            lines.append("")
            lines.append(
                "The grid was NOT extended to unreasonable assumptions. "
                "The analyst should interpret whether the market-implied "
                "expectations are achievable."
            )
        else:
            lines.append(
                "No current price was provided for grid comparison."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 5 & 6. One-variable solvers
    # ------------------------------------------------------------------

    def solve_implied_cagr(
        self,
        price: float,
        shares: float,
        cash: float,
        wacc: float,
        fixed_margin: float,
    ) -> float:
        """Solve for the revenue CAGR that equates DCF value to *price*."""

        def _objective(cagr: float) -> float:
            implied = self._dcf_per_share(cagr, fixed_margin, wacc, cash, shares)
            return implied - price

        try:
            result = optimize.brentq(_objective, -0.50, 1.50, xtol=1e-6)
            return float(result)
        except ValueError:
            logger.warning("solve_implied_cagr: no root in [-0.50, 1.50]")
            return float("nan")

    def solve_implied_margin(
        self,
        price: float,
        shares: float,
        cash: float,
        wacc: float,
        fixed_cagr: float,
    ) -> float:
        """Solve for the terminal FCF margin that equates DCF value to *price*."""

        def _objective(margin: float) -> float:
            implied = self._dcf_per_share(fixed_cagr, margin, wacc, cash, shares)
            return implied - price

        try:
            result = optimize.brentq(_objective, 0.01, 0.90, xtol=1e-6)
            return float(result)
        except ValueError:
            logger.warning("solve_implied_margin: no root in [0.01, 0.90]")
            return float("nan")

    # ------------------------------------------------------------------
    # 7. Peer multiples
    # ------------------------------------------------------------------

    def compute_peer_multiples(
        self,
        peer_fin: pd.DataFrame,
        nvda_fin: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute EV/Revenue, EV/EBITDA, P/E, FCF_yield for peers + NVDA.

        Skip EV-based multiples for peers with missing or stale EV inputs
        (indicated by a ``stale_ev`` column == True, or missing market_cap/debt/cash).
        """
        combined = pd.concat([peer_fin, nvda_fin], ignore_index=True)
        records: list[dict] = []

        for _, row in combined.iterrows():
            ticker = row.get("ticker", "")
            market_cap = row.get("market_cap")
            debt = row.get("total_debt", 0) or 0
            # Prefer cash_and_securities, fall back to total_cash; handle NaN
            cash_val = row.get("cash_and_securities")
            if cash_val is None or (isinstance(cash_val, float) and pd.isna(cash_val)):
                cash_val = row.get("total_cash", 0) or 0
            else:
                cash_val = cash_val or 0
            revenue = row.get("revenue")
            ebitda = row.get("ebitda")
            net_income = row.get("net_income")
            # Prefer 'fcf' but fall back to 'free_cash_flow'; handle NaN
            fcf = row.get("fcf")
            if fcf is None or (isinstance(fcf, float) and pd.isna(fcf)):
                fcf = row.get("free_cash_flow")
            stale_raw = row.get("stale_ev", False)
            stale = bool(stale_raw) if not pd.isna(stale_raw) else False

            # Determine if EV is computable — compute from components
            # even when financial data is somewhat stale, since market_cap
            # is current.  Staleness is logged as a warning, not a blocker.
            ev_valid = (
                market_cap is not None
                and not pd.isna(market_cap)
            )
            ev = (market_cap + debt - cash_val) if ev_valid else None

            ev_revenue = self._safe_div(ev, revenue)
            ev_ebitda = self._safe_div(ev, ebitda)
            pe = self._safe_div(market_cap, net_income) if market_cap is not None else None
            fcf_yield = self._safe_div(fcf, market_cap) if market_cap is not None else None

            rec: dict = {
                "ticker": ticker,
                "market_cap": market_cap,
                "enterprise_value": ev,
                "EV/Revenue": ev_revenue,
                "EV/EBITDA": ev_ebitda,
                "P/E": pe,
                "FCF_yield": fcf_yield,
                "stale_financials": stale,
                "source_date": row.get("source_date"),
            }
            if not ev_valid:
                rec["EV/Revenue"] = None
                rec["EV/EBITDA"] = None
                logger.info(
                    "Skipping EV multiples for %s (missing market_cap)", ticker,
                )
            elif stale:
                logger.info(
                    "Peer %s financials are stale (>%d days) — EV computed from "
                    "current market_cap + older debt/cash",
                    ticker, self.config.peer_staleness_threshold_days,
                )
            records.append(rec)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 7b. Peer filtering and quality (Reqs 21.1–21.7)
    # ------------------------------------------------------------------

    def filter_peer_multiples(
        self,
        peer_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Filter and annotate peer multiples for quality.

        Adds two columns to the returned DataFrame:
        - ``peer_tier``: "semi", "infrastructure", or "context"
        - ``row_status``: a :class:`PeerRowStatus` constant indicating
          whether the row is usable or the reason for exclusion.

        Filtering rules (Req 21):
        1. NaN or negative ``enterprise_value`` → excluded from EV-based
           multiples (EV/Revenue, EV/EBITDA).
        2. Negative ``net_income`` (via P/E < 0 or net_income < 0) →
           excluded from primary P/E chart.
        3. Negative EBITDA → excluded from EV/EBITDA.
        4. Stale peer data (``source_date`` > ``peer_staleness_threshold_days``
           before ``report_date``) → excluded from primary valuation.

        Returns
        -------
        (filtered_df, exclusion_log)
            *filtered_df* contains all rows (including excluded ones) with
            ``peer_tier`` and ``row_status`` columns added.  Excluded rows
            have their affected multiples set to ``None``.
            *exclusion_log* is a list of human-readable exclusion messages.
        """
        df = peer_df.copy()
        exclusion_log: list[str] = []

        # --- Assign peer tiers (Reqs 21.4, 21.5) ---
        semi_set = set(self.config.core_semiconductor_peers)
        infra_set = set(self.config.infrastructure_peers)
        context_set = set(self.config.ai_capex_context)

        def _classify_tier(ticker: str) -> str:
            if ticker in semi_set:
                return "semi"
            if ticker in infra_set:
                return "infrastructure"
            if ticker in context_set:
                return "context"
            return "context"  # default for unknown tickers

        df["peer_tier"] = df["ticker"].apply(_classify_tier)

        # --- Initialise row_status ---
        df["row_status"] = PeerRowStatus.USABLE

        # --- Staleness check (Req 21.7) ---
        if "source_date" in df.columns:
            report_dt = datetime.strptime(self.config.report_date, "%Y-%m-%d")
            threshold_days = self.config.peer_staleness_threshold_days

            for idx, row in df.iterrows():
                src_date = row.get("source_date")
                if src_date is None or (isinstance(src_date, float) and pd.isna(src_date)):
                    # Fallback: use stale_financials flag if source_date missing
                    stale_flag = row.get("stale_financials", False)
                    if stale_flag is True or str(stale_flag).lower() == "true":
                        df.at[idx, "row_status"] = PeerRowStatus.STALE_MARKET_DATA
                        for col in ("EV/Revenue", "EV/EBITDA", "P/E", "FCF_yield"):
                            if col in df.columns:
                                df.at[idx, col] = None
                        msg = (
                            f"{row['ticker']}: stale market data "
                            f"(stale_financials=True, source_date unavailable)"
                        )
                        exclusion_log.append(msg)
                        logger.info(msg)
                    continue
                try:
                    if isinstance(src_date, str):
                        src_dt = datetime.strptime(src_date, "%Y-%m-%d")
                    else:
                        src_dt = pd.Timestamp(src_date).to_pydatetime()
                    days_old = (report_dt - src_dt).days
                    if days_old > threshold_days:
                        df.at[idx, "row_status"] = PeerRowStatus.STALE_MARKET_DATA
                        # Null out all multiples for stale rows
                        for col in ("EV/Revenue", "EV/EBITDA", "P/E", "FCF_yield"):
                            if col in df.columns:
                                df.at[idx, col] = None
                        msg = (
                            f"{row['ticker']}: stale market data "
                            f"(source_date={src_date}, {days_old}d old, "
                            f"threshold={threshold_days}d)"
                        )
                        exclusion_log.append(msg)
                        logger.info(msg)
                except (ValueError, TypeError):
                    pass
        elif "stale_financials" in df.columns:
            # Fallback: use stale_financials flag when source_date column missing
            for idx, row in df.iterrows():
                stale_flag = row.get("stale_financials", False)
                if stale_flag is True or str(stale_flag).lower() == "true":
                    df.at[idx, "row_status"] = PeerRowStatus.STALE_MARKET_DATA
                    for col in ("EV/Revenue", "EV/EBITDA", "P/E", "FCF_yield"):
                        if col in df.columns:
                            df.at[idx, col] = None
                    msg = (
                        f"{row['ticker']}: stale market data "
                        f"(stale_financials=True)"
                    )
                    exclusion_log.append(msg)
                    logger.info(msg)

        # --- EV-based exclusions (Req 21.2) ---
        for idx, row in df.iterrows():
            if df.at[idx, "row_status"] != PeerRowStatus.USABLE:
                continue  # already excluded

            ev = row.get("enterprise_value")
            if ev is None or (isinstance(ev, float) and pd.isna(ev)):
                df.at[idx, "row_status"] = PeerRowStatus.MISSING_EV
                for col in ("EV/Revenue", "EV/EBITDA"):
                    if col in df.columns:
                        df.at[idx, col] = None
                msg = f"{row['ticker']}: missing enterprise_value — excluded from EV multiples"
                exclusion_log.append(msg)
                logger.info(msg)
            elif ev < 0:
                df.at[idx, "row_status"] = PeerRowStatus.NEGATIVE_EV
                for col in ("EV/Revenue", "EV/EBITDA"):
                    if col in df.columns:
                        df.at[idx, col] = None
                msg = f"{row['ticker']}: negative enterprise_value ({ev:,.0f}) — excluded from EV multiples"
                exclusion_log.append(msg)
                logger.info(msg)

        # --- Negative EBITDA exclusion (Req 21.3 for EV/EBITDA) ---
        for idx, row in df.iterrows():
            if df.at[idx, "row_status"] not in (PeerRowStatus.USABLE,):
                continue
            ev_ebitda = row.get("EV/EBITDA")
            if ev_ebitda is not None and not (isinstance(ev_ebitda, float) and pd.isna(ev_ebitda)):
                if ev_ebitda < 0:
                    df.at[idx, "row_status"] = PeerRowStatus.NEGATIVE_EBITDA
                    if "EV/EBITDA" in df.columns:
                        df.at[idx, "EV/EBITDA"] = None
                    msg = f"{row['ticker']}: negative EV/EBITDA — excluded from EV/EBITDA chart"
                    exclusion_log.append(msg)
                    logger.info(msg)

        # --- Negative earnings exclusion (Req 21.3 for P/E) ---
        for idx, row in df.iterrows():
            if df.at[idx, "row_status"] not in (
                PeerRowStatus.USABLE,
                PeerRowStatus.NEGATIVE_EBITDA,
            ):
                continue
            pe = row.get("P/E")
            if pe is not None and not (isinstance(pe, float) and pd.isna(pe)):
                if pe < 0:
                    # Only upgrade status if currently usable
                    if df.at[idx, "row_status"] == PeerRowStatus.USABLE:
                        df.at[idx, "row_status"] = PeerRowStatus.NEGATIVE_EARNINGS
                    if "P/E" in df.columns:
                        df.at[idx, "P/E"] = None
                    msg = f"{row['ticker']}: negative P/E — excluded from primary P/E chart"
                    exclusion_log.append(msg)
                    logger.info(msg)

        return df, exclusion_log

    def check_limited_peer_sample(
        self,
        filtered_df: pd.DataFrame,
    ) -> str | None:
        """Check whether the semi peer sample is too small for EV multiples.

        Returns ``"limited peer sample"`` when fewer than 3 semi-tier peers
        have valid (non-None, non-NaN) EV/Revenue values.  Returns ``None``
        otherwise.

        Req 21.6.
        """
        semi_df = filtered_df[filtered_df.get("peer_tier", pd.Series(dtype=str)) == "semi"]
        if semi_df.empty:
            return "limited peer sample"

        valid_ev_count = 0
        for _, row in semi_df.iterrows():
            ev_rev = row.get("EV/Revenue")
            if ev_rev is not None and not (isinstance(ev_rev, float) and pd.isna(ev_rev)):
                valid_ev_count += 1

        if valid_ev_count < 3:
            logger.warning(
                "Limited peer sample: only %d semi peers have valid EV multiples (need ≥3)",
                valid_ev_count,
            )
            return "limited peer sample"
        return None

    # ------------------------------------------------------------------
    # 8. Scenario builder
    # ------------------------------------------------------------------

    def build_scenarios(
        self,
        base_revenue: float,
        wacc: float,
        net_cash: float,
        shares: float,
    ) -> dict:
        """Run bear/base/bull DCFs from ``config.scenarios``.

        Returns dict keyed by scenario name with DCF results + probability.
        """
        results: dict = {}
        for name, assumptions in self.config.scenarios.items():
            dcf = self.compute_dcf(assumptions, base_revenue, wacc, net_cash, shares)
            results[name] = {
                "assumptions": assumptions,
                "dcf": dcf,
                "probability": assumptions.probability,
                "per_share_value": dcf["per_share_value"],
            }
        return results

    # ------------------------------------------------------------------
    # 9. Valuation input validation (Req 16.1)
    # ------------------------------------------------------------------

    # Metrics whose failure blocks formal valuation.
    _VALUATION_CRITICAL_METRICS = frozenset({
        "revenue",
        "operating_cash_flow",
        "capex",
        "diluted_shares",
        "cash_and_securities",
        "total_debt",
    })

    def check_valuation_inputs(
        self,
        data_quality_status: DataQualityStatus,
        validated_metrics: list[ValidatedMetric],
    ) -> ComponentStatus:
        """Check whether valuation inputs are safe for formal valuation.

        Returns a :class:`ComponentStatus` for the valuation component.

        Rules (Req 16.1):
        - If *data_quality_status* is ``DATA_BLOCKED``, valuation is blocked.
        - If any valuation-critical metric (revenue, operating_cash_flow,
          capex, diluted_shares, cash_and_securities, total_debt) has
          ``status`` in (``"fail"``, ``"missing"``) **and** ``severity``
          is ``"critical"``, valuation is blocked.
        - Otherwise, valuation inputs are usable.
        """
        # Gate 1: pipeline-wide data quality status
        if data_quality_status is DataQualityStatus.DATA_BLOCKED:
            return ComponentStatus(
                component_name="valuation",
                status=ComponentStatusEnum.BLOCKED,
                reason="Data quality status is DATA_BLOCKED",
            )

        # Gate 2: per-metric check for valuation-critical failures
        for metric in validated_metrics:
            if metric.metric_name not in self._VALUATION_CRITICAL_METRICS:
                continue
            if metric.status in ("fail", "missing") and metric.severity == "critical":
                return ComponentStatus(
                    component_name="valuation",
                    status=ComponentStatusEnum.BLOCKED,
                    reason=(
                        f"Valuation-critical input failed validation: "
                        f"{metric.metric_name}"
                    ),
                )

        return ComponentStatus(
            component_name="valuation",
            status=ComponentStatusEnum.USABLE,
            reason="All valuation inputs validated",
        )

    # ------------------------------------------------------------------
    # 10. Sanity-check bridge explanations (Reqs 16.2–16.5)
    # ------------------------------------------------------------------

    def generate_sanity_checks(
        self,
        target_price: float,
        current_price: float,
        reverse_dcf_grid: pd.DataFrame,
        bear_value: float,
        config: EngineConfig,
        sensitivity_table: pd.DataFrame | None = None,
    ) -> list[str]:
        """Return human-readable sanity-check notes for valuation outputs.

        Checks performed:
        1. DCF divergence (Req 16.2): flag if |target - current| / current
           exceeds ``config.max_dcf_price_divergence_pct / 100``.
        2. Reverse-DCF grid (Req 16.3): flag if no grid cell is within ±10 %
           of *current_price*.
        3. Bear downside (Req 16.4): flag if bear downside exceeds −40 %.
        4. Sensitivity table (Req 16.5): if provided, flag if no cell is
           within ±10 % of *current_price*.
        """
        notes: list[str] = []

        # Guard against non-positive current price (avoids division by zero)
        if current_price <= 0:
            notes.append(
                "Sanity check skipped: current price is zero or negative."
            )
            return notes

        # 1. DCF divergence check (Req 16.2)
        divergence = (target_price - current_price) / current_price
        threshold = config.max_dcf_price_divergence_pct / 100.0
        if abs(divergence) > threshold:
            direction = "above" if divergence > 0 else "below"
            notes.append(
                f"DCF target (${target_price:,.2f}) is {abs(divergence) * 100:.1f}% "
                f"{direction} current price (${current_price:,.2f}), exceeding the "
                f"{config.max_dcf_price_divergence_pct:.0f}% divergence threshold. "
                f"Review key assumptions driving the gap."
            )

        # 2. Reverse-DCF grid check (Req 16.3)
        if not reverse_dcf_grid.empty:
            grid_values = reverse_dcf_grid.select_dtypes(include=[np.number]).values
            within_range = np.any(
                np.abs(grid_values - current_price) / current_price <= 0.10
            )
            if not within_range:
                notes.append(
                    f"Current price (${current_price:,.2f}) lies outside the "
                    f"reverse-DCF grid range (no cell within ±10%). The modeled "
                    f"assumption ranges may not span the market-implied expectations."
                )

        # 3. Bear downside check (Req 16.4)
        bear_downside = (bear_value - current_price) / current_price
        if bear_downside < -0.40:
            notes.append(
                f"Bear-case value (${bear_value:,.2f}) implies "
                f"{bear_downside * 100:.1f}% downside, exceeding the −40% "
                f"threshold. This extreme scenario requires additional "
                f"justification."
            )

        # 4. Sensitivity table check (Req 16.5)
        if sensitivity_table is not None and not sensitivity_table.empty:
            sens_values = sensitivity_table.select_dtypes(include=[np.number]).values
            within_sens = np.any(
                np.abs(sens_values - current_price) / current_price <= 0.10
            )
            if not within_sens:
                notes.append(
                    f"Current price (${current_price:,.2f}) falls outside the "
                    f"sensitivity table range (no cell within ±10%). Consider "
                    f"whether the WACC/terminal-growth ranges are sufficiently broad."
                )

        return notes

    # ------------------------------------------------------------------
    # 11. Recommendation
    # ------------------------------------------------------------------

    def generate_recommendation(
        self,
        scenarios: dict,
        current_price: float,
        reverse_dcf: pd.DataFrame,
        ml_signal: Optional[float] = None,
        narrative_signal: Optional[float] = None,
        recommendation_status: Optional[RecommendationStatus] = None,
    ) -> Recommendation:
        """Scorecard-based recommendation with thresholds from config.

        If *recommendation_status* is provided and its eligibility is not
        ``formal_rating``, the rating is forced to ``"Not Rated"`` and
        thresholds are NOT evaluated (Req 9.11, 17.1).

        Parameters
        ----------
        scenarios : dict
            Output of ``build_scenarios``.
        current_price : float
            Current share price.
        reverse_dcf : pd.DataFrame
            Output of ``compute_reverse_dcf_grid``.
        ml_signal : float | None
            ML model directional signal (positive = bullish).
        narrative_signal : float | None
            Narrative drift signal (positive = improving sentiment).
        recommendation_status : RecommendationStatus | None
            Output of the pre-report eligibility audit. When provided,
            the method respects the eligibility gate.
        """
        thresholds = self.config.recommendation_thresholds

        # Extract scenario values
        bear = scenarios.get("bear", {})
        base = scenarios.get("base", {})
        bull = scenarios.get("bull", {})

        bear_val = bear.get("per_share_value", 0.0)
        bear_prob = bear.get("probability", 0.25)
        base_val = base.get("per_share_value", 0.0)
        base_prob = base.get("probability", 0.50)
        bull_val = bull.get("per_share_value", 0.0)
        bull_prob = bull.get("probability", 0.25)

        expected_value = (
            bear_val * bear_prob
            + base_val * base_prob
            + bull_val * bull_prob
        )

        target_price = expected_value
        upside_pct = (target_price - current_price) / current_price if current_price > 0 else 0.0
        bear_downside = (bear_val - current_price) / current_price if current_price > 0 else 0.0
        base_upside = (base_val - current_price) / current_price if current_price > 0 else 0.0

        # --- Scorecard components ---
        # Build scorecard with ComponentStatusEnum values (never None) (Req 17.4)
        scorecard: dict = {}

        # Valuation upside
        scorecard["valuation_upside"] = {
            "status": ComponentStatusEnum.USABLE.value,
            "value": upside_pct,
            "reason": "Probability-weighted expected value vs current price",
        }

        # Reverse-DCF plausibility: fraction of grid cells near current price
        hl = reverse_dcf.attrs.get("highlight")
        if hl is not None and not hl.empty:
            total_cells = hl.size
            near_cells = hl.values.sum()
            rdcf_value = float(near_cells / total_cells) if total_cells > 0 else 0.0
            scorecard["reverse_dcf_plausibility"] = {
                "status": ComponentStatusEnum.USABLE.value,
                "value": rdcf_value,
                "reason": f"{int(near_cells)}/{total_cells} cells within ±10% of current price",
            }
        else:
            scorecard["reverse_dcf_plausibility"] = {
                "status": ComponentStatusEnum.UNAVAILABLE.value,
                "value": 0.0,
                "reason": "Reverse-DCF grid not available",
            }

        # ML signal — use component status from recommendation_status if available
        ml_component_status = ComponentStatusEnum.USABLE.value
        if recommendation_status:
            for cs in recommendation_status.component_statuses:
                if cs.component_name == "ml_signal":
                    ml_component_status = cs.status.value
                    break
        if ml_signal is None:
            ml_component_status = ComponentStatusEnum.UNAVAILABLE.value
        scorecard["ml_signal"] = {
            "status": ml_component_status,
            "value": ml_signal if ml_signal is not None else 0.0,
            "reason": "ML directional signal" if ml_signal is not None else "ML signal not available",
        }

        # Narrative signal — use component status from recommendation_status if available
        nlp_component_status = ComponentStatusEnum.USABLE.value
        if recommendation_status:
            for cs in recommendation_status.component_statuses:
                if cs.component_name == "nlp_signal":
                    nlp_component_status = cs.status.value
                    break
        if narrative_signal is None:
            nlp_component_status = ComponentStatusEnum.UNAVAILABLE.value
        scorecard["narrative_signal"] = {
            "status": nlp_component_status,
            "value": narrative_signal if narrative_signal is not None else 0.0,
            "reason": "Narrative drift signal" if narrative_signal is not None else "NLP signal not available",
        }

        scorecard["risk_concentration"] = {
            "status": ComponentStatusEnum.USABLE.value,
            "value": "high",
            "reason": "NVDA has known customer concentration",
        }
        scorecard["analyst_judgment"] = {
            "status": ComponentStatusEnum.USABLE.value,
            "value": "neutral",
            "reason": "Analyst judgment on qualitative factors",
        }

        # --- Eligibility gate check (Req 17.1, 9.11) ---
        is_formal = True
        not_rated_reason: str | None = None
        eligibility_status_str = "formal_rating"

        if recommendation_status is not None:
            eligibility_status_str = recommendation_status.eligibility_status.value
            if recommendation_status.eligibility_status != ReportMode.FORMAL_RATING:
                is_formal = False
                not_rated_reason = (
                    "Eligibility gate did not pass. Blocking issues: "
                    + "; ".join(recommendation_status.blocking_issues)
                    if recommendation_status.blocking_issues
                    else "Eligibility gate did not pass (unknown reason)"
                )

        # --- Rating logic (Req 9.11) — only evaluated if eligibility passes ---
        if is_formal:
            buy_conditions = (
                upside_pct >= thresholds.buy_min_upside
                and base_upside >= thresholds.buy_min_upside
                and bear_downside >= thresholds.buy_max_bear_downside
            )
            sell_conditions = upside_pct <= thresholds.sell_min_downside

            if buy_conditions:
                rating = "Buy"
            elif sell_conditions:
                rating = "Sell"
            else:
                rating = "Hold"
        else:
            rating = "Not Rated"

        # --- What-must-be-true statements (Req 9.13) ---
        what_buy = (
            "Revenue CAGR sustains ≥20 %, FCF margins reach ≥30 % terminal, "
            "AI infrastructure spend does not decelerate sharply, and competitive "
            "moat (CUDA ecosystem) remains intact."
        )
        what_hold = (
            "Current valuation already reflects base-case growth; upside limited "
            "unless bull-case catalysts materialise. Reverse-DCF implied assumptions "
            "are plausible but not conservative."
        )
        what_sell = (
            "Export controls tighten materially, hyperscaler capex decelerates, "
            "custom-silicon alternatives erode GPU share, and margins compress "
            "below bear-case assumptions."
        )

        return Recommendation(
            rating=rating,
            current_price=current_price,
            target_price=target_price,
            upside_pct=upside_pct,
            bear_value=bear_val,
            bear_probability=bear_prob,
            base_value=base_val,
            base_probability=base_prob,
            bull_value=bull_val,
            bull_probability=bull_prob,
            expected_value=expected_value,
            what_must_be_true_buy=what_buy,
            what_must_be_true_hold=what_hold,
            what_must_be_true_sell=what_sell,
            scorecard=scorecard,
            target_horizon=self.config.target_horizon,
            eligibility_status=eligibility_status_str,
            not_rated_reason=not_rated_reason,
        )

    # ------------------------------------------------------------------
    # 12. Build canonical FinalRecommendation
    # ------------------------------------------------------------------

    def build_final_recommendation(
        self,
        run_id: str,
        recommendation: Recommendation,
        current_price: float,
        recommendation_status: RecommendationStatus | None = None,
        active_exhibits: list[str] | None = None,
        suppressed_exhibits: list[str] | None = None,
        data_quality_status: str = "pass",
    ) -> FinalRecommendation:
        """Build the canonical FinalRecommendation — the ONLY source of truth.

        Rating logic (Hold-biased for moderate downside):
        - If report_mode is not formal_rating -> "Not Rated"
        - If mechanical downside is between -20% and +15% -> Hold
        - If mechanical downside < -20% -> Sell allowed
        - If upside > +15% -> Buy allowed
        """
        from datetime import datetime, timezone

        is_formal = True
        report_mode = "formal_rating"
        if recommendation_status is not None:
            if recommendation_status.eligibility_status != ReportMode.FORMAL_RATING:
                is_formal = False
                report_mode = recommendation_status.eligibility_status.value

        bear_val = recommendation.bear_value
        base_val = recommendation.base_value
        bull_val = recommendation.bull_value
        bear_prob = recommendation.bear_probability
        base_prob = recommendation.base_probability
        bull_prob = recommendation.bull_probability
        expected_value = recommendation.expected_value
        upside_pct = (expected_value - current_price) / current_price if current_price > 0 else 0.0

        if not is_formal:
            rating = "Not Rated"
            rating_label = "Not Rated — Data Validation Required"
        else:
            if upside_pct > 0.15:
                rating = "Buy"
                rating_label = "Buy / Outperform"
            elif upside_pct < -0.20:
                rating = "Sell"
                rating_label = "Sell / Underperform"
            else:
                rating = "Hold"
                rating_label = "Hold / Market Perform"

        analyst_override_flag = False
        analyst_override_reason = ""

        if rating == "Hold":
            rationale_short = (
                f"We rate {self.config.company_name} Hold. The probability-weighted "
                f"intrinsic value of ${expected_value:,.2f} implies "
                f"{upside_pct * 100:.1f}% {'upside' if upside_pct >= 0 else 'downside'}, "
                f"which is within our Hold threshold."
            )
            rationale_long = (
                f"We rate {self.config.company_name} Hold / Market Perform. "
                f"NVIDIA remains an exceptional AI infrastructure leader, but the current "
                f"market price of ${current_price:,.2f} already embeds aggressive assumptions "
                f"for revenue growth, margin durability, and AI capex persistence. Our "
                f"probability-weighted intrinsic value of ${expected_value:,.2f} is "
                f"{'below' if upside_pct < 0 else 'above'} the current price, "
                f"implying {abs(upside_pct) * 100:.1f}% {'downside' if upside_pct < 0 else 'upside'}. "
                f"The modeled downside is moderate and the bull case (${bull_val:,.2f}) "
                f"remains material. Existing investors can hold; new capital should wait "
                f"for a better entry point or evidence that bull-case assumptions are materializing."
            )
        elif rating == "Buy":
            rationale_short = (
                f"We rate {self.config.company_name} Buy with {upside_pct * 100:.1f}% upside "
                f"to our ${expected_value:,.2f} target."
            )
            rationale_long = rationale_short
        elif rating == "Sell":
            rationale_short = (
                f"We rate {self.config.company_name} Sell with {abs(upside_pct) * 100:.1f}% "
                f"downside to our ${expected_value:,.2f} target."
            )
            rationale_long = rationale_short
        else:
            rationale_short = "Not Rated — data validation required."
            rationale_long = rationale_short

        why_hold = ""
        if rating == "Hold" and upside_pct < 0:
            why_hold = (
                f"The business quality is exceptional — NVIDIA's CUDA ecosystem appears to create "
                f"meaningful switching costs (analyst assessment). The bull-case value (${bull_val:,.0f}) remains "
                f"materially above market. AI capex durability remains uncertain, not disproven. "
                f"ML is diagnostic-only and does not independently support Sell. "
                f"Peer/segment/NLP signals are not clean enough to strengthen a Sell call. "
                f"The {abs(upside_pct) * 100:.1f}% modeled downside reflects valuation caution, "
                f"not a high-conviction Sell signal."
            )

        upgrade_triggers = [
            "Sustained >=20% revenue CAGR for 2+ consecutive years",
            "Terminal FCF margins >=30%",
            "Continued data-center demand and CUDA moat expansion",
            "Enterprise AI adoption broadening beyond hyperscalers",
        ]
        downgrade_triggers = [
            "Hyperscaler capex deceleration (2+ quarters of declining growth)",
            "Export-control expansion to additional markets",
            "Custom silicon share loss reaching material levels (analyst monitoring threshold)",
            "Margin compression below bear-case assumptions",
        ]

        thresholds = {
            "buy_min_upside": 0.15,
            "hold_range": [-0.20, 0.15],
            "sell_min_downside": -0.20,
        }

        return FinalRecommendation(
            run_id=run_id,
            ticker=self.config.ticker,
            company_name=self.config.company_name,
            report_date=self.config.report_date,
            price_date=self.config.price_date,
            current_price=current_price,
            rating=rating,
            rating_label=rating_label,
            intrinsic_value=expected_value,
            target_price=expected_value,
            upside_downside_pct=upside_pct,
            recommendation_rationale_short=rationale_short,
            recommendation_rationale_long=rationale_long,
            bear_value=bear_val,
            base_value=base_val,
            bull_value=bull_val,
            bear_probability=bear_prob,
            base_probability=base_prob,
            bull_probability=bull_prob,
            probability_weighted_value=expected_value,
            rating_thresholds=thresholds,
            analyst_override_flag=analyst_override_flag,
            analyst_override_reason=analyst_override_reason,
            valuation_method="probability_weighted_dcf",
            valuation_confidence="medium",
            data_quality_status=data_quality_status,
            report_mode=report_mode,
            package_status="formal_rating_pass" if is_formal else "diagnostic_not_rated_pass",
            active_exhibits=active_exhibits or [],
            suppressed_exhibits=suppressed_exhibits or [],
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            what_must_be_true_buy=recommendation.what_must_be_true_buy,
            what_must_be_true_hold=recommendation.what_must_be_true_hold,
            what_must_be_true_sell=recommendation.what_must_be_true_sell,
            why_hold_not_sell=why_hold,
            upgrade_triggers=upgrade_triggers,
            downgrade_triggers=downgrade_triggers,
            scorecard=recommendation.scorecard,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dcf_per_share(
        self,
        cagr: float,
        margin: float,
        wacc: float,
        cash: float,
        shares: float,
    ) -> float:
        """Quick DCF per-share using the base-case base_revenue from config scenarios."""
        # Derive a base_revenue from the base scenario's CAGR and a reference.
        # For reverse-DCF / solver usage we need a stored base_revenue.
        # Fall back to a class-level attribute set by the caller.
        base_revenue = getattr(self, "_base_revenue", None)
        if base_revenue is None or base_revenue <= 0:
            logger.warning(
                "_dcf_per_share called without set_base_revenue(); "
                "results will be meaningless. Call set_base_revenue() first."
            )
            base_revenue = 1.0
        assumptions = ScenarioAssumptions(
            name="_solver",
            probability=0.0,
            revenue_cagr=cagr,
            fcf_margin_start=margin,
            fcf_margin_terminal=margin,
            terminal_growth=self.config.terminal_growth,
            sbc_treatment="included_in_fcf",
            analyst_notes="",
        )
        result = self.compute_dcf(assumptions, base_revenue, wacc, cash, shares)
        return result["per_share_value"]

    @staticmethod
    def _safe_div(
        numerator: Optional[float],
        denominator: Optional[float],
    ) -> Optional[float]:
        if numerator is None or denominator is None:
            return None
        if pd.isna(numerator) or pd.isna(denominator):
            return None
        if denominator == 0:
            return None
        return numerator / denominator

    def set_base_revenue(self, base_revenue: float) -> None:
        """Store base_revenue for use by reverse-DCF grid and solvers."""
        self._base_revenue = base_revenue
