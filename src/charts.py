"""
NVDA Quantamental Engine — Chart Generation.

Produces 6-8 decision-useful exhibits for the equity research report.
All figures saved to outputs/figures/ with source captions and
ExhibitRecord metadata.  Uses matplotlib Agg backend for non-interactive
rendering.  (Req 10.5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import ChartStatus, ComponentStatusEnum, EngineConfig, ExhibitRecord, RecommendationStatus, get_default_config  # noqa: E402

# ---------------------------------------------------------------------------
# Shared style defaults
# ---------------------------------------------------------------------------
_STYLE_DEFAULTS = {
    "figure.figsize": (10, 6),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}


class ChartGenerator:
    """Generates all required report exhibits."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or get_default_config()
        self.figures_dir = self.config.outputs_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.exhibits: list[ExhibitRecord] = []
        self._exhibit_counter: int = 0
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(_STYLE_DEFAULTS)

    def _next_exhibit_id(self) -> str:
        self._exhibit_counter += 1
        return f"EX-{self._exhibit_counter:02d}"

    # ------------------------------------------------------------------
    # Helper: save with attribution
    # ------------------------------------------------------------------
    def _save_with_attribution(
        self,
        fig: plt.Figure,
        filename: str,
        source_caption: str,
        title: str = "",
        data_source: str = "",
        date_range: str = "",
        exhibit_key: str = "",
    ) -> Path:
        """Save *fig* to outputs/figures/<filename> with a source caption.

        Adds a small footnote to the figure, writes the PNG, closes the
        figure, and appends an :class:`ExhibitRecord` to ``self.exhibits``.

        Returns the saved file path.
        """
        # Add source caption as footnote
        fig.text(
            0.5, 0.01, source_caption,
            ha="center", fontsize=7, style="italic", color="gray",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        out_path = self.figures_dir / filename
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        # Derive exhibit_key from filename if not provided
        if not exhibit_key:
            exhibit_key = filename.replace(".png", "")

        record = ExhibitRecord(
            exhibit_id=self._next_exhibit_id(),
            title=title or filename.replace(".png", "").replace("_", " ").title(),
            source_caption=source_caption,
            data_source=data_source,
            date_range=date_range,
            file_path=str(out_path),
            exhibit_key=exhibit_key,
        )
        self.exhibits.append(record)
        return out_path

    # ------------------------------------------------------------------
    # 1. Revenue / Segment Mix  (stacked bar)
    # ------------------------------------------------------------------
    def plot_revenue_segment_mix(
        self,
        segment_data: pd.DataFrame,
        chart_status: Optional[ChartStatus] = None,
    ) -> Optional[Path]:
        """Stacked bar chart of normalised segment revenue over time.

        *segment_data* should follow the ``nvda_segment_revenue_normalized.csv``
        schema (columns include ``fiscal_period``, ``normalized_category``,
        ``value``).  The chart labels whether the basis is reportable
        segments or market/platform revenue.

        If *chart_status* is provided and ``renderable`` is ``False``,
        the chart is suppressed and a fallback placeholder is shown instead.
        """
        # --- Chart gating: suppress if not renderable ---
        if chart_status is not None and not chart_status.renderable:
            fallback = chart_status.fallback_message or f"Chart suppressed: {chart_status.reason}"
            return self._empty_chart("revenue_segment_mix.png", fallback)
        if segment_data.empty:
            return self._empty_chart("revenue_segment_mix.png", "No segment data available")

        df = segment_data.copy()

        # Build proper period labels if fiscal_year column exists
        # (actual CSV has separate fiscal_year + fiscal_period columns)
        if "fiscal_year" in df.columns and not df["fiscal_period"].str.startswith("FY").all():
            df["period_label"] = df.apply(
                lambda r: f"FY{r['fiscal_year']}" if r["fiscal_period"] == "FY"
                else f"FY{r['fiscal_year']}-{r['fiscal_period']}",
                axis=1,
            )
        else:
            # Already has combined labels like "FY2024", "FY2024-Q1"
            df["period_label"] = df["fiscal_period"]

        # Filter to annual periods from FY2018 for a clean chart
        annual = df[df["period_label"].str.match(r"^FY\d{4}$")]
        annual = annual[annual["period_label"] >= "FY2018"]

        if annual.empty:
            return self._empty_chart("revenue_segment_mix.png", "No segment data for FY2018+")

        pivot = annual.pivot_table(
            index="period_label",
            columns="normalized_category",
            values="value",
            aggfunc="sum",
        ).fillna(0)

        # Sort periods chronologically
        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_title("NVDA Revenue by Segment / Platform")
        ax.set_ylabel("Revenue (USD)")
        ax.set_xlabel("Fiscal Period")
        ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")

        # Determine basis label
        methods = segment_data["extraction_method"].unique() if "extraction_method" in segment_data.columns else []
        basis = "Reportable Segments" if "xbrl_dimension" in list(methods) else "Market / Platform Revenue"
        caption = f"Source: NVIDIA SEC filings (XBRL + text). Basis: {basis}."

        return self._save_with_attribution(
            fig, "revenue_segment_mix.png", caption,
            title="Revenue Segment / Platform Mix",
            data_source="SEC EDGAR XBRL / filing text",
            date_range=f"{pivot.index.min()} – {pivot.index.max()}" if len(pivot) else "",
        )

    # ------------------------------------------------------------------
    # 2. Margin Trends  (line chart)
    # ------------------------------------------------------------------
    def plot_margin_trends(self, metrics: pd.DataFrame) -> Path:
        """Line chart of gross, operating, and net margin over time.

        Shows quarterly periods starting from FY2018-Q1.

        *metrics* follows the ``nvda_metrics.csv`` long-format schema
        (columns: ``fiscal_period``, ``metric_name``, ``metric_value``).
        """
        margin_names = ["gross_margin", "operating_margin", "net_margin"]
        df = metrics[metrics["metric_name"].isin(margin_names)].copy()

        # Filter to quarterly periods only, starting from FY2018-Q1
        df = df[df["fiscal_period"].str.match(r"^FY\d{4}-Q\d$")]
        df = df[df["fiscal_period"] >= "FY2018-Q1"]

        if df.empty:
            return self._empty_chart("margin_trends.png", "No margin data available")

        pivot = df.pivot_table(
            index="fiscal_period", columns="metric_name",
            values="metric_value", aggfunc="first",
        ).sort_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        for col in margin_names:
            if col in pivot.columns:
                ax.plot(pivot.index, pivot[col], marker="o", label=col.replace("_", " ").title())

        ax.set_title("NVDA Margin Trends")
        ax.set_ylabel("Margin (%)" if (pivot.max().max() <= 1.0) else "Margin")
        ax.set_xlabel("Fiscal Period")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        caption = "Source: NVIDIA SEC filings (XBRL). Computed metrics."
        return self._save_with_attribution(
            fig, "margin_trends.png", caption,
            title="Margin Trends (Gross / Operating / Net)",
            data_source="SEC EDGAR XBRL",
            date_range=f"{pivot.index.min()} – {pivot.index.max()}" if len(pivot) else "",
        )

    # ------------------------------------------------------------------
    # 3. FCF Trend  (bar + line combo)
    # ------------------------------------------------------------------
    def plot_fcf_trend(self, metrics: pd.DataFrame) -> Path:
        """Bar chart of FCF with an overlaid FCF-margin line.

        Shows quarterly periods (Q1/Q2/Q3) starting from FY2018-Q1.
        Annual (FY) periods are excluded to avoid mixing full-year totals
        with quarterly YTD values.

        *metrics* follows the ``nvda_metrics.csv`` long-format schema.
        """
        fcf_df = metrics[metrics["metric_name"] == "FCF"].copy()
        margin_df = metrics[metrics["metric_name"] == "FCF_margin"].copy()

        # Filter to quarterly periods only (FYnnnn-Qn), exclude annual
        fcf_df = fcf_df[fcf_df["fiscal_period"].str.match(r"^FY\d{4}-Q\d$")]
        margin_df = margin_df[margin_df["fiscal_period"].str.match(r"^FY\d{4}-Q\d$")]

        # Start from FY2018-Q1
        fcf_df = fcf_df[fcf_df["fiscal_period"] >= "FY2018-Q1"]
        margin_df = margin_df[margin_df["fiscal_period"] >= "FY2018-Q1"]

        # Drop NaN values
        fcf_df = fcf_df.dropna(subset=["metric_value"])

        if fcf_df.empty:
            return self._empty_chart("fcf_trend.png", "No FCF data available")

        fcf_df = fcf_df.sort_values("fiscal_period")
        margin_df = margin_df.sort_values("fiscal_period")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bars for FCF
        x = np.arange(len(fcf_df))
        ax1.bar(x, fcf_df["metric_value"].values, color="steelblue", alpha=0.7, label="FCF")
        ax1.set_xticks(x)
        ax1.set_xticklabels(fcf_df["fiscal_period"].values, rotation=45, ha="right")
        ax1.set_ylabel("Free Cash Flow (USD)")
        ax1.set_xlabel("Fiscal Period")
        ax1.set_title("NVDA Free Cash Flow & FCF Margin")

        # Line for FCF margin on secondary axis
        if not margin_df.empty:
            ax2 = ax1.twinx()
            # Align margin to same x positions by fiscal_period
            merged = fcf_df[["fiscal_period"]].merge(
                margin_df[["fiscal_period", "metric_value"]],
                on="fiscal_period", how="left",
            )
            ax2.plot(x, merged["metric_value"].values, color="darkorange",
                     marker="s", linewidth=2, label="FCF Margin")
            ax2.set_ylabel("FCF Margin")
            ax2.legend(loc="upper left")

        ax1.legend(loc="upper right")

        caption = "Source: NVIDIA SEC filings (XBRL). FCF = Operating CF − CapEx."
        return self._save_with_attribution(
            fig, "fcf_trend.png", caption,
            title="Free Cash Flow & FCF Margin Trend",
            data_source="SEC EDGAR XBRL",
            date_range=f"{fcf_df['fiscal_period'].iloc[0]} – {fcf_df['fiscal_period'].iloc[-1]}" if len(fcf_df) else "",
        )

    # ------------------------------------------------------------------
    # 4. Narrative Drift  (line chart)
    # ------------------------------------------------------------------
    def plot_narrative_drift(self, nlp: pd.DataFrame) -> Path:
        """Line chart of TF-IDF similarity scores over time.

        *nlp* follows the ``nvda_nlp_features.csv`` schema (columns:
        ``filing_date``, ``section``, ``feature_type``, ``feature_name``,
        ``value``).  Plots ``tfidf_similarity`` for risk_factors and mda.
        """
        sim = nlp[nlp["feature_type"] == "tfidf_similarity"].copy()

        if sim.empty:
            return self._empty_chart("narrative_drift.png", "No TF-IDF similarity data available")

        sim = sim.sort_values("filing_date")

        fig, ax = plt.subplots(figsize=(10, 6))
        for section, grp in sim.groupby("section"):
            ax.plot(grp["filing_date"], grp["value"], marker="o", label=section.replace("_", " ").title())

        ax.set_title("NVDA Narrative Drift (TF-IDF Cosine Similarity)")
        ax.set_ylabel("Cosine Similarity (0 = very different, 1 = identical)")
        ax.set_xlabel("Filing Date")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        caption = "Source: NVIDIA SEC filings (Risk Factors & MD&A). TF-IDF cosine similarity between consecutive filings."
        return self._save_with_attribution(
            fig, "narrative_drift.png", caption,
            title="Narrative Drift — TF-IDF Similarity",
            data_source="SEC EDGAR filing text",
            date_range=f"{sim['filing_date'].iloc[0]} – {sim['filing_date'].iloc[-1]}" if len(sim) else "",
        )

    # ------------------------------------------------------------------
    # 5. DCF Scenarios  (bar chart)
    # ------------------------------------------------------------------
    def plot_dcf_scenarios(
        self,
        scenarios: dict,
        diagnostic_label: Optional[str] = None,
    ) -> Path:
        """Bar chart of bear / base / bull per-share values with probability weights.

        *scenarios* is a dict keyed by scenario name (``bear``, ``base``,
        ``bull``), each containing at least ``per_share_value`` and
        ``probability``.

        If *diagnostic_label* is provided, it is displayed as a subtitle
        annotation on the chart (e.g. "Diagnostic only — do not use for
        recommendation").
        """
        if not scenarios:
            return self._empty_chart("dcf_scenarios.png", "No scenario data available")

        names = []
        values = []
        probs = []
        colors = {"bear": "#d9534f", "base": "#5bc0de", "bull": "#5cb85c"}

        for key in ["bear", "base", "bull"]:
            if key in scenarios:
                s = scenarios[key]
                names.append(key.title())
                val = s.get("per_share_value", s.get("equity_per_share", 0))
                values.append(val)
                probs.append(s.get("probability", 0))

        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = [colors.get(n.lower(), "gray") for n in names]
        bars = ax.bar(names, values, color=bar_colors, edgecolor="black", alpha=0.85)

        # Annotate with value and probability
        for bar, val, prob in zip(bars, values, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"${val:,.0f}\n({prob:.0%})", ha="center", va="bottom", fontsize=10,
            )

        ax.set_title("NVDA DCF Scenario Valuations")
        ax.set_ylabel("Per-Share Value (USD)")

        # Add diagnostic label if provided
        if diagnostic_label:
            ax.text(
                0.5, -0.12, f"⚠️ {diagnostic_label}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="red", style="italic",
            )

        caption = "Source: Analyst DCF model. Probability weights shown in parentheses."
        return self._save_with_attribution(
            fig, "dcf_scenarios.png", caption,
            title="DCF Scenario Valuations (Bear / Base / Bull)",
            data_source="Analyst DCF model",
        )

    # ------------------------------------------------------------------
    # 6. Reverse-DCF Grid  (heatmap)
    # ------------------------------------------------------------------
    def plot_reverse_dcf_grid(
        self,
        grid: pd.DataFrame,
        current_price: float = 0.0,
        grid_warning: Optional[str] = None,
    ) -> Path:
        """Heatmap of the reverse-DCF grid (CAGR × margin → implied price).

        *grid* is a DataFrame with revenue-CAGR values as the index and
        terminal-FCF-margin values as columns; cell values are implied
        per-share prices.

        If *current_price* is provided and falls outside the grid range,
        an annotation is added. If *grid_warning* is provided, it is
        displayed as a footnote.
        """
        if grid.empty:
            return self._empty_chart("reverse_dcf_grid.png", "No reverse-DCF grid data")

        fig, ax = plt.subplots(figsize=(10, 7))

        # Index/columns may be strings like "cagr=0.10" or raw floats.
        # Parse numeric values for display formatting.
        def _parse_label(label: str) -> str:
            """Extract numeric part from 'key=value' labels and format as %."""
            s = str(label)
            if "=" in s:
                s = s.split("=", 1)[1]
            try:
                v = float(s)
                return f"{v:.0%}"
            except (ValueError, TypeError):
                return str(label)

        fmt_idx = [_parse_label(v) for v in grid.index]
        fmt_col = [_parse_label(v) for v in grid.columns]

        sns.heatmap(
            grid.values.astype(float),
            annot=True, fmt=",.0f",
            xticklabels=fmt_col,
            yticklabels=fmt_idx,
            cmap="RdYlGn", center=float(grid.values.mean()),
            linewidths=0.5, ax=ax,
        )
        ax.set_title("Reverse-DCF Implied Share Price Grid")
        ax.set_xlabel("Terminal FCF Margin")
        ax.set_ylabel("Revenue CAGR")

        # Annotate when current price is outside grid
        if current_price > 0:
            grid_min = float(grid.values.min())
            grid_max = float(grid.values.max())
            if current_price < grid_min or current_price > grid_max:
                ax.text(
                    0.5, -0.10,
                    f"⚠️ Current price (${current_price:,.0f}) is outside grid range "
                    f"(${grid_min:,.0f} – ${grid_max:,.0f})",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color="red", style="italic",
                )

        if grid_warning:
            ax.text(
                0.5, -0.15 if current_price > 0 else -0.10,
                grid_warning,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="orange", style="italic",
            )

        caption = "Source: Analyst reverse-DCF model. Cell values = implied per-share price (USD)."
        return self._save_with_attribution(
            fig, "reverse_dcf_grid.png", caption,
            title="Reverse-DCF Implied Price Grid (CAGR × Margin)",
            data_source="Analyst reverse-DCF model",
        )

    # ------------------------------------------------------------------
    # 7. Recommendation Scorecard  (status matrix, not numeric bars for diagnostic)
    # ------------------------------------------------------------------
    def plot_recommendation_scorecard(
        self,
        scorecard: dict,
        recommendation_status: Optional[RecommendationStatus] = None,
    ) -> Path:
        """Scorecard visualization.

        Diagnostic-only, blocked, and unavailable components are NEVER
        shown as numeric bars. They appear only in a status annotation
        table below the chart. This prevents diagnostic ML/NLP from
        visually appearing to support the rating.
        """
        if not scorecard:
            return self._empty_chart("recommendation_scorecard.png", "No scorecard data")

        # Build status lookup from recommendation_status
        status_lookup: dict[str, str] = {}
        if recommendation_status is not None:
            for cs in recommendation_status.component_statuses:
                status_lookup[cs.component_name] = cs.status.value

        # Separate usable components (get bars) from non-usable (status only)
        bar_components = []
        bar_scores = []
        bar_statuses = []
        excluded_components: list[tuple[str, str, str]] = []  # (name, status, reason)

        for k, v in scorecard.items():
            if isinstance(v, dict):
                raw_val = v.get("value", v)
                comp_status = v.get("status", status_lookup.get(k, "usable"))
                reason = v.get("reason", "")
            else:
                raw_val = v
                comp_status = status_lookup.get(k, "usable")
                reason = ""

            # Diagnostic-only, blocked, unavailable → exclude from bars
            if comp_status in ("diagnostic_only", "blocked", "unavailable"):
                excluded_components.append((k, comp_status, reason))
                continue

            try:
                bar_scores.append(float(raw_val))
                bar_components.append(k)
                bar_statuses.append(comp_status)
            except (TypeError, ValueError):
                excluded_components.append((k, comp_status, f"non-numeric: {raw_val}"))

        if not bar_components and not excluded_components:
            return self._empty_chart("recommendation_scorecard.png", "No scorecard data")

        fig, ax = plt.subplots(figsize=(9, max(4, len(bar_components) * 0.6 + len(excluded_components) * 0.3 + 2)))

        if bar_components:
            labels = [k.replace("_", " ").title() for k in bar_components]
            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, bar_scores, color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Score")
            ax.invert_yaxis()
            for bar, score in zip(bars, bar_scores):
                ax.text(bar.get_width() + max(bar_scores) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{score:.2f}", va="center", fontsize=9)
        else:
            ax.text(0.5, 0.7, "No usable numeric components", ha="center",
                    va="center", fontsize=11, color="gray", transform=ax.transAxes)
            ax.set_axis_off()

        ax.set_title("NVDA Recommendation Scorecard")

        # Add excluded components as text annotation
        if excluded_components:
            excl_text = "Excluded from rating (diagnostic/blocked):\n"
            for name, status, reason in excluded_components:
                label = name.replace("_", " ").title()
                excl_text += f"  • {label}: [{status}] {reason}\n"
            fig.text(0.5, 0.01, excl_text, ha="center", fontsize=7,
                     style="italic", color="#666", family="monospace")

        caption = "Source: Analyst scorecard. Diagnostic-only components excluded from bars."
        return self._save_with_attribution(
            fig, "recommendation_scorecard.png", caption,
            title="Recommendation Scorecard",
            data_source="Analyst judgment + quantitative signals",
        )

    # ------------------------------------------------------------------
    # 8. Revenue Growth Trend  (bar chart)  [optional]
    # ------------------------------------------------------------------
    def plot_revenue_growth_trend(self, metrics: pd.DataFrame) -> Path:
        """Bar chart of YoY revenue growth with a 0% reference line.

        Uses annual (FY) periods starting from FY2018 since YoY growth
        is computed on an annual basis.

        *metrics* follows the ``nvda_metrics.csv`` long-format schema.
        Filters to ``metric_name == "revenue_growth_YoY"``.
        """
        df = metrics[metrics["metric_name"] == "revenue_growth_YoY"].copy() if not metrics.empty else pd.DataFrame()

        # Revenue growth is annual-only; filter to FY periods from FY2018
        if not df.empty:
            df = df[df["fiscal_period"].str.match(r"^FY\d{4}$")]
            df = df[df["fiscal_period"] >= "FY2018"]
            df = df.dropna(subset=["metric_value"])

        if df.empty:
            return self._empty_chart("revenue_growth_trend.png", "No revenue growth data available")

        df = df.sort_values("fiscal_period")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#5cb85c" if v >= 0 else "#d9534f" for v in df["metric_value"].values]
        ax.bar(df["fiscal_period"].values, df["metric_value"].values, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("NVDA Revenue Growth (YoY)")
        ax.set_ylabel("YoY Growth")
        ax.set_xlabel("Fiscal Period")
        ax.tick_params(axis="x", rotation=45)

        caption = "Source: NVIDIA SEC filings (XBRL). Revenue growth = (curr − prev) / prev."
        return self._save_with_attribution(
            fig, "revenue_growth_trend.png", caption,
            title="Revenue Growth Trend (YoY)",
            data_source="SEC EDGAR XBRL",
            date_range=f"{df['fiscal_period'].iloc[0]} – {df['fiscal_period'].iloc[-1]}" if len(df) else "",
        )

    # ------------------------------------------------------------------
    # 9. Peer Multiples Comparison  (grouped bar)  [optional]
    # ------------------------------------------------------------------
    def plot_peer_multiples_comparison(self, peer_multiples: pd.DataFrame) -> Path:
        """Grouped bar chart comparing EV/Revenue and P/E across peers.

        *peer_multiples* is the DataFrame returned by
        ``ValuationModule.compute_peer_multiples()``.  Expected columns
        include ``ticker``, ``EV/Revenue``, and ``P/E``.
        """
        if peer_multiples.empty:
            return self._empty_chart("peer_multiples_comparison.png", "No peer multiples data available")

        # Pick the two key multiples; tolerate different column naming conventions
        mult_cols = [c for c in ("EV/Revenue", "P/E", "ev_to_revenue", "pe_ratio") if c in peer_multiples.columns]
        if not mult_cols:
            return self._empty_chart("peer_multiples_comparison.png", "No EV/Revenue or P/E columns found")

        ticker_col = "ticker" if "ticker" in peer_multiples.columns else peer_multiples.columns[0]
        plot_df = peer_multiples[[ticker_col] + mult_cols].dropna(subset=mult_cols, how="all").copy()

        if plot_df.empty:
            return self._empty_chart("peer_multiples_comparison.png", "No valid peer multiples to plot")

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(plot_df))
        width = 0.35

        for i, col in enumerate(mult_cols):
            offset = (i - (len(mult_cols) - 1) / 2) * width
            vals = plot_df[col].fillna(0).values
            ax.bar(x + offset, vals, width, label=col.replace("_", " ").title())

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df[ticker_col].values, rotation=45, ha="right")
        ax.set_title("Peer Multiples Comparison")
        ax.set_ylabel("Multiple")
        ax.legend()

        caption = "Source: yfinance peer financials. Stale (>90 days) EV multiples excluded."
        return self._save_with_attribution(
            fig, "peer_multiples_comparison.png", caption,
            title="Peer Multiples Comparison (EV/Revenue & P/E)",
            data_source="yfinance",
        )

    # ------------------------------------------------------------------
    # 10. Keyword Theme Heatmap  [optional]
    # ------------------------------------------------------------------
    def plot_keyword_theme_heatmap(self, nlp: pd.DataFrame) -> Path:
        """Heatmap of keyword scores across themes and filing dates.

        *nlp* follows the ``nvda_nlp_features.csv`` schema.  Filters to
        ``feature_type == "keyword_score"`` and pivots with filing_date
        as rows and feature_name (theme) as columns.
        """
        kw = nlp[nlp["feature_type"] == "keyword_score"].copy() if not nlp.empty else pd.DataFrame()

        if kw.empty:
            return self._empty_chart("keyword_theme_heatmap.png", "No keyword score data available")

        pivot = kw.pivot_table(
            index="filing_date", columns="feature_name", values="value", aggfunc="mean",
        ).fillna(0)

        if pivot.empty:
            return self._empty_chart("keyword_theme_heatmap.png", "No keyword scores to pivot")

        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.5 + 2)))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
        )
        ax.set_title("Keyword Theme Scores by Filing Date")
        ax.set_ylabel("Filing Date")
        ax.set_xlabel("Theme")
        # Rotate x-axis labels diagonally to prevent overlap
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        caption = "Source: NVIDIA SEC filings (Risk Factors & MD&A). Normalized keyword frequency per theme."
        return self._save_with_attribution(
            fig, "keyword_theme_heatmap.png", caption,
            title="Keyword Theme Heatmap",
            data_source="SEC EDGAR filing text",
            date_range=f"{pivot.index.min()} – {pivot.index.max()}" if len(pivot) else "",
        )

    # ------------------------------------------------------------------
    # Utility: empty chart placeholder
    # ------------------------------------------------------------------
    def _empty_chart(self, filename: str, message: str) -> Path:
        """Create a minimal placeholder chart when data is unavailable."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="gray")
        ax.set_axis_off()
        return self._save_with_attribution(
            fig, filename, "No data available.",
            title=filename.replace(".png", "").replace("_", " ").title(),
        )
