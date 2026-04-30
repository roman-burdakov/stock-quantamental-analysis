"""
NVDA Quantamental Engine — Segment Revenue Normalizer.

Normalizes Nvidia's changing segment/platform labels into consistent
categories for trend analysis. XBRL dimensions are the primary extraction
method; table extraction from filing text is the fallback.

Requirements: 4.1–4.5
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import ChartStatus, EngineConfig, SegmentRevenueRecord

logger = logging.getLogger(__name__)


class SegmentRevenueNormalizer:
    """Normalize Nvidia segment/platform revenue across fiscal years."""

    # Maps original Nvidia segment/platform labels (case-insensitive key)
    # to normalized category names.  Nvidia changed its reportable segments
    # and market-platform breakdowns several times between FY2016 and FY2026.
    LABEL_MAPPING: dict[str, str] = {
        # --- Reportable segments (current) ---
        "compute & networking": "compute_and_networking",
        "compute and networking": "compute_and_networking",
        # --- Reportable segments (older) ---
        "graphics": "graphics",
        "gpu": "gpu",
        "gpu business": "gpu",
        "tegra processor": "tegra",
        "tegra processor business": "tegra",
        # --- Market / platform categories ---
        "data center": "data_center",
        "datacenter": "data_center",
        "gaming": "gaming",
        "professional visualization": "professional_visualization",
        "proviz": "professional_visualization",
        "automotive": "automotive",
        "oem and other": "oem_and_other",
        "oem & other": "oem_and_other",
        "oem and ip": "oem_and_other",
        "all other": "oem_and_other",
    }

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self._extraction_log: list[dict] = []
        self._unmapped_labels: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        xbrl_segments: pd.DataFrame,
        filing_texts: dict | None = None,
    ) -> pd.DataFrame:
        """Normalize segment revenue into a consistent schema.

        Parameters
        ----------
        xbrl_segments:
            DataFrame of XBRL segment_revenue facts.  Expected columns:
            ticker, fiscal_period, fiscal_year, filing_date,
            source_available_date, accession_number, value, unit,
            form_type.  May also contain an ``original_label`` column
            from XBRL dimension parsing.
        filing_texts:
            Optional mapping ``{accession_number: filing_html_text}``
            used as a fallback when XBRL dimensions are insufficient.

        Returns
        -------
        pd.DataFrame
            Columns matching :class:`SegmentRevenueRecord`.
        """
        self._extraction_log = []
        self._unmapped_labels = []

        records: list[dict] = []

        # --- Primary: XBRL dimension extraction ---
        xbrl_records = self._extract_from_xbrl(xbrl_segments)
        records.extend(xbrl_records)

        # Track which (fiscal_year, fiscal_period) combos have actual data
        covered_periods = {
            (r["fiscal_year"], r["fiscal_period"])
            for r in xbrl_records
            if r.get("value") is not None
        }

        # --- Fallback: table extraction from filing text ---
        if filing_texts:
            for accession, html_text in filing_texts.items():
                # Find the matching filing metadata from xbrl_segments
                meta = xbrl_segments[
                    xbrl_segments["accession_number"] == accession
                ]
                if meta.empty:
                    continue
                row_meta = meta.iloc[0]
                fy = int(row_meta["fiscal_year"])
                fp = str(row_meta["fiscal_period"])

                if (fy, fp) in covered_periods:
                    continue  # already have XBRL data for this period

                table_records = self._extract_from_table(
                    html_text=html_text,
                    accession=accession,
                    fiscal_year=fy,
                    fiscal_period=fp,
                    filing_date=str(row_meta["filing_date"]),
                    source_available_date=str(row_meta["source_available_date"]),
                )
                if table_records:
                    records.extend(table_records)
                    covered_periods.add((fy, fp))

        if not records:
            return self._empty_output()

        df = pd.DataFrame(records)

        # Ensure correct column order matching SegmentRevenueRecord
        columns = [
            "ticker", "fiscal_period", "fiscal_year", "filing_date",
            "source_available_date", "source_accession", "original_label",
            "normalized_category", "value", "unit", "extraction_method",
            "mapping_notes",
        ]
        df = df[columns]

        # Save to processed directory
        out_path = self.config.processed_dir / "nvda_segment_revenue_normalized.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved normalized segment revenue to %s (%d rows)", out_path, len(df))

        return df

    def generate_mapping_report(self) -> str:
        """Document all label mappings and extraction methods used.

        Returns the markdown text and appends it to
        ``outputs/data_quality_report.md``.
        """
        lines: list[str] = [
            "",
            "## Segment Revenue Normalization",
            "",
            f"**Ticker:** {self.config.ticker}",
            "",
        ]

        # --- Label mapping table ---
        lines.append("### Label Mapping")
        lines.append("")
        lines.append("| Original Label | Normalized Category |")
        lines.append("|---------------|-------------------|")
        for original, normalized in sorted(self.LABEL_MAPPING.items()):
            lines.append(f"| {original} | {normalized} |")
        lines.append("")

        # --- Extraction methods used ---
        lines.append("### Extraction Methods Used")
        lines.append("")
        if self._extraction_log:
            lines.append("| Fiscal Year | Period | Method | Records |")
            lines.append("|------------|--------|--------|---------|")
            method_summary: dict[tuple, dict] = {}
            for entry in self._extraction_log:
                key = (entry["fiscal_year"], entry["fiscal_period"], entry["method"])
                if key not in method_summary:
                    method_summary[key] = {"count": 0}
                method_summary[key]["count"] += entry.get("count", 1)
            for (fy, fp, method), info in sorted(method_summary.items()):
                lines.append(f"| {fy} | {fp} | {method} | {info['count']} |")
        else:
            lines.append("No extraction performed yet.")
        lines.append("")

        # --- Unmapped labels ---
        lines.append("### Unmapped / Changed Labels")
        lines.append("")
        if self._unmapped_labels:
            for item in self._unmapped_labels:
                lines.append(
                    f"- **{item['label']}** (FY{item.get('fiscal_year', '?')}, "
                    f"{item.get('fiscal_period', '?')}): {item.get('notes', 'no mapping found')}"
                )
        else:
            lines.append("All labels successfully mapped.")
        lines.append("")

        report_text = "\n".join(lines)

        # Append to data_quality_report.md
        report_path = self.config.outputs_dir / "data_quality_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(report_text)
        logger.info("Appended segment mapping report to %s", report_path)

        return report_text

    def check_segment_quality(
        self,
        normalized: pd.DataFrame,
        reported_revenue: dict[int, float] | None = None,
    ) -> ChartStatus:
        """Assess segment data quality and decide if chart is renderable.

        Parameters
        ----------
        normalized:
            DataFrame produced by :meth:`normalize` (SegmentRevenueRecord schema).
        reported_revenue:
            Optional mapping ``{fiscal_year: total_revenue}`` for reconciliation.
            When provided, a warning is emitted if segment totals differ by >5%.

        Returns
        -------
        ChartStatus
            ``renderable=False`` when all categories are "other" / "unclassified".
        """
        if normalized.empty:
            return ChartStatus(
                chart_name="segment_revenue_mix",
                renderable=False,
                reason="no_segment_data",
                fallback_message="No segment revenue data available.",
            )

        # Check if ALL categories are "other" or "unclassified"
        other_labels = {"other", "unclassified"}
        unique_categories = set(normalized["normalized_category"].str.lower().unique())
        if unique_categories.issubset(other_labels):
            return ChartStatus(
                chart_name="segment_revenue_mix",
                renderable=False,
                reason="all_categories_other",
                fallback_message=(
                    "Segment normalization failed: all categories mapped to "
                    "'Other' or 'Unclassified'. Chart suppressed."
                ),
            )

        # Reconciliation check: segment totals vs reported revenue
        warnings: list[str] = []
        if reported_revenue:
            for fy, reported in reported_revenue.items():
                if reported <= 0:
                    continue
                fy_data = normalized[normalized["fiscal_year"] == fy]
                if fy_data.empty:
                    continue
                segment_total = fy_data["value"].sum()
                diff_pct = abs(segment_total - reported) / reported * 100
                if diff_pct > 5.0:
                    warnings.append(
                        f"FY{fy}: segment total ${segment_total:,.0f} differs from "
                        f"reported revenue ${reported:,.0f} by {diff_pct:.1f}%"
                    )

        reason = "ok" if not warnings else "reconciliation_warning"
        fallback = None
        if warnings:
            fallback = "Reconciliation warnings: " + "; ".join(warnings)

        return ChartStatus(
            chart_name="segment_revenue_mix",
            renderable=True,
            reason=reason,
            fallback_message=fallback,
        )

    def generate_footnotes(self, normalized: pd.DataFrame) -> list[str]:
        """Generate footnotes for the segment revenue chart.

        Labels whether the chart shows reportable segments or market/platform
        categories, and notes which fiscal years use fallback extraction.

        Parameters
        ----------
        normalized:
            DataFrame produced by :meth:`normalize` (SegmentRevenueRecord schema).

        Returns
        -------
        list[str]
            Footnote strings suitable for chart annotation.
        """
        footnotes: list[str] = []

        if normalized.empty:
            return footnotes

        # Determine segment type from categories present
        reportable_segment_cats = {
            "compute_and_networking", "graphics", "gpu", "tegra",
        }
        market_platform_cats = {
            "data_center", "gaming", "professional_visualization",
            "automotive", "oem_and_other",
        }
        categories = set(normalized["normalized_category"].unique())

        has_reportable = bool(categories & reportable_segment_cats)
        has_market = bool(categories & market_platform_cats)

        if has_reportable and has_market:
            footnotes.append(
                "Chart shows a mix of reportable segments and market/platform categories."
            )
        elif has_reportable:
            footnotes.append("Chart shows reportable segments.")
        elif has_market:
            footnotes.append("Chart shows market/platform revenue categories.")

        # Note fiscal years using fallback (table_extraction)
        fallback_rows = normalized[
            normalized["extraction_method"] == "table_extraction"
        ]
        if not fallback_rows.empty:
            fallback_years = sorted(fallback_rows["fiscal_year"].unique())
            years_str = ", ".join(f"FY{y}" for y in fallback_years)
            footnotes.append(
                f"Fiscal years using fallback table extraction: {years_str}."
            )

        return footnotes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_label(self, label: str) -> tuple[str, str]:
        """Map an original label to its normalized category.

        Returns
        -------
        tuple[str, str]
            (normalized_category, mapping_notes)
        """
        cleaned = label.strip()
        key = cleaned.lower()

        if key in self.LABEL_MAPPING:
            return self.LABEL_MAPPING[key], ""

        # Try partial matching for labels with extra whitespace or punctuation
        for map_key, category in self.LABEL_MAPPING.items():
            if map_key in key or key in map_key:
                return category, f"partial match: '{cleaned}' → '{map_key}'"

        return "other", f"unmapped label: '{cleaned}'"

    def _extract_from_xbrl(self, xbrl_segments: pd.DataFrame) -> list[dict]:
        """Extract segment records from XBRL dimension data."""
        records: list[dict] = []

        if xbrl_segments.empty:
            return records

        has_label = "original_label" in xbrl_segments.columns

        for _, row in xbrl_segments.iterrows():
            original_label = str(row["original_label"]) if has_label else "segment_revenue"
            normalized, notes = self._normalize_label(original_label)

            if normalized == "other":
                self._unmapped_labels.append({
                    "label": original_label,
                    "fiscal_year": row.get("fiscal_year"),
                    "fiscal_period": row.get("fiscal_period"),
                    "notes": notes,
                })

            value = row.get("value")
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue

            records.append({
                "ticker": self.config.ticker,
                "fiscal_period": str(row.get("fiscal_period", "")),
                "fiscal_year": int(row.get("fiscal_year", 0)),
                "filing_date": str(row.get("filing_date", "")),
                "source_available_date": str(row.get("source_available_date", "")),
                "source_accession": str(row.get("accession_number", "")),
                "original_label": original_label,
                "normalized_category": normalized,
                "value": float(value),
                "unit": str(row.get("unit", "USD")),
                "extraction_method": "xbrl_dimension",
                "mapping_notes": notes,
            })

        if records:
            # Log extraction summary per period
            period_counts: dict[tuple, int] = {}
            for r in records:
                key = (r["fiscal_year"], r["fiscal_period"])
                period_counts[key] = period_counts.get(key, 0) + 1
            for (fy, fp), count in period_counts.items():
                self._extraction_log.append({
                    "fiscal_year": fy,
                    "fiscal_period": fp,
                    "method": "xbrl_dimension",
                    "count": count,
                })

        return records

    def _extract_from_table(
        self,
        html_text: str,
        accession: str,
        fiscal_year: int,
        fiscal_period: str,
        filing_date: str,
        source_available_date: str,
    ) -> list[dict]:
        """Fallback: extract segment revenue from filing HTML tables.

        Strips HTML tags, then uses regex to find segment/platform labels
        paired with dollar amounts.
        """
        records: list[dict] = []

        # Strip HTML tags to get plain text
        plain = re.sub(r"<[^>]+>", " ", html_text)
        # Collapse whitespace
        plain = re.sub(r"\s+", " ", plain)

        # Pattern: label text followed by a dollar value (with optional
        # thousands separators and $ sign).
        pattern = re.compile(
            r"(?P<label>[A-Za-z][A-Za-z &/,\-]+?)"
            r"\s*\$?\s*"
            r"(?P<value>[\d,]+(?:\.\d+)?)"
            r"\s*(?:million|Million|M)?\b",
            re.IGNORECASE,
        )

        # Known segment/platform keywords to filter noise
        segment_keywords = set(self.LABEL_MAPPING.keys())

        for m in pattern.finditer(plain):
            label = m.group("label").strip()
            label_lower = label.lower().strip()

            # Only accept labels that look like known segments
            if not any(kw in label_lower for kw in segment_keywords):
                continue

            try:
                value_str = m.group("value").replace(",", "")
                value = float(value_str)
            except (ValueError, TypeError):
                continue

            # Bounds validation: reject values that are clearly not
            # segment revenue (e.g., page numbers, percentages, dates).
            # Segment revenue should be at least $1M and less than $200B.
            if value <= 0 or value > 200_000_000:
                # If "million" or "M" suffix was present, value is in millions
                # Otherwise skip unreasonably small/large values
                if value <= 0:
                    continue
                # Without explicit "million" marker, raw numbers > 200B are noise
                if value > 200_000_000_000:
                    continue

            normalized, notes = self._normalize_label(label)
            if not notes:
                notes = "table extraction fallback"
            else:
                notes = f"table extraction fallback; {notes}"

            records.append({
                "ticker": self.config.ticker,
                "fiscal_period": fiscal_period,
                "fiscal_year": fiscal_year,
                "filing_date": filing_date,
                "source_available_date": source_available_date,
                "source_accession": accession,
                "original_label": label,
                "normalized_category": normalized,
                "value": value,
                "unit": "USD",
                "extraction_method": "table_extraction",
                "mapping_notes": notes,
            })

        if records:
            self._extraction_log.append({
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "method": "table_extraction",
                "count": len(records),
            })

        return records

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema."""
        return pd.DataFrame(columns=[
            "ticker", "fiscal_period", "fiscal_year", "filing_date",
            "source_available_date", "source_accession", "original_label",
            "normalized_category", "value", "unit", "extraction_method",
            "mapping_notes",
        ])
