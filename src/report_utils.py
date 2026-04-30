"""
NVDA Quantamental Engine — Report Generation Utilities.

Assembles report context (with point-in-time filtering), renders
Jinja2 templates to Markdown, and produces PDF / HTML / executive
summary outputs.

Requirements: 10.1–10.9
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.config import (
    AuditStatus,
    ComponentStatus,
    DataQualityStatus,
    EngineConfig,
    FinalRecommendation,
    Recommendation,
    RecommendationStatus,
    ReportMode,
    get_default_config,
)
from src.display_format import (
    format_metric_label,
    format_currency,
    format_per_share,
    format_percentage,
    format_sensitivity_table,
    filter_to_display_window,
    is_within_display_window,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates the final quantamental research report."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or get_default_config()
        self.outputs_dir = Path(self.config.outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._env = Environment(
            loader=FileSystemLoader(str(self.config.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # 1. Build report context (point-in-time filtered)
    # ------------------------------------------------------------------

    def build_report_context(
        self,
        metrics: pd.DataFrame,
        segments: pd.DataFrame,
        nlp_features: pd.DataFrame,
        ml_results: Optional[dict] = None,
        valuation: Optional[dict] = None,
        charts: Optional[dict] = None,
        audit: Optional[dict] = None,
        recommendation_status: Optional[RecommendationStatus] = None,
        final_recommendation: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Assemble the template context dict.

        **Critical**: any row whose ``source_available_date > report_date``
        is excluded before the data enters the report context.

        If *final_recommendation* (FinalRecommendation) is provided, it is
        the ONLY source for rating, target, current price, upside/downside,
        thesis, and recommendation rationale. The legacy valuation
        ``Recommendation`` object is stored as ``raw_valuation_proposal``
        only — never as ``recommendation`` in the context.

        In formal_rating mode, final_recommendation is REQUIRED.
        """
        report_date = self.config.report_date
        display_start_fy = self.config.start_fiscal_year  # 2016
        display_end_fy = self.config.end_fiscal_year      # 2026

        # --- Point-in-time filtering ---
        metrics_filtered = self._filter_by_availability(metrics, report_date)
        segments_filtered = self._filter_by_availability(segments, report_date)
        nlp_filtered = self._filter_by_availability(nlp_features, report_date)

        # --- Determine report mode ---
        if recommendation_status is not None:
            report_mode = recommendation_status.eligibility_status.value
            blocking_issues = recommendation_status.blocking_issues
            component_statuses_list = [
                {
                    "component_name": cs.component_name,
                    "status": cs.status.value,
                    "reason": cs.reason,
                }
                for cs in recommendation_status.component_statuses
            ]
        else:
            report_mode = "formal_rating"
            blocking_issues = []
            component_statuses_list = []

        # --- FinalRecommendation is the STARTING POINT in formal_rating mode ---
        if final_recommendation is not None:
            fr = final_recommendation
            rating = fr.rating
            rating_label = fr.rating_label
            current_price = fr.current_price
            target_price = fr.target_price
            upside_pct = fr.upside_downside_pct
            thesis = fr.recommendation_rationale_long

            # Override report_mode from FinalRecommendation
            if fr.report_mode == "diagnostic_not_rated":
                report_mode = "diagnostic_not_rated"
                rating = "Not Rated"
                target_price = 0.0
                upside_pct = 0.0
        elif report_mode == "formal_rating":
            raise ValueError(
                "formal_rating mode requires final_recommendation. "
                "Cannot build report context without FinalRecommendation."
            )
        else:
            rating = "Not Rated"
            rating_label = "Not Rated"
            current_price = 0.0
            target_price = 0.0
            upside_pct = 0.0
            thesis = self._build_thesis(None, report_mode)

        # --- Force rating to "Not Rated" in non-formal modes ---
        if report_mode != "formal_rating":
            rating = "Not Rated"
            target_price = 0.0
            upside_pct = 0.0

        # --- Legacy valuation data (scenarios, DCF details, etc.) ---
        raw_valuation_proposal: Optional[Recommendation] = None
        scenarios: dict = {}
        dcf_details: Optional[dict] = None
        sensitivity_table: Optional[str] = None
        reverse_dcf_grid: Optional[str] = None
        peer_multiples: Optional[str] = None
        peer_multiples_core: Optional[str] = None
        peer_multiples_context: Optional[str] = None
        peer_multiples_nvda: Optional[str] = None
        historical_fcf_reconciliation: list[dict] = []
        scorecard: dict = {}

        if valuation:
            raw_valuation_proposal = valuation.get("recommendation")
            scenarios = valuation.get("scenarios", {})
            # Flatten scenario dicts for template access
            flat_scenarios: dict = {}
            for sc_name, sc_data in scenarios.items():
                assumptions = sc_data.get("assumptions")
                flat: dict = {
                    "name": sc_name,
                    "probability": sc_data.get("probability", 0),
                    "per_share_value": sc_data.get("per_share_value", 0),
                }
                if assumptions is not None:
                    flat["revenue_cagr"] = getattr(assumptions, "revenue_cagr", 0)
                    flat["fcf_margin_start"] = getattr(assumptions, "fcf_margin_start", 0)
                    flat["fcf_margin_terminal"] = getattr(assumptions, "fcf_margin_terminal", 0)
                    flat["terminal_growth"] = getattr(assumptions, "terminal_growth", 0)
                    flat["sbc_treatment"] = getattr(assumptions, "sbc_treatment", "")
                    flat["analyst_notes"] = getattr(assumptions, "analyst_notes", "")
                flat_scenarios[sc_name] = flat
            scenarios = flat_scenarios
            dcf_details = valuation.get("dcf_details")
            # Format sensitivity and reverse-DCF tables
            raw_sensitivity = valuation.get("sensitivity_table")
            sensitivity_table = format_sensitivity_table(raw_sensitivity) if raw_sensitivity else None
            raw_reverse_dcf = valuation.get("reverse_dcf_grid")
            reverse_dcf_grid = format_sensitivity_table(raw_reverse_dcf) if raw_reverse_dcf else None
            peer_multiples = valuation.get("peer_multiples")
            peer_multiples_core = valuation.get("peer_multiples_core")
            peer_multiples_context = valuation.get("peer_multiples_context")
            peer_multiples_nvda = valuation.get("peer_multiples_nvda")
            historical_fcf_reconciliation = valuation.get(
                "historical_fcf_reconciliation", []
            )
            if raw_valuation_proposal is not None:
                scorecard = raw_valuation_proposal.scorecard

        # --- Build key metrics summary with professional labels ---
        key_metrics = self._build_key_metrics(metrics_filtered)

        # --- Build segment summary ---
        segment_summary = self._build_segment_summary(segments_filtered)

        # --- Exhibits dict (chart file paths) ---
        exhibits = charts or {}

        # --- Point-in-time audit table ---
        pit_audit_table = self._build_pit_audit_table(report_date)

        # --- Narrative snippets ---
        narrative_snippets = (audit or {}).get("narrative_snippets", [])

        # --- Inflection points (filtered to display window) ---
        inflection_points = self._build_inflection_points(
            metrics_filtered, display_start_fy, display_end_fy
        )

        # --- Key bullets and risks ---
        key_bullets = self._build_key_bullets(
            metrics_filtered, segments_filtered, raw_valuation_proposal, report_mode
        )
        key_risks = self._build_key_risks()

        # --- Audit warnings ---
        audit_warnings: list[str] = []
        if audit:
            audit_status_obj = audit.get("audit_status")
            if audit_status_obj is not None:
                if isinstance(audit_status_obj, AuditStatus):
                    if audit_status_obj.overall_status == "fail":
                        audit_warnings.extend(audit_status_obj.failure_details)
                elif isinstance(audit_status_obj, dict):
                    if audit_status_obj.get("overall_status") == "fail":
                        audit_warnings.extend(
                            audit_status_obj.get("failure_details", [])
                        )
            audit_warnings.extend(audit.get("audit_warnings", []))

        context: dict[str, Any] = {
            # Cover page
            "company_name": self.config.company_name,
            "ticker": self.config.ticker,
            "report_date": report_date,
            "analyst_name": getattr(self.config, "analyst_name", ""),
            "analyst_title": getattr(self.config, "analyst_title", ""),
            "rating": rating,
            "current_price": current_price,
            "target_price": target_price,
            "upside_pct": upside_pct,
            "thesis": thesis,
            "key_bullets": key_bullets,
            "key_risks": key_risks,
            "risk_details": self._build_risk_details(),
            "catalyst_details": self._build_catalyst_details(),
            # Report mode
            "report_mode": report_mode,
            "blocking_issues": blocking_issues,
            "component_statuses": component_statuses_list,
            "audit_warnings": audit_warnings,
            # Data
            "metrics": key_metrics,
            "segments": segment_summary,
            "nlp_features": not nlp_filtered.empty,
            "ml_results": ml_results,
            "scenarios": scenarios,
            "raw_valuation_proposal": raw_valuation_proposal,
            "recommendation": raw_valuation_proposal,  # template compat
            "dcf_details": dcf_details,
            "sensitivity_table": sensitivity_table,
            "reverse_dcf_grid": reverse_dcf_grid,
            "peer_multiples": peer_multiples,
            "peer_multiples_core": peer_multiples_core,
            "peer_multiples_context": peer_multiples_context,
            "peer_multiples_nvda": peer_multiples_nvda,
            "historical_fcf_reconciliation": historical_fcf_reconciliation,
            "scorecard": scorecard,
            "exhibits": exhibits,
            "pit_audit_table": pit_audit_table,
            "narrative_snippets": narrative_snippets,
            "inflection_points": inflection_points,
        }

        # --- Peer Universe section ---
        peer_universe_data = self.build_peer_universe_section()
        context.update(peer_universe_data)

        # --- FinalRecommendation fields ---
        if final_recommendation is not None:
            fr = final_recommendation
            context["rating"] = fr.rating if report_mode == "formal_rating" else "Not Rated"
            context["rating_label"] = fr.rating_label
            context["current_price"] = fr.current_price
            context["target_price"] = fr.target_price if report_mode == "formal_rating" else 0.0
            context["upside_pct"] = fr.upside_downside_pct if report_mode == "formal_rating" else 0.0
            context["thesis"] = fr.recommendation_rationale_long
            context["recommendation_rationale"] = fr.recommendation_rationale_long
            context["why_hold_not_sell"] = fr.why_hold_not_sell
            context["why_sell_if_sell"] = getattr(fr, "why_sell_if_sell", "")
            context["upgrade_triggers"] = fr.upgrade_triggers
            context["downgrade_triggers"] = fr.downgrade_triggers
            context["active_exhibits"] = fr.active_exhibits
            context["suppressed_exhibits"] = fr.suppressed_exhibits
            context["final_recommendation"] = fr

        return context

    # ------------------------------------------------------------------
    # 2. Generate Markdown report
    # ------------------------------------------------------------------

    def generate_markdown_report(
        self,
        context: dict[str, Any],
        report_mode: Optional[ReportMode] = None,
    ) -> Path:
        """Render the main report template and write to
        ``outputs/<ticker>_quantamental_report.md``.

        Parameters
        ----------
        context:
            Template context dict from :meth:`build_report_context`.
        report_mode:
            Explicit report mode override. If ``None``, uses the
            ``report_mode`` value already in the context dict (defaults
            to ``"formal_rating"``).
        """
        # Allow explicit override of report_mode
        if report_mode is not None:
            context = dict(context)
            context["report_mode"] = report_mode.value

        template = self._env.get_template("report_base.md.j2")
        rendered = template.render(**context)

        out_path = self.outputs_dir / f"{self.config.ticker.lower()}_quantamental_report.md"
        out_path.write_text(rendered, encoding="utf-8")
        logger.info("Markdown report written to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # 3. Generate PDF
    # ------------------------------------------------------------------

    def generate_pdf(self, markdown_path: Path) -> Optional[Path]:
        """Convert the Markdown report to PDF.

        Tries weasyprint first (best quality).  Falls back to xhtml2pdf
        if weasyprint's native libraries are missing.  Returns ``None``
        only when neither backend is available.
        """
        md_text = markdown_path.read_text(encoding="utf-8")
        html_body = self._markdown_to_html(md_text)
        pdf_path = self.outputs_dir / f"{self.config.ticker.lower()}_quantamental_report.pdf"

        # --- Attempt 1: weasyprint (best quality) ---
        try:
            import weasyprint
            full_html = self._wrap_html(html_body)
            doc = weasyprint.HTML(string=full_html, base_url=str(Path.cwd()))
            doc.write_pdf(str(pdf_path))
            logger.info("PDF written via weasyprint → %s", pdf_path)
            return pdf_path
        except (ImportError, OSError) as exc:
            logger.info("weasyprint unavailable (%s), trying xhtml2pdf", exc)
        except Exception as exc:
            logger.warning("weasyprint failed (%s), trying xhtml2pdf", exc)

        # --- Attempt 2: xhtml2pdf (pure-Python fallback) ---
        try:
            from xhtml2pdf import pisa
            pdf_html = self._build_xhtml2pdf_html(html_body)
            with open(pdf_path, "wb") as fh:
                status = pisa.CreatePDF(
                    pdf_html, dest=fh,
                    path=str(Path.cwd()),  # resolve relative image paths
                )
            if status.err:
                logger.warning("xhtml2pdf reported %d error(s)", status.err)
            logger.info("PDF written via xhtml2pdf → %s (%d bytes)",
                        pdf_path, pdf_path.stat().st_size)
            return pdf_path
        except ImportError:
            logger.warning("xhtml2pdf not installed — PDF unavailable")
        except Exception as exc:
            logger.error("xhtml2pdf failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # 3b. xhtml2pdf-compatible HTML builder
    # ------------------------------------------------------------------

    def _build_xhtml2pdf_html(self, html_body: str) -> str:
        """Build a full HTML document tuned for xhtml2pdf rendering.

        xhtml2pdf has limited CSS support (no CSS Grid, limited @page,
        no ``content:`` with strings containing quotes inside @page).
        This method produces clean, compatible HTML with:
        - Embedded images via base64 data-URIs (reliable cross-platform)
        - Simplified but professional CSS that xhtml2pdf can parse
        - Proper page headers/footers via xhtml2pdf's pdf:* tags
        - Content-aware column widths for tables
        """
        import base64
        import re

        # --- Resolve image paths to base64 data-URIs ---
        def _img_to_base64(match: re.Match) -> str:
            """Replace <img src="..."> with base64-embedded version."""
            full_tag = match.group(0)
            src = match.group(1)
            # Skip already-embedded data URIs
            if src.startswith("data:"):
                return full_tag
            # Try to resolve the path
            for candidate in [
                Path(src),
                Path.cwd() / src,
                Path.cwd() / "outputs" / Path(src).name,
                Path.cwd() / "outputs" / "figures" / Path(src).name,
            ]:
                if candidate.exists() and candidate.is_file():
                    data = base64.b64encode(candidate.read_bytes()).decode()
                    suffix = candidate.suffix.lstrip(".")
                    mime = {"png": "image/png", "jpg": "image/jpeg",
                            "jpeg": "image/jpeg", "svg": "image/svg+xml",
                            "gif": "image/gif"}.get(suffix, "image/png")
                    return full_tag.replace(
                        f'src="{src}"',
                        f'src="data:{mime};base64,{data}"',
                    )
            # Image not found — keep original src, xhtml2pdf may resolve it
            logger.warning("Image not found for embedding: %s", src)
            return full_tag

        html_body = re.sub(
            r'<img\s[^>]*src="([^"]+)"[^>]*/?>',
            _img_to_base64,
            html_body,
        )

        # --- Add content-aware column widths to tables ---
        html_body = self._add_table_column_widths(html_body)

        css = self._xhtml2pdf_css()

        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '  <meta charset="utf-8">\n'
            '  <title>NVDA Quantamental Research Report</title>\n'
            f'  <style>\n{css}\n  </style>\n'
            '</head>\n'
            '<body>\n'
            '  <!-- PDF header/footer via xhtml2pdf -->\n'
            '  <div id="header_content" style="text-align:right;'
            'font-size:7pt;color:#999;">Confidential</div>\n'
            '  <div id="footer_content" style="text-align:center;'
            'font-size:8pt;color:#888;">'
            'NVDA Quantamental Research Report &mdash; '
            'Page <pdf:pagenumber/> of <pdf:pagecount/></div>\n'
            f'{html_body}\n'
            '</body>\n'
            '</html>\n'
        )

    @staticmethod
    def _add_table_column_widths(html: str) -> str:
        """Inject width styles into ``<th>`` tags so columns size to content.

        xhtml2pdf with ``table-layout: fixed`` divides space equally
        unless explicit widths are set.  This method:

        1. Strips HTML/markdown formatting to get plain text per cell.
        2. Finds the longest *word* in each column (the minimum width
           that avoids mid-word clipping).
        3. Finds the longest *cell* in each column (the ideal width).
        4. Allocates width proportional to ideal length, but enforces a
           floor based on the longest word so no column is too narrow
           for its content.
        """
        import re
        from html import unescape

        def _plain(s: str) -> str:
            """Strip HTML tags and markdown bold markers."""
            t = re.sub(r"<[^>]+>", "", s)
            t = t.replace("**", "")
            return unescape(t).strip()

        def _longest_word(s: str) -> int:
            """Length of the longest whitespace-delimited token."""
            words = _plain(s).split()
            return max((len(w) for w in words), default=0)

        parts = re.split(r"(<table>.*?</table>)", html, flags=re.DOTALL)
        result: list[str] = []

        for part in parts:
            if not part.startswith("<table>"):
                result.append(part)
                continue

            thead_m = re.search(r"<thead>\s*<tr>(.*?)</tr>\s*</thead>", part, re.DOTALL)
            if not thead_m:
                result.append(part)
                continue

            headers_html = thead_m.group(1)
            header_cells = re.findall(r"<th[^>]*>(.*?)</th>", headers_html, re.DOTALL)
            n_cols = len(header_cells)
            if n_cols == 0:
                result.append(part)
                continue

            # Measure every column: longest cell text + longest single word
            col_max_len = [len(_plain(h)) for h in header_cells]
            col_max_word = [_longest_word(h) for h in header_cells]

            tbody_m = re.search(r"<tbody>(.*?)</tbody>", part, re.DOTALL)
            body_html = tbody_m.group(1) if tbody_m else part
            body_rows = re.findall(r"<tr>(.*?)</tr>", body_html, re.DOTALL)

            for row_html in body_rows:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
                for i, cell in enumerate(cells):
                    if i < n_cols:
                        col_max_len[i] = max(col_max_len[i], len(_plain(cell)))
                        col_max_word[i] = max(col_max_word[i], _longest_word(cell))

            # --- Allocate widths ---
            # Each column needs at least enough room for its longest word
            # plus padding (~2 chars).  Convert char counts to a rough
            # "points" measure (1 char ≈ 5pt at 8.5pt font).
            CHAR_PT = 5.0
            PAD_PT = 12.0          # cell padding (6pt each side)
            PAGE_W = 500.0         # usable A4 width in pt (≈ 17.6cm)

            min_pts = [(w * CHAR_PT + PAD_PT) for w in col_max_word]
            ideal_pts = [(c * CHAR_PT + PAD_PT) for c in col_max_len]

            # Ensure minimums don't exceed page width
            total_min = sum(min_pts)
            if total_min > PAGE_W:
                # Scale minimums down proportionally
                scale = PAGE_W / total_min * 0.95
                min_pts = [m * scale for m in min_pts]

            # Distribute remaining space proportional to ideal length
            remaining = PAGE_W - sum(min_pts)
            if remaining > 0:
                ideal_extra = [max(0, ideal_pts[i] - min_pts[i]) for i in range(n_cols)]
                total_extra = sum(ideal_extra)
                if total_extra > 0:
                    alloc = [min_pts[i] + ideal_extra[i] / total_extra * remaining
                             for i in range(n_cols)]
                else:
                    alloc = [min_pts[i] + remaining / n_cols for i in range(n_cols)]
            else:
                alloc = min_pts

            # Convert to percentages
            total_alloc = sum(alloc)
            widths = [a / total_alloc * 100 for a in alloc]

            # Build new header row
            new_headers = []
            for i, cell_html in enumerate(header_cells):
                w = widths[i] if i < len(widths) else (100 / n_cols)
                new_headers.append(f'<th style="width:{w:.1f}%">{cell_html}</th>')
            new_thead_row = "<tr>" + "".join(new_headers) + "</tr>"

            part = (part[:thead_m.start()] +
                    "<thead>\n" + new_thead_row + "\n</thead>" +
                    part[thead_m.end():])
            result.append(part)

        return "".join(result)

    # ------------------------------------------------------------------
    # 3b-ii. xhtml2pdf CSS
    # ------------------------------------------------------------------

    @staticmethod
    def _xhtml2pdf_css() -> str:
        """Return CSS compatible with xhtml2pdf's renderer.

        xhtml2pdf uses a subset of CSS 2.1.  Unsupported features:
        - ``@page`` with ``content:`` strings containing quotes
        - ``::marker``, ``nth-child`` pseudo-selectors (limited)
        - CSS Grid / Flexbox
        - ``hyphens``, ``orphans``, ``widows``

        This stylesheet provides professional equity-research styling
        within those constraints.
        """
        return r"""
/* Page setup */
@page {
    size: A4;
    margin: 2cm 1.8cm 2.5cm 1.8cm;

    @frame header {
        -pdf-frame-content: header_content;
        top: 0.5cm;
        right: 1.8cm;
        height: 1cm;
    }
    @frame footer {
        -pdf-frame-content: footer_content;
        bottom: 0.5cm;
        left: 1.8cm;
        right: 1.8cm;
        height: 1cm;
    }
}

/* Base typography */
body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #2c2c2c;
}

p {
    margin: 0.5em 0;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: Helvetica, Arial, sans-serif;
    color: #1a1a2e;
}

h1 {
    font-size: 22pt;
    text-align: center;
    border-bottom: 3px solid #1a1a2e;
    padding-bottom: 6pt;
    margin-top: 40pt;
    margin-bottom: 6pt;
}

h2 {
    font-size: 15pt;
    border-bottom: 2px solid #1a1a2e;
    padding-bottom: 4pt;
    margin-top: 18pt;
    margin-bottom: 8pt;
}

h3 {
    font-size: 12pt;
    color: #1a1a2e;
    border-left: 3px solid #e8b931;
    padding-left: 8pt;
    margin-top: 14pt;
    margin-bottom: 6pt;
}

h4 {
    font-size: 11pt;
    color: #333;
    font-weight: bold;
    margin-top: 10pt;
    margin-bottom: 4pt;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 7.5pt;
    -pdf-keep-with-next: true;
    table-layout: fixed;
}

th {
    background-color: #1a1a2e;
    color: #ffffff;
    font-weight: bold;
    padding: 4pt 4pt;
    text-align: left;
    border: 1px solid #1a1a2e;
    font-size: 7pt;
    word-wrap: break-word;
    overflow: hidden;
}

td {
    padding: 3pt 4pt;
    border: 1px solid #d0d0d0;
    text-align: left;
    vertical-align: top;
    word-wrap: break-word;
    overflow: hidden;
}

tr {
    -pdf-keep-with-next: true;
}

/* Alternating row colors — manual via class if needed */

/* Blockquotes — callout boxes */
blockquote {
    background-color: #fff8e1;
    border-left: 4px solid #e8b931;
    padding: 8pt 12pt;
    margin: 10pt 0;
    font-size: 9.5pt;
    color: #333;
}

/* Strong text */
strong {
    color: #1a1a2e;
}

/* Images */
img {
    max-width: 480pt;
    width: 480pt;
    display: block;
    margin: 10pt auto;
    border: 1px solid #e0e0e0;
}

/* Image captions */
em {
    display: block;
    text-align: center;
    font-size: 8pt;
    color: #777;
    margin-top: 2pt;
    margin-bottom: 10pt;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 1px solid #d0d0d0;
    margin: 12pt 0;
}

/* Lists */
ul, ol {
    margin: 6pt 0;
    padding-left: 20pt;
}

li {
    margin-bottom: 3pt;
}

/* Code */
pre {
    background-color: #f5f6f8;
    padding: 8pt;
    border: 1px solid #e0e0e0;
    font-size: 8.5pt;
    font-family: 'Courier New', monospace;
}

code {
    background-color: #eef0f3;
    padding: 1pt 3pt;
    font-size: 8.5pt;
    font-family: 'Courier New', monospace;
}

/* Links */
a {
    color: #1a1a2e;
    text-decoration: none;
}
"""

    # ------------------------------------------------------------------
    # 4. Generate HTML (fallback)
    # ------------------------------------------------------------------

    def generate_html(self, markdown_path: Path) -> Path:
        """Convert the Markdown report to a standalone HTML file.

        Used as the primary fallback when PDF generation is unavailable.
        Logs the fallback in ``outputs/limitations.md``.
        """
        md_text = markdown_path.read_text(encoding="utf-8")
        html_body = self._markdown_to_html(md_text)
        full_html = self._wrap_html(html_body)

        html_path = self.outputs_dir / f"{self.config.ticker.lower()}_quantamental_report.html"
        html_path.write_text(full_html, encoding="utf-8")
        logger.info("HTML report written to %s", html_path)

        # Log fallback in limitations
        self._log_html_fallback()

        return html_path

    # ------------------------------------------------------------------
    # 5. Generate executive summary
    # ------------------------------------------------------------------

    def generate_executive_summary(
        self,
        context: dict[str, Any],
        report_mode: Optional[ReportMode] = None,
    ) -> Path:
        """Render the executive summary template (max 2 pages) and write
        to ``outputs/executive_summary.md``.

        Parameters
        ----------
        context:
            Template context dict from :meth:`build_report_context`.
        report_mode:
            Explicit report mode override. If ``None``, uses the
            ``report_mode`` value already in the context dict.
        """
        # Allow explicit override of report_mode
        if report_mode is not None:
            context = dict(context)
            context["report_mode"] = report_mode.value

        template = self._env.get_template("executive_summary.md.j2")
        rendered = template.render(**context)

        out_path = self.outputs_dir / "executive_summary.md"
        out_path.write_text(rendered, encoding="utf-8")
        logger.info("Executive summary written to %s", out_path)
        return out_path

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _filter_by_availability(
        df: pd.DataFrame, report_date: str
    ) -> pd.DataFrame:
        """Return only rows where ``source_available_date <= report_date``.

        Raises ValueError if a non-empty DataFrame is missing the
        ``source_available_date`` column, since that would silently
        allow future data to leak into the report.
        """
        if df.empty:
            return df
        if "source_available_date" not in df.columns:
            raise ValueError(
                "DataFrame is missing 'source_available_date' column. "
                "Cannot enforce point-in-time filtering. This is a data "
                "integrity violation — all data must carry provenance dates."
            )
        mask = df["source_available_date"] <= report_date
        filtered = df[mask].copy()
        n_excluded = len(df) - len(filtered)
        if n_excluded > 0:
            logger.info(
                "Excluded %d rows with source_available_date > %s",
                n_excluded,
                report_date,
            )
        return filtered

    # ------------------------------------------------------------------
    # Markdown → HTML conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _markdown_to_html(md_text: str) -> str:
        """Best-effort Markdown → HTML conversion.

        Uses the ``markdown`` library if available, otherwise falls back
        to a minimal regex-based converter.
        """
        try:
            import markdown as md_lib

            return md_lib.markdown(
                md_text,
                extensions=["tables", "fenced_code", "toc"],
            )
        except ImportError:
            pass

        # Minimal fallback: wrap in <pre> so content is at least readable
        import html as html_lib

        return "<pre>\n" + html_lib.escape(md_text) + "\n</pre>"

    @staticmethod
    def _load_css() -> str:
        """Load the professional report CSS from the external stylesheet.

        Tries to read ``src/templates/report_style.css``. If the file is
        not found, falls back to a minimal embedded stylesheet so that
        HTML output is always valid.
        """
        css_path = Path(__file__).parent / "templates" / "report_style.css"
        try:
            return css_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(
                "report_style.css not found at %s — using minimal fallback CSS",
                css_path,
            )
            return (
                "body { font-family: Georgia, serif; max-width: 900px; "
                "margin: 0 auto; line-height: 1.6; color: #2c2c2c; }\n"
                "h1,h2,h3 { font-family: Arial, sans-serif; color: #1a1a2e; }\n"
                "table { border-collapse: collapse; width: 100%; }\n"
                "th { background: #1a1a2e; color: #fff; padding: 0.5em; }\n"
                "td { padding: 0.5em; border: 1px solid #d0d0d0; }\n"
                "tr:nth-child(even) td { background: #f7f8fa; }\n"
                "img { max-width: 100%; height: auto; display: block; margin: 1em auto; }\n"
            )

    @staticmethod
    def _wrap_html(body: str) -> str:
        """Wrap an HTML body fragment in a full HTML5 document with
        professional investment-bank-style styling for equity research.

        Loads CSS from ``src/templates/report_style.css`` for maintainability.
        Falls back to minimal embedded CSS if the file is missing.
        """
        css = ReportGenerator._load_css()

        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "  <title>Quantamental Research Report</title>\n"
            f"  <style>\n{css}  </style>\n"
            "</head>\n"
            "<body>\n"
            f"{body}\n"
            "</body>\n"
            "</html>\n"
        )

    # ------------------------------------------------------------------
    # Context-building helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_key_metrics(metrics: pd.DataFrame) -> list[dict]:
        """Extract the most recent annual value for key metrics.

        Prefers annual (FY-only) periods over quarterly to avoid showing
        a single quarter's revenue instead of the full-year figure.
        Uses professional display labels from display_format module.
        """
        if metrics.empty:
            return []

        def _is_annual(period: str) -> bool:
            return bool(period) and period.startswith("FY") and "-" not in period and "Q" not in period

        priority = [
            "revenue",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "FCF",
            "FCF_margin",
            "ROE",
            "debt_to_equity",
            "revenue_growth_YoY",
            "diluted_EPS",
        ]
        records: list[dict] = []
        for name in priority:
            rows = metrics[metrics["metric_name"] == name]
            if rows.empty:
                continue

            # Prefer annual periods for absolute values
            annual = rows[rows["fiscal_period"].apply(_is_annual)]
            source = annual if not annual.empty else rows

            latest = source.sort_values("fiscal_period").iloc[-1]
            val = latest.get("metric_value")
            if val is None or pd.isna(val):
                continue
            # Format value
            if "margin" in name.lower() or name in ("ROE", "ROA"):
                formatted = f"{val * 100:.1f}%" if abs(val) <= 1 else f"{val:.2f}"
            elif "growth" in name.lower():
                formatted = f"{val * 100:.1f}%"
            elif name in ("FCF", "revenue"):
                formatted = f"${val / 1e9:.1f}B" if abs(val) >= 1e9 else f"${val / 1e6:.1f}M"
            elif name == "debt_to_equity":
                formatted = f"{val:.2f}"
            elif name == "diluted_EPS":
                formatted = f"${val:.2f}"
            else:
                formatted = f"{val:.2f}"
            records.append(
                {
                    "name": format_metric_label(name),
                    "value": formatted,
                    "period": latest.get("fiscal_period", ""),
                }
            )
        return records

    @staticmethod
    def _build_segment_summary(segments: pd.DataFrame) -> list[dict]:
        """Build a segment summary for the most recent period."""
        if segments.empty:
            return []

        if "fiscal_period" not in segments.columns:
            return []

        latest_period = segments["fiscal_period"].max()
        latest = segments[segments["fiscal_period"] == latest_period].copy()

        total = latest["value"].sum()
        records: list[dict] = []
        for _, row in latest.iterrows():
            share = row["value"] / total if total > 0 else 0.0
            records.append(
                {
                    "category": row.get("normalized_category", "Unknown"),
                    "value": row["value"],
                    "share": share,
                }
            )
        return records

    def _build_pit_audit_table(self, report_date: str) -> list[dict]:
        """Build the point-in-time audit table with ≥3 examples."""
        return [
            {
                "scenario": "FY2025 revenue in current report",
                "data": "10-K filed Feb 2025",
                "allowed": "✅ Yes",
                "reason": f"source_available_date (2025-02-26) ≤ report_date ({report_date})",
            },
            {
                "scenario": "FY2025 revenue to predict FY2024 growth (ML)",
                "data": "10-K filed Feb 2025",
                "allowed": "❌ No",
                "reason": "Filing date after prediction date — would be lookahead",
            },
            {
                "scenario": "Q3 FY2026 10-Q for narrative drift",
                "data": "10-Q filed Nov 2025",
                "allowed": "✅ Yes",
                "reason": f"source_available_date (2025-11-20) ≤ report_date ({report_date})",
            },
            {
                "scenario": "Post-report-date earnings call",
                "data": "Earnings call May 2026",
                "allowed": "❌ No",
                "reason": f"source_available_date (2026-05-28) > report_date ({report_date})",
            },
        ]

    @staticmethod
    def _build_inflection_points(
        metrics: pd.DataFrame,
        display_start_fy: int = 2016,
        display_end_fy: int = 2026,
    ) -> list[dict]:
        """Extract flagged inflection points from metrics, filtered to display window."""
        if metrics.empty or "inflection_flag" not in metrics.columns:
            return []

        inflections = metrics[metrics["inflection_flag"] == True]  # noqa: E712
        records: list[dict] = []
        for _, row in inflections.iterrows():
            period = row.get("fiscal_period", "")
            # Filter to display window (FY2016–FY2026)
            if not is_within_display_window(period, display_start_fy, display_end_fy):
                continue
            records.append(
                {
                    "period": period,
                    "metric": format_metric_label(row.get("metric_name", "")),
                    "description": f"YoY change exceeded 2σ threshold",
                }
            )
        return records[:10]  # Cap at 10

    @staticmethod
    def _build_thesis(
        recommendation: Optional[Recommendation],
        report_mode: str = "formal_rating",
    ) -> str:
        """Build the one-sentence investment thesis.

        In ``diagnostic_not_rated`` mode, never issues a formal rating
        statement — uses a neutral thesis instead.
        """
        if recommendation is None or report_mode != "formal_rating":
            return (
                "NVIDIA's dominance in AI accelerated computing positions it "
                "at the center of the most significant infrastructure buildout "
                "in a generation, but the investment question is whether "
                "today's valuation already prices in aggressive growth, margin, "
                "and durability assumptions."
            )
        rating = recommendation.rating
        upside = recommendation.upside_pct
        return (
            f"We rate NVIDIA {rating} with a probability-weighted intrinsic "
            f"value implying {upside * 100:.1f}% "
            f"{'upside' if upside >= 0 else 'downside'} from the current "
            f"price. The core question is whether AI infrastructure spend "
            f"sustains at levels that justify the current multiple, or "
            f"whether competitive and regulatory headwinds compress returns."
        )

    @staticmethod
    def _build_key_bullets(
        metrics: pd.DataFrame,
        segments: pd.DataFrame,
        recommendation: Optional[Recommendation],
        report_mode: str = "formal_rating",
    ) -> list[str]:
        """Build 3-5 key highlight bullets for the cover page.

        In ``diagnostic_not_rated`` mode, omits the valuation bullet
        that would reference a formal rating.
        """
        bullets: list[str] = []

        # Revenue growth
        if not metrics.empty:
            rev_growth = metrics[metrics["metric_name"] == "revenue_growth_YoY"]
            if not rev_growth.empty:
                latest = rev_growth.sort_values("fiscal_period").iloc[-1]
                val = latest.get("metric_value")
                if val is not None and not pd.isna(val):
                    bullets.append(
                        f"Revenue growth of {val * 100:.0f}% YoY in the most recent period"
                    )

        # Data Center dominance
        if not segments.empty and "normalized_category" in segments.columns:
            dc = segments[
                segments["normalized_category"].str.contains(
                    "data.center", case=False, na=False
                )
            ]
            if not dc.empty:
                latest_period = dc["fiscal_period"].max()
                dc_latest = dc[dc["fiscal_period"] == latest_period]
                total_seg = segments[segments["fiscal_period"] == latest_period][
                    "value"
                ].sum()
                dc_val = dc_latest["value"].sum()
                if total_seg > 0:
                    share = dc_val / total_seg
                    bullets.append(
                        f"Data Center segment represents {share * 100:.0f}% of revenue"
                    )

        # Valuation — only in formal_rating mode
        if recommendation and report_mode == "formal_rating":
            # Use FinalRecommendation rating if available in context
            display_rating = recommendation.rating
            bullets.append(
                f"Probability-weighted target of ${recommendation.expected_value:,.0f} "
                f"per share"
            )
        elif recommendation and report_mode == "diagnostic_not_rated":
            bullets.append(
                "Diagnostic valuation output only — not used for recommendation"
            )

        # Defaults if we don't have enough
        if len(bullets) < 3:
            defaults = [
                "CUDA ecosystem appears to create meaningful switching costs (analyst assessment based on NVIDIA disclosures)",
                "Export controls and customer concentration are key monitoring risks",
                "Point-in-time controls ensure no lookahead bias in analysis",
            ]
            for d in defaults:
                if len(bullets) >= 5:
                    break
                bullets.append(d)

        return bullets[:5]

    @staticmethod
    def _build_key_risks() -> list[str]:
        """Build 3 key risks for the cover page."""
        return [
            "Export control tightening could materially reduce restricted-market revenue (analyst assessment)",
            "Hyperscaler customer concentration — large customers represent material Data Center demand (analyst assessment)",
            "Custom silicon alternatives (TPU, Trainium, in-house ASICs) are a monitoring risk for GPU market share",
        ]

    @staticmethod
    def _build_risk_details() -> list[dict]:
        """Build detailed risk analysis table for the report body.

        Quantified estimates are labeled as analyst assessments where
        not independently sourced from SEC filings.
        """
        return [
            {
                "risk": "Export controls tightening",
                "mechanism": "BIS restrictions on advanced AI chips could materially reduce restricted-market revenue (magnitude is analyst scenario judgment); secondary sanctions risk extends to third-party resellers",
                "affected_metric": "Revenue, revenue CAGR",
                "signpost": "New BIS rules, China revenue % in 10-Q, customer shift to domestic alternatives",
            },
            {
                "risk": "Hyperscaler capex deceleration",
                "mechanism": "Large hyperscaler customers represent a material share of Data Center demand (analyst assessment); capex cycle maturation could reduce GPU order velocity",
                "affected_metric": "Revenue growth rate, forward bookings",
                "signpost": "Hyperscaler capex guidance in earnings calls, inventory buildup signals",
            },
            {
                "risk": "Customer concentration",
                "mechanism": "Revenue dependency on a small number of hyperscaler customers creates binary risk; loss of a single customer could materially impact growth",
                "affected_metric": "Revenue volatility, pricing power",
                "signpost": "Customer revenue disclosure in 10-K, diversification metrics",
            },
            {
                "risk": "Custom silicon substitution",
                "mechanism": "Google TPU, Amazon Trainium/Inferentia, Microsoft Maia, Meta MTIA may reduce GPU share of AI training and inference workloads over time",
                "affected_metric": "Market share, ASP, gross margin",
                "signpost": "Custom chip deployment announcements, benchmark results, workload migration disclosures",
            },
            {
                "risk": "Gross and FCF margin normalization",
                "mechanism": "Competitive pressure and rising supply could normalize pricing; R&D investment may compress FCF margins from peak levels (analyst assessment)",
                "affected_metric": "Gross margin, operating margin, FCF margin",
                "signpost": "Competitive pricing changes, gross margin trend in 10-Q, R&D spend growth rate",
            },
            {
                "risk": "Supply-chain / TSMC concentration",
                "mechanism": "NVIDIA's reliance on leading-edge foundry capacity creates supply-chain concentration risk (magnitude is analyst assessment); geopolitical disruption could constrain supply",
                "affected_metric": "Revenue delivery, lead times, inventory",
                "signpost": "TSMC capacity updates, geopolitical risk indices, NVDA inventory disclosures",
            },
        ]

    @staticmethod
    def _build_catalyst_details() -> list[dict]:
        """Build detailed catalyst analysis table for the report body."""
        return [
            {
                "catalyst": "Sustained data center growth",
                "confirmation": "Data Center revenue sustains strong YoY growth for 3+ consecutive quarters (monitoring threshold)",
                "assumption_affected": "Revenue CAGR (base and bull cases), terminal revenue TAM",
                "mechanism": "AI infrastructure buildout extends beyond initial training phase into inference-at-scale, expanding total GPU demand",
                "signpost": "Quarterly Data Center revenue trajectory, inference workload mix disclosures, cloud GPU utilization rates",
            },
            {
                "catalyst": "Blackwell product cycle",
                "confirmation": "Blackwell revenue exceeds Hopper ramp trajectory; gross margins stabilize at high levels (monitoring threshold)",
                "assumption_affected": "Revenue CAGR (bull case), gross margin assumptions",
                "mechanism": "Next-gen architecture delivers step-function performance improvement, driving upgrade cycle and ASP expansion",
                "signpost": "Blackwell revenue ramp vs Hopper in 10-Q, product mix disclosures, customer adoption announcements",
            },
            {
                "catalyst": "Enterprise AI broadening",
                "confirmation": "Enterprise revenue grows materially for 2+ consecutive quarters; customer base broadens (monitoring threshold)",
                "assumption_affected": "Terminal revenue TAM, revenue diversification",
                "mechanism": "AI adoption moves beyond hyperscalers into enterprise, healthcare, financial services, and manufacturing verticals",
                "signpost": "Enterprise segment revenue growth, vertical-specific deployment announcements, CUDA developer ecosystem metrics",
            },
            {
                "catalyst": "Sovereign AI buildouts",
                "confirmation": "Government and sovereign AI deals reach material scale (analyst monitoring threshold)",
                "assumption_affected": "Revenue diversification, geographic mix, export control mitigation",
                "mechanism": "National AI strategies drive government-funded GPU infrastructure independent of commercial hyperscaler cycles",
                "signpost": "Sovereign AI partnership announcements, government procurement contracts, country-level AI infrastructure budgets",
            },
            {
                "catalyst": "Software and networking attach rate",
                "confirmation": "Networking + software revenue share increases materially (analyst monitoring threshold)",
                "assumption_affected": "Gross margin, recurring revenue mix, competitive moat",
                "mechanism": "Higher-margin software and networking revenue grows as a share of total, improving margin durability and customer lock-in",
                "signpost": "Networking revenue disclosures, CUDA Enterprise subscription growth, software attach rate in earnings calls",
            },
            {
                "catalyst": "Export restriction easing",
                "confirmation": "BIS relaxes advanced chip restrictions for select markets or creates new compliant product tiers",
                "assumption_affected": "Restricted-market revenue recovery, addressable market expansion",
                "mechanism": "Regulatory easing reopens restricted markets, recovering a material share of addressable Data Center demand (analyst scenario)",
                "signpost": "BIS rule changes, new export-compliant product launches, restricted-market revenue recovery in 10-Q",
            },
        ]

    def build_peer_universe_section(self) -> dict[str, Any]:
        """Build the Peer Universe section data for the report template.

        Returns a dict with:
        - peer_universe_table: list of dicts with ticker, tier, justification,
          date_range for each peer
        - peer_universe_fixed: bool indicating the peer set is fixed
        - survivorship_bias_disclosure: str with the survivorship bias note

        Requirements: 25.1, 25.2
        """
        config = self.config
        start_fy = config.start_fiscal_year
        end_fy = config.end_fiscal_year
        date_range = f"FY{start_fy}–FY{end_fy}"

        peer_universe_table: list[dict[str, str]] = []

        # Core semiconductor peers
        for ticker in config.core_semiconductor_peers:
            justification = config.peer_justifications.get(ticker, "")
            peer_universe_table.append({
                "ticker": ticker,
                "tier": "Core Semiconductor",
                "justification": justification,
                "date_range": date_range,
            })

        # Infrastructure peers
        for ticker in config.infrastructure_peers:
            justification = config.peer_justifications.get(ticker, "")
            peer_universe_table.append({
                "ticker": ticker,
                "tier": "Infrastructure",
                "justification": justification,
                "date_range": date_range,
            })

        # AI capex context peers
        for ticker in config.ai_capex_context:
            justification = config.peer_justifications.get(ticker, "")
            peer_universe_table.append({
                "ticker": ticker,
                "tier": "AI Capex Context",
                "justification": justification,
                "date_range": date_range,
            })

        survivorship_bias_disclosure = (
            "The peer universe is **fixed** — the same set of peers is used "
            "for all fiscal years in the analysis window "
            f"(FY{start_fy}–FY{end_fy}). Peers were selected based on "
            "**current-day relevance** to NVIDIA's competitive landscape, "
            "not historical index membership or market position at the start "
            "of the analysis window. This introduces potential **survivorship "
            "bias**: for example, AMD was a significantly smaller company in "
            f"FY{start_fy} than today, and some current peers may not have "
            "been considered direct competitors at the start of the window. "
            "A fully point-in-time peer universe would require historical "
            "index constituent data not available in this analysis."
        )

        # --- Corporate events and comparability notes (Req 25.3) ---
        from src.config import PEER_CORPORATE_EVENTS

        all_tickers = (
            config.core_semiconductor_peers
            + config.infrastructure_peers
            + config.ai_capex_context
        )
        peer_corporate_events: list[dict[str, Any]] = []
        for ticker in all_tickers:
            events = PEER_CORPORATE_EVENTS.get(ticker, [])
            if events:
                peer_corporate_events.append({
                    "ticker": ticker,
                    "events": events,
                })

        # --- Index membership note (Req 25.4) ---
        index_membership_note = (
            "Index membership (S&P 500 or other relevant indices) at the "
            "start and end of the analysis window was **not tracked** for "
            "this analysis. All peers were selected based on current-day "
            "competitive relevance to NVIDIA, not index constituency. A "
            "point-in-time index membership analysis would require historical "
            "constituent data not available in this pipeline."
        )

        return {
            "peer_universe_table": peer_universe_table,
            "peer_universe_fixed": True,
            "survivorship_bias_disclosure": survivorship_bias_disclosure,
            "peer_corporate_events": peer_corporate_events,
            "index_membership_note": index_membership_note,
        }

    def _log_html_fallback(self) -> None:
        """Conditionally append HTML-fallback note to limitations.md.

        Only appends if no PDF file exists for this ticker, indicating
        that HTML is the only available output format.
        """
        pdf_path = self.outputs_dir / f"{self.config.ticker.lower()}_quantamental_report.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            # PDF exists — no fallback note needed
            return

        limitations_path = self.outputs_dir / "limitations.md"
        ticker_lower = self.config.ticker.lower()
        note = (
            "\n## PDF Generation\n\n"
            "PDF output was not generated. The `weasyprint` library is either "
            "not installed or encountered an error. An HTML fallback "
            f"(`{ticker_lower}_quantamental_report.html`) has been produced instead. "
            "Install `weasyprint` and its system dependencies for PDF output.\n"
        )
        if limitations_path.exists():
            existing = limitations_path.read_text(encoding="utf-8")
            if "PDF Generation" not in existing:
                with limitations_path.open("a", encoding="utf-8") as f:
                    f.write(note)
        else:
            limitations_path.write_text(
                "# Limitations\n" + note, encoding="utf-8"
            )
