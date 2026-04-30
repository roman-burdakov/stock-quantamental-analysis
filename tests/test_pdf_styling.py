"""
Tests for highly polished PDF styling (Task 17.7).

Validates:
- HTML output contains enhanced CSS from report_style.css
- Cover page has proper styling (centered h1, navy border)
- Tables have alternating row styling
- CSS includes page-break rules for PDF
- Figures/images are properly styled
- HTML output is valid and well-formed
- CSS file loads correctly and fallback works
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.report_utils import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator() -> ReportGenerator:
    return ReportGenerator()


@pytest.fixture
def sample_body() -> str:
    """A representative HTML body fragment mimicking report output."""
    return (
        "<h1>NVIDIA Corporation (NVDA)</h1>\n"
        "<h2>Quantamental Equity Research Report</h2>\n"
        "<p><strong>Report Date:</strong> 2026-04-25</p>\n"
        "<hr>\n"
        "<table>\n"
        "  <thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
        "  <tbody>\n"
        "    <tr><td><strong>Rating</strong></td><td>Buy</td></tr>\n"
        "    <tr><td><strong>Current Price</strong></td><td>$120.00</td></tr>\n"
        "    <tr><td><strong>Target Price</strong></td><td>$150.00</td></tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "<h2>Executive Summary</h2>\n"
        "<p>NVIDIA dominates AI accelerated computing.</p>\n"
        '<img src="outputs/figures/margin_trends.png" alt="Margin Trends">\n'
        "<em>Exhibit: Margin trends (FY2020-FY2025)</em>\n"
        "<h2>Valuation</h2>\n"
        "<p>DCF analysis suggests upside.</p>\n"
        "<blockquote><p>Diagnostic only — do not use for recommendation</p></blockquote>\n"
    )


@pytest.fixture
def wrapped_html(sample_body: str) -> str:
    """Full HTML document from _wrap_html."""
    return ReportGenerator._wrap_html(sample_body)


# ---------------------------------------------------------------------------
# Test: CSS file exists and loads
# ---------------------------------------------------------------------------

class TestCSSFileLoading:
    def test_css_file_exists(self):
        """The report_style.css file should exist in src/templates/."""
        css_path = Path("src/templates/report_style.css")
        assert css_path.exists(), "report_style.css not found"

    def test_css_file_not_empty(self):
        """The CSS file should contain substantial styling."""
        css_path = Path("src/templates/report_style.css")
        content = css_path.read_text(encoding="utf-8")
        assert len(content) > 500, "CSS file is too small to be a complete stylesheet"

    def test_load_css_returns_content(self):
        """_load_css should return the CSS file content."""
        css = ReportGenerator._load_css()
        assert len(css) > 500
        assert "body" in css

    def test_load_css_fallback_on_missing_file(self):
        """_load_css should return fallback CSS when file is missing."""
        with patch("src.report_utils.Path.read_text", side_effect=FileNotFoundError):
            css = ReportGenerator._load_css()
            assert "body" in css
            assert "font-family" in css


# ---------------------------------------------------------------------------
# Test: HTML structure is valid and well-formed
# ---------------------------------------------------------------------------

class TestHTMLStructure:
    def test_html5_doctype(self, wrapped_html: str):
        """Output should start with HTML5 doctype."""
        assert wrapped_html.startswith("<!DOCTYPE html>")

    def test_html_lang_attribute(self, wrapped_html: str):
        """HTML tag should have lang attribute."""
        assert '<html lang="en">' in wrapped_html

    def test_meta_charset(self, wrapped_html: str):
        """Head should include UTF-8 charset."""
        assert '<meta charset="utf-8">' in wrapped_html

    def test_meta_viewport(self, wrapped_html: str):
        """Head should include viewport meta tag."""
        assert "viewport" in wrapped_html

    def test_title_present(self, wrapped_html: str):
        """Head should include a title."""
        assert "<title>Quantamental Research Report</title>" in wrapped_html

    def test_style_tag_present(self, wrapped_html: str):
        """CSS should be embedded in a style tag."""
        assert "<style>" in wrapped_html
        assert "</style>" in wrapped_html

    def test_body_tags_present(self, wrapped_html: str):
        """Body tags should wrap the content."""
        assert "<body>" in wrapped_html
        assert "</body>" in wrapped_html

    def test_closing_tags(self, wrapped_html: str):
        """All major tags should be properly closed."""
        assert "</head>" in wrapped_html
        assert "</html>" in wrapped_html

    def test_body_content_preserved(self, wrapped_html: str):
        """The original body content should be present in the output."""
        assert "NVIDIA Corporation (NVDA)" in wrapped_html
        assert "Quantamental Equity Research Report" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Professional typography
# ---------------------------------------------------------------------------

class TestTypography:
    def test_serif_body_font(self, wrapped_html: str):
        """Body should use serif font (Georgia)."""
        assert "Georgia" in wrapped_html

    def test_sans_serif_headings(self, wrapped_html: str):
        """Headings should use sans-serif font."""
        assert "Helvetica Neue" in wrapped_html or "Arial" in wrapped_html

    def test_navy_heading_color(self, wrapped_html: str):
        """Headings should use navy color #1a1a2e."""
        assert "#1a1a2e" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Cover page styling
# ---------------------------------------------------------------------------

class TestCoverPageStyling:
    def test_cover_h1_centered(self, wrapped_html: str):
        """Cover page h1 should be centered."""
        assert "h1:first-of-type" in wrapped_html
        assert "text-align: center" in wrapped_html

    def test_cover_h1_border(self, wrapped_html: str):
        """Cover page h1 should have a navy bottom border."""
        assert "border-bottom: 3px solid #1a1a2e" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Table styling with alternating rows
# ---------------------------------------------------------------------------

class TestTableStyling:
    def test_table_border_collapse(self, wrapped_html: str):
        """Tables should use border-collapse."""
        assert "border-collapse: collapse" in wrapped_html

    def test_table_full_width(self, wrapped_html: str):
        """Tables should be full width."""
        assert "width: 100%" in wrapped_html

    def test_table_header_navy_background(self, wrapped_html: str):
        """Table headers should have navy background."""
        # The CSS has th { background: #1a1a2e; }
        assert "background: #1a1a2e" in wrapped_html

    def test_table_header_white_text(self, wrapped_html: str):
        """Table headers should have white text."""
        assert "color: #ffffff" in wrapped_html

    def test_alternating_row_even(self, wrapped_html: str):
        """Even rows should have gray background."""
        assert "tr:nth-child(even)" in wrapped_html
        assert "#f7f8fa" in wrapped_html

    def test_alternating_row_odd(self, wrapped_html: str):
        """Odd rows should have white background."""
        assert "tr:nth-child(odd)" in wrapped_html

    def test_header_uppercase(self, wrapped_html: str):
        """Table headers should be uppercase."""
        assert "text-transform: uppercase" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Page break rules for PDF
# ---------------------------------------------------------------------------

class TestPageBreaks:
    def test_h2_page_break_before(self, wrapped_html: str):
        """h2 sections should have page-break-before."""
        assert "page-break-before: always" in wrapped_html

    def test_h3_no_page_break_after(self, wrapped_html: str):
        """h3/h4 should avoid page-break-after."""
        assert "page-break-after: avoid" in wrapped_html

    def test_table_no_page_break_inside(self, wrapped_html: str):
        """Tables should avoid page-break-inside."""
        assert "page-break-inside: avoid" in wrapped_html

    def test_at_page_rule(self, wrapped_html: str):
        """CSS should include @page rule for PDF margins."""
        assert "@page" in wrapped_html
        assert "size: A4" in wrapped_html

    def test_page_footer_content(self, wrapped_html: str):
        """CSS should include page footer with page numbers."""
        assert "counter(page)" in wrapped_html
        assert "counter(pages)" in wrapped_html

    def test_print_media_query(self, wrapped_html: str):
        """CSS should include @media print rules."""
        assert "@media print" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Figure / image styling
# ---------------------------------------------------------------------------

class TestFigureStyling:
    def test_img_max_width(self, wrapped_html: str):
        """Images should have max-width: 100%."""
        assert "max-width: 100%" in wrapped_html

    def test_img_auto_height(self, wrapped_html: str):
        """Images should have height: auto for responsive sizing."""
        assert "height: auto" in wrapped_html

    def test_img_centered(self, wrapped_html: str):
        """Images should be centered with margin auto."""
        assert "margin: 1.5em auto" in wrapped_html

    def test_img_border_and_shadow(self, wrapped_html: str):
        """Images should have subtle border and shadow."""
        assert "box-shadow" in wrapped_html
        assert "border-radius" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Executive summary box styling
# ---------------------------------------------------------------------------

class TestExecutiveSummaryBox:
    def test_summary_box_background(self, wrapped_html: str):
        """Executive summary box should have light background."""
        assert "#f0f4f8" in wrapped_html

    def test_summary_box_left_border(self, wrapped_html: str):
        """Executive summary box should have left accent border."""
        assert "border-left: 4px solid #1a1a2e" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Blockquote / warning box styling
# ---------------------------------------------------------------------------

class TestBlockquoteStyling:
    def test_blockquote_background(self, wrapped_html: str):
        """Blockquotes should have a warm background for callouts."""
        assert "#fff8e1" in wrapped_html

    def test_blockquote_gold_border(self, wrapped_html: str):
        """Blockquotes should have gold left border."""
        assert "border-left: 4px solid #e8b931" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Footnote and source attribution styling
# ---------------------------------------------------------------------------

class TestFootnoteStyling:
    def test_footnote_class(self, wrapped_html: str):
        """CSS should include .footnote styling."""
        assert ".footnote" in wrapped_html

    def test_source_attribution_class(self, wrapped_html: str):
        """CSS should include .source-attribution styling."""
        assert ".source-attribution" in wrapped_html


# ---------------------------------------------------------------------------
# Test: Link styling
# ---------------------------------------------------------------------------

class TestLinkStyling:
    def test_link_gold_underline(self, wrapped_html: str):
        """Links should have gold underline accent."""
        assert "text-decoration-color: #e8b931" in wrapped_html


# ---------------------------------------------------------------------------
# Test: _wrap_html with empty body
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_body(self):
        """_wrap_html should handle empty body gracefully."""
        html = ReportGenerator._wrap_html("")
        assert "<!DOCTYPE html>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_body_with_special_characters(self):
        """_wrap_html should handle special characters in body."""
        body = "<p>Revenue &gt; $100B &amp; growing</p>"
        html = ReportGenerator._wrap_html(body)
        assert "&gt;" in html
        assert "&amp;" in html
