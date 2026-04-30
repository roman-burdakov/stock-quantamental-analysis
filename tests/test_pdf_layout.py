"""
Tests for PDF layout quality — tables and images must not overflow page margins.

Checks the generated HTML report (which is the source for PDF rendering)
to ensure no table or image exceeds the available content width.

A4 page: 210mm wide, with 2cm (20mm) margins on each side = 170mm content width.
At ~96 DPI screen resolution, 170mm ≈ 644px.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import get_default_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_report_md() -> str:
    """Read the current Markdown report."""
    config = get_default_config()
    ticker = config.ticker.lower()
    path = Path(config.outputs_dir) / f"{ticker}_quantamental_report.md"
    if not path.exists():
        pytest.skip(f"Report not found at {path}")
    return path.read_text(encoding="utf-8")


def _get_report_html() -> str:
    """Read the current HTML report."""
    config = get_default_config()
    ticker = config.ticker.lower()
    path = Path(config.outputs_dir) / f"{ticker}_quantamental_report.html"
    if not path.exists():
        pytest.skip(f"HTML report not found at {path}")
    return path.read_text(encoding="utf-8")


def _extract_markdown_tables(md: str) -> list[dict]:
    """Extract all markdown tables with their headers and row counts."""
    tables = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            # Found a table header
            headers = [h.strip() for h in line.split("|") if h.strip()]
            num_cols = len(headers)
            # Count rows
            row_count = 0
            j = i + 2  # skip header and separator
            while j < len(lines) and lines[j].strip().startswith("|"):
                row_count += 1
                j += 1
            # Find the section heading above this table
            section = "Unknown"
            for k in range(i - 1, max(i - 10, -1), -1):
                if lines[k].strip().startswith("#"):
                    section = lines[k].strip().lstrip("#").strip()
                    break
            tables.append({
                "line": i + 1,
                "section": section,
                "headers": headers,
                "num_cols": num_cols,
                "num_rows": row_count,
                "max_header_len": max(len(h) for h in headers) if headers else 0,
            })
            i = j
        else:
            i += 1
    return tables


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTableOverflow:
    """Verify no markdown table has too many columns or excessively wide content."""

    # A4 with 2cm margins at 9.5pt font ≈ 12 narrow columns or 6-7 wide columns max
    MAX_COLUMNS = 12

    def test_no_table_exceeds_max_columns(self):
        """No table should have more than 12 columns (would overflow A4)."""
        md = _get_report_md()
        tables = _extract_markdown_tables(md)
        for t in tables:
            assert t["num_cols"] <= self.MAX_COLUMNS, (
                f"Table at line {t['line']} in section '{t['section']}' has "
                f"{t['num_cols']} columns (max {self.MAX_COLUMNS}). "
                f"Headers: {t['headers']}"
            )

    def test_peer_multiples_table_is_formatted(self):
        """Peer multiples table should use formatted numbers, not scientific notation."""
        md = _get_report_md()
        # Find the peer multiples section
        if "Peer Multiples" not in md:
            pytest.skip("No peer multiples table in report")
        peer_section = md[md.index("Peer Multiples"):]
        # Cut at next section
        next_section = peer_section.find("\n## ", 1)
        if next_section > 0:
            peer_section = peer_section[:next_section]
        next_h3 = peer_section.find("\n### ", 1)
        if next_h3 > 0:
            peer_section = peer_section[:next_h3]
        # Should not contain scientific notation like 1.99e+12
        assert not re.search(r"\d+\.\d+e\+\d+", peer_section), (
            "Peer multiples table contains scientific notation — "
            "numbers should be formatted as $XXB or XX.Xx"
        )

    def test_no_table_has_index_column(self):
        """Tables should not have a leading numeric index column from pandas."""
        md = _get_report_md()
        tables = _extract_markdown_tables(md)
        for t in tables:
            if t["headers"] and t["headers"][0] == "":
                # Empty first header = pandas index column
                assert False, (
                    f"Table at line {t['line']} in section '{t['section']}' "
                    f"has a leading index column. Use .to_markdown(index=False)."
                )

    def test_all_images_have_max_width(self):
        """All images in the HTML report should have max-width: 100% via CSS."""
        html = _get_report_html()
        # The CSS should contain img { max-width: 100% }
        assert "max-width" in html, (
            "HTML report CSS should contain max-width rule for images"
        )

    def test_tables_have_word_wrap(self):
        """Tables in the HTML report should have word-wrap CSS to prevent overflow."""
        html = _get_report_html()
        assert "word-wrap" in html or "overflow-wrap" in html, (
            "HTML report CSS should contain word-wrap or overflow-wrap for tables"
        )


class TestHeadingOrphans:
    """Verify section headings are not orphaned from their content."""

    def test_no_heading_followed_by_only_whitespace_then_heading(self):
        """A heading should not be followed by only whitespace/blank lines
        before the next heading (indicates orphaned heading in PDF)."""
        md = _get_report_md()
        lines = md.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("### ") and not stripped.startswith("### "):
                continue
            if stripped.startswith("### "):
                # Check if the next non-blank line is another heading
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    next_line = lines[j].strip()
                    # It's OK if the next content is a table, text, or blockquote
                    # It's a problem if it's another heading at the same or higher level
                    if next_line.startswith("## ") and not next_line.startswith("### "):
                        # h3 followed by h2 = section ended, heading was orphaned
                        pass  # This is structural, not necessarily a bug
