"""
NVDA Quantamental Engine — Centralized Display Formatting Utilities.

All report-facing label mapping, numeric formatting, and table
formatting lives here. No machine-generated labels should appear
in the final PDF/HTML/Markdown.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Metric label mapping (machine name → professional display name)
# ---------------------------------------------------------------------------

METRIC_LABEL_MAP: dict[str, str] = {
    "Fcf": "FCF",
    "Fcf Margin": "FCF Margin",
    "Roe": "ROE",
    "Roa": "ROA",
    "Debt To Equity": "Debt / Equity",
    "Revenue Growth Yoy": "Revenue Growth YoY",
    "Diluted Eps": "Diluted EPS",
    "Operating Cash Flow": "Operating Cash Flow",
    "Net Income": "Net Income",
    "Gross Margin": "Gross Margin",
    "Operating Margin": "Operating Margin",
    "Net Margin": "Net Margin",
    "Revenue": "Revenue",
    "Current Ratio": "Current Ratio",
    # Additional raw metric_name → display mappings
    "fcf": "FCF",
    "FCF": "FCF",
    "fcf_margin": "FCF Margin",
    "FCF_margin": "FCF Margin",
    "roe": "ROE",
    "ROE": "ROE",
    "roa": "ROA",
    "ROA": "ROA",
    "debt_to_equity": "Debt / Equity",
    "revenue_growth_YoY": "Revenue Growth YoY",
    "revenue_growth_yoy": "Revenue Growth YoY",
    "diluted_EPS": "Diluted EPS",
    "diluted_eps": "Diluted EPS",
    "operating_cash_flow": "Operating Cash Flow",
    "net_income": "Net Income",
    "gross_margin": "Gross Margin",
    "operating_margin": "Operating Margin",
    "net_margin": "Net Margin",
    "revenue": "Revenue",
    "current_ratio": "Current Ratio",
    "gross_profit": "Gross Profit",
    "operating_income": "Operating Income",
    "capex": "CapEx",
    "cash_and_securities": "Cash & Securities",
    "total_debt": "Total Debt",
    "diluted_shares": "Diluted Shares",
    "r_and_d": "R&D Expense",
}

# Banned tokens that must never appear in the final report
BANNED_TOKENS: list[str] = [
    "Fcf", "Roe", "Roa", "Debt To Equity", "Revenue Growth Yoy",
    "Diluted Eps", "TG=", "tg=", "wacc=", "cagr=", "margin=",
]


def format_metric_label(raw_name: str) -> str:
    """Convert a raw metric name to a professional display label.

    Tries exact match first, then title-cased match, then falls back
    to a cleaned-up version of the raw name.
    """
    # Exact match
    if raw_name in METRIC_LABEL_MAP:
        return METRIC_LABEL_MAP[raw_name]

    # Try title-cased version (from .replace("_", " ").title())
    title_version = raw_name.replace("_", " ").title()
    if title_version in METRIC_LABEL_MAP:
        return METRIC_LABEL_MAP[title_version]

    # Fallback: clean up underscores but preserve known acronyms
    cleaned = raw_name.replace("_", " ").title()
    # Fix common acronyms that title() breaks
    for acronym in ("Fcf", "Roe", "Roa", "Eps", "Yoy", "Ocf"):
        upper = acronym.upper()
        cleaned = cleaned.replace(acronym, upper)
    return cleaned


def format_currency(value: float, precision: int = 2) -> str:
    """Format a currency value with appropriate scale suffix.

    Examples: $215.9B, $96.7B, $1.99T, $173.04
    """
    if value is None:
        return "—"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:,.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:,.1f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:,.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val:,.0f}"
    else:
        return f"{sign}${abs_val:,.{precision}f}"


def format_per_share(value: float) -> str:
    """Format a per-share value: $173.04, $354.89."""
    if value is None:
        return "—"
    return f"${value:,.2f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format a ratio or percentage value: 10.0%, 13.3%, 25.0%.

    Expects value as a decimal (0.10 → 10.0%) unless abs(value) > 1,
    in which case it's treated as already a percentage.
    """
    if value is None:
        return "—"
    if abs(value) <= 1.0:
        return f"{value * 100:.{decimal_places}f}%"
    else:
        return f"{value:.{decimal_places}f}%"


def format_multiple(value: float) -> str:
    """Format a valuation multiple: 24.3x."""
    if value is None:
        return "—"
    return f"{value:.1f}x"


def format_ratio(value: float) -> str:
    """Format a ratio like debt/equity: 0.05."""
    if value is None:
        return "—"
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# Sensitivity / reverse-DCF table label formatting
# ---------------------------------------------------------------------------

def format_sensitivity_label(raw_label: str) -> str:
    """Convert raw sensitivity/DCF labels to professional format.

    Examples:
        tg=0.020 → Terminal Growth 2.0%
        wacc=0.080 → WACC 8.0%
        cagr=0.050 → Revenue CAGR 5.0%
        margin=0.250 → Terminal FCF Margin 25.0%
    """
    s = str(raw_label).strip()

    # Match patterns like "tg=0.020", "wacc=0.080", etc.
    m = re.match(r"^(tg|wacc|cagr|margin)\s*=\s*([0-9.]+)$", s, re.IGNORECASE)
    if m:
        key = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            return s

        label_map = {
            "tg": "Terminal Growth",
            "wacc": "WACC",
            "cagr": "Revenue CAGR",
            "margin": "Terminal FCF Margin",
        }
        display_name = label_map.get(key, key.upper())
        return f"{display_name} {val * 100:.1f}%"

    return s


def format_sensitivity_table(table_md: str) -> str:
    """Post-process a markdown sensitivity table to replace raw labels
    and round cell values.

    Handles both column headers and row index labels.
    """
    if not table_md:
        return table_md

    lines = table_md.split("\n")
    formatted_lines = []

    for line in lines:
        # Replace raw labels in table cells
        line = re.sub(
            r'\btg=([0-9.]+)',
            lambda m: f"TG {float(m.group(1)) * 100:.1f}%",
            line,
        )
        line = re.sub(
            r'\bwacc=([0-9.]+)',
            lambda m: f"WACC {float(m.group(1)) * 100:.1f}%",
            line,
        )
        line = re.sub(
            r'\bcagr=([0-9.]+)',
            lambda m: f"CAGR {float(m.group(1)) * 100:.0f}%",
            line,
        )
        line = re.sub(
            r'\bmargin=([0-9.]+)',
            lambda m: f"Margin {float(m.group(1)) * 100:.0f}%",
            line,
        )

        # Round numbers with excessive decimals in table cells
        # Match numbers like 183.196, 195.705 and round to 1 decimal
        def _round_cell(m: re.Match) -> str:
            val = float(m.group(0))
            if abs(val) >= 10:
                return f"{val:.1f}"
            else:
                return f"{val:.2f}"

        # Round floating point numbers with 2+ decimal places in table cells
        line = re.sub(r'(?<=[\s|])(\d+\.\d{2,})(?=[\s|])', _round_cell, line)

        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def format_inflection_metric(metric_name: str) -> str:
    """Format an inflection point metric name for display."""
    return format_metric_label(metric_name)


def scan_for_banned_tokens(text: str) -> list[str]:
    """Scan text for banned machine-format tokens.

    Returns a list of found banned tokens. Empty list means clean.
    """
    found = []
    for token in BANNED_TOKENS:
        # Use word boundary matching for short tokens to avoid false positives
        if len(token) <= 4:
            # For short tokens like "Fcf", "Roe", match as whole words
            pattern = r'\b' + re.escape(token) + r'\b'
            if re.search(pattern, text):
                found.append(token)
        else:
            if token in text:
                found.append(token)
    return found


# ---------------------------------------------------------------------------
# Report display window enforcement
# ---------------------------------------------------------------------------

def is_within_display_window(
    fiscal_period: str,
    display_start_fy: int = 2016,
    display_end_fy: int = 2026,
) -> bool:
    """Check if a fiscal period falls within the display window.

    Accepts formats like "FY2024", "FY2024-Q1", "FY2024-Q2", etc.
    Returns True if the fiscal year is within [display_start_fy, display_end_fy].
    """
    m = re.match(r"FY(\d{4})", str(fiscal_period))
    if not m:
        return False
    fy = int(m.group(1))
    return display_start_fy <= fy <= display_end_fy


def filter_to_display_window(
    items: list[dict],
    period_key: str = "period",
    display_start_fy: int = 2016,
    display_end_fy: int = 2026,
) -> list[dict]:
    """Filter a list of dicts to only include items within the display window."""
    return [
        item for item in items
        if is_within_display_window(
            item.get(period_key, ""),
            display_start_fy,
            display_end_fy,
        )
    ]
