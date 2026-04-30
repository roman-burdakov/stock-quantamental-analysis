"""
Regression tests for fiscal-year selection in XBRL parsing.

Focused on the FY2025 revenue misparse scenario where the original parser
selected $26.974B (FY2023 comparative) instead of $130.497B (actual FY2025).

Uses fixtures from Milestone 0.3 and the real companyfacts JSON for
integration tests.

Reqs: 15.7
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.config import EngineConfig
from src.xbrl_parser import XBRLParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures")
REGRESSION_FIXTURES = FIXTURES / "regression"
RAW_DATA = Path("data/raw")


def _load_fy2025_candidates() -> dict:
    with open(REGRESSION_FIXTURES / "fy2025_revenue_candidates.json") as f:
        return json.load(f)


def _load_published_values() -> dict:
    with open(FIXTURES / "nvda_published_values.json") as f:
        return json.load(f)


def _make_parser(**overrides) -> XBRLParser:
    config = EngineConfig(**overrides)
    return XBRLParser(config)


def _make_candidates_df(candidates: list[dict], metric: str = "revenue") -> pd.DataFrame:
    """Build a DataFrame from the regression fixture candidates."""
    rows = []
    for c in candidates:
        rows.append({
            "ticker": "NVDA",
            "fiscal_period": c.get("fp", "FY"),
            "fiscal_year": c.get("fy", 0),
            "filing_date": c.get("filed", ""),
            "source_available_date": c.get("source_available_date", c.get("filed", "")),
            "accession_number": c.get("accn", ""),
            "metric_name": metric,
            "value": c.get("val"),
            "unit": "USD",
            "form_type": c.get("form", ""),
            "fiscal_period_type": c.get("period_type", ""),
            "period_start": c.get("start"),
            "period_end": c.get("end"),
            "duration_days": c.get("duration_days"),
            "frame": c.get("frame"),
            "xbrl_concept": "Revenues",
            "selection_rank": 0,
            "selection_reason": "",
        })
    return pd.DataFrame(rows)


# ===================================================================
# 1. Core regression: FY2025 revenue must be ~$130.5B, not $27B
# ===================================================================


class TestFY2025RevenueRegression:
    """Core regression tests for the FY2025 revenue misparse."""

    def test_fy2025_revenue_approximately_130_5b(self):
        """FY2025 revenue must be ~$130.5B (±1%), not $27B.

        This is THE core regression test. The original parser returned
        $26.974B because it selected a comparative-period fact from the
        same 10-K filing.
        """
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None, "No annual fact selected for FY2025"

        expected = 130497000000
        tolerance = expected * 0.01  # ±1%
        assert abs(result["value"] - expected) <= tolerance, (
            f"FY2025 revenue should be ~$130.5B (±1%), got ${result['value'] / 1e9:.3f}B"
        )

    def test_misparse_value_not_selected_for_fy2025(self):
        """The $26.974B misparse value must NOT be selected for FY2025.

        This value is FY2023 revenue reported as a comparative period in
        the FY2025 10-K. It shares fy=2025, fp=FY, form=10-K but has
        period_end=2023-01-29 (not 2025-01-26).
        """
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None
        assert result["value"] != 26974000000, (
            "Selected the $26.974B FY2023 comparative instead of $130.497B FY2025"
        )

    def test_fy2025_correct_period_dates(self):
        """The selected FY2025 fact must have period_start=2024-01-29, period_end=2025-01-26."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None
        assert result["period_start"] == "2024-01-29", (
            f"Expected period_start=2024-01-29, got {result['period_start']}"
        )
        assert result["period_end"] == "2025-01-26", (
            f"Expected period_end=2025-01-26, got {result['period_end']}"
        )


# ===================================================================
# 2. Quarterly facts must never be selected as annual
# ===================================================================


class TestQuarterlyRejection:
    """All quarterly facts in the fixture must be classified as quarterly
    and rejected for annual selection."""

    def test_all_quarterly_facts_classified_correctly(self):
        """Every fact with duration < 100 days must be classified as 'quarterly'."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        quarterly = all_candidates[all_candidates["duration_days"] < 100]
        assert not quarterly.empty, "Fixture should contain quarterly facts"

        for _, row in quarterly.iterrows():
            classification = XBRLParser._classify_period_type(
                row["period_start"], row["period_end"], row["duration_days"],
            )
            assert classification == "quarterly", (
                f"Fact with duration={row['duration_days']}d classified as "
                f"'{classification}' instead of 'quarterly': "
                f"val=${row['value'] / 1e9:.3f}B, period={row['period_start']}→{row['period_end']}"
            )

    def test_quarterly_facts_never_selected_as_annual(self):
        """Quarterly-only candidates must return None from _select_annual_fact."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        # Keep only quarterly facts
        quarterly_only = all_candidates[all_candidates["fiscal_period_type"] == "quarterly"].copy()
        assert not quarterly_only.empty

        parser = _make_parser()
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(quarterly_only, "revenue", 2025, fy_end)

        assert result is None, "Quarterly facts must never be selected as annual"


# ===================================================================
# 3. YTD facts must never be selected as annual
# ===================================================================


class TestYTDRejection:
    """All YTD facts in the fixture must be classified as YTD and rejected."""

    def test_all_ytd_facts_classified_correctly(self):
        """Every fact with 100 <= duration < 340 must be classified as 'ytd'."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        ytd = all_candidates[
            (all_candidates["duration_days"] >= 100)
            & (all_candidates["duration_days"] < 340)
        ]
        assert not ytd.empty, "Fixture should contain YTD facts"

        for _, row in ytd.iterrows():
            classification = XBRLParser._classify_period_type(
                row["period_start"], row["period_end"], row["duration_days"],
            )
            assert classification == "ytd", (
                f"Fact with duration={row['duration_days']}d classified as "
                f"'{classification}' instead of 'ytd': "
                f"val=${row['value'] / 1e9:.3f}B, period={row['period_start']}→{row['period_end']}"
            )

    def test_ytd_facts_never_selected_as_annual(self):
        """YTD-only candidates must return None from _select_annual_fact."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        # Keep only YTD facts
        ytd_only = all_candidates[all_candidates["fiscal_period_type"] == "ytd"].copy()
        assert not ytd_only.empty

        parser = _make_parser()
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(ytd_only, "revenue", 2025, fy_end)

        assert result is None, "YTD facts must never be selected as annual"


# ===================================================================
# 4. Amendment handling: latest amendment supersedes
# ===================================================================


class TestAmendmentRegression:
    """Regression tests for amendment handling in fiscal-year selection."""

    def test_latest_amendment_supersedes_original(self):
        """When two 10-K filings report the same FY, the latest filing_date wins."""
        parser = _make_parser()

        row_original = pd.Series({
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
            "filing_date": "2025-02-26", "source_available_date": "2025-02-26",
            "accession_number": "accn-original", "metric_name": "revenue",
            "value": 130497000000, "unit": "USD", "form_type": "10-K",
            "fiscal_period_type": "annual",
            "period_start": "2024-01-29", "period_end": "2025-01-26",
            "duration_days": 363, "frame": None, "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        })
        row_amendment = pd.Series({
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
            "filing_date": "2025-05-15", "source_available_date": "2025-05-15",
            "accession_number": "accn-amendment", "metric_name": "revenue",
            "value": 130500000000, "unit": "USD", "form_type": "10-K/A",
            "fiscal_period_type": "annual",
            "period_start": "2024-01-29", "period_end": "2025-01-26",
            "duration_days": 363, "frame": None, "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        })

        df = pd.DataFrame([row_original.to_dict(), row_amendment.to_dict()])
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is not None
        assert result["filing_date"] == "2025-05-15", (
            "Latest amendment should supersede original filing"
        )
        assert result["value"] == 130500000000

    def test_amendment_handling_deterministic(self):
        """Amendment handling must be deterministic regardless of input order."""
        parser = _make_parser()

        base = {
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
            "source_available_date": "2025-02-26",
            "accession_number": "accn-1", "metric_name": "revenue",
            "unit": "USD", "form_type": "10-K",
            "fiscal_period_type": "annual",
            "period_start": "2024-01-29", "period_end": "2025-01-26",
            "duration_days": 363, "frame": None, "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        }

        row_a = {**base, "filing_date": "2025-02-26", "value": 130497000000, "accession_number": "accn-a"}
        row_b = {**base, "filing_date": "2025-04-01", "value": 130500000000, "accession_number": "accn-b"}
        row_c = {**base, "filing_date": "2025-06-01", "value": 130510000000, "accession_number": "accn-c"}

        fy_end = date.fromisoformat("2025-01-26")

        # Order 1: a, b, c
        df1 = pd.DataFrame([row_a, row_b, row_c])
        result1 = parser._select_annual_fact(df1, "revenue", 2025, fy_end)

        # Order 2: c, a, b
        df2 = pd.DataFrame([row_c, row_a, row_b])
        result2 = parser._select_annual_fact(df2, "revenue", 2025, fy_end)

        assert result1 is not None and result2 is not None
        assert result1["value"] == result2["value"], (
            "Amendment handling must be deterministic regardless of input order"
        )
        assert result1["filing_date"] == "2025-06-01"
        assert result2["filing_date"] == "2025-06-01"


# ===================================================================
# 5. Frame-tagged preference
# ===================================================================


class TestFrameTaggedPreference:
    """Frame-tagged facts should get appropriate ranking."""

    def test_frame_tagged_10k_gets_rank_1(self):
        """A 10-K fact with a frame tag should get rank 1."""
        parser = _make_parser()

        df = pd.DataFrame([{
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
            "filing_date": "2025-02-26", "source_available_date": "2025-02-26",
            "accession_number": "accn-1", "metric_name": "revenue",
            "value": 130497000000, "unit": "USD", "form_type": "10-K",
            "fiscal_period_type": "annual",
            "period_start": "2024-01-29", "period_end": "2025-01-26",
            "duration_days": 363, "frame": "CY2024", "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        }])

        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is not None
        assert result["selection_rank"] == 1
        assert "frame" in result["selection_reason"].lower()

    def test_frame_tagged_non_10k_gets_rank_2(self):
        """A non-10-K fact with a frame tag should get rank 2."""
        parser = _make_parser()

        df = pd.DataFrame([{
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2026,
            "filing_date": "2026-02-25", "source_available_date": "2026-02-25",
            "accession_number": "accn-1", "metric_name": "revenue",
            "value": 130497000000, "unit": "USD", "form_type": "20-F",
            "fiscal_period_type": "annual",
            "period_start": "2024-01-29", "period_end": "2025-01-26",
            "duration_days": 363, "frame": "CY2024", "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        }])

        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is not None
        assert result["selection_rank"] == 2
        assert "frame" in result["selection_reason"].lower()

    def test_10k_without_frame_preferred_over_non_10k_with_frame(self):
        """10-K without frame (rank 1) should be preferred over non-10-K with frame (rank 2)."""
        parser = _make_parser()

        df = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
                "filing_date": "2025-02-26", "source_available_date": "2025-02-26",
                "accession_number": "accn-10k", "metric_name": "revenue",
                "value": 130497000000, "unit": "USD", "form_type": "10-K",
                "fiscal_period_type": "annual",
                "period_start": "2024-01-29", "period_end": "2025-01-26",
                "duration_days": 363, "frame": None, "xbrl_concept": "Revenues",
                "selection_rank": 0, "selection_reason": "",
            },
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2026,
                "filing_date": "2026-02-25", "source_available_date": "2026-02-25",
                "accession_number": "accn-other", "metric_name": "revenue",
                "value": 130497000000, "unit": "USD", "form_type": "20-F",
                "fiscal_period_type": "annual",
                "period_start": "2024-01-29", "period_end": "2025-01-26",
                "duration_days": 363, "frame": "CY2024", "xbrl_concept": "Revenues",
                "selection_rank": 0, "selection_reason": "",
            },
        ])

        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is not None
        assert result["selection_rank"] == 1
        assert result["form_type"] == "10-K"


# ===================================================================
# 6. Multi-year revenue from fixture: FY2024 and FY2023
# ===================================================================


class TestMultiYearRevenue:
    """FY2024 and FY2023 revenue must also parse correctly from the fixture."""

    def test_fy2024_revenue_from_fixture(self):
        """FY2024 revenue must be $60.922B when selecting for FY2024."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        fy_end = date.fromisoformat("2024-01-28")
        result = parser._select_annual_fact(annual_fy, "revenue", 2024, fy_end)

        assert result is not None, "No annual fact selected for FY2024"
        assert result["value"] == 60922000000, (
            f"Expected FY2024 revenue $60.922B, got ${result['value'] / 1e9:.3f}B"
        )

    def test_fy2023_revenue_from_fixture(self):
        """FY2023 revenue must be $26.974B when selecting for FY2023."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        fy_end = date.fromisoformat("2023-01-29")
        result = parser._select_annual_fact(annual_fy, "revenue", 2023, fy_end)

        assert result is not None, "No annual fact selected for FY2023"
        assert result["value"] == 26974000000, (
            f"Expected FY2023 revenue $26.974B, got ${result['value'] / 1e9:.3f}B"
        )

    def test_fy2023_value_not_confused_with_fy2025(self):
        """$26.974B must be selected for FY2023, not FY2025."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()

        fy2023_end = date.fromisoformat("2023-01-29")
        fy2025_end = date.fromisoformat("2025-01-26")

        result_2023 = parser._select_annual_fact(annual_fy, "revenue", 2023, fy2023_end)
        result_2025 = parser._select_annual_fact(annual_fy, "revenue", 2025, fy2025_end)

        assert result_2023 is not None and result_2025 is not None
        assert result_2023["value"] == 26974000000
        assert result_2025["value"] == 130497000000
        assert result_2023["value"] != result_2025["value"], (
            "FY2023 and FY2025 must select different values"
        )


# ===================================================================
# 7. Integration: full companyfacts JSON parsing
# ===================================================================


@pytest.mark.skipif(
    not (RAW_DATA / "companyfacts_CIK0001045810.json").exists(),
    reason="Real companyfacts JSON not available",
)
class TestFullCompanyfactsIntegration:
    """Integration tests using the real companyfacts JSON file."""

    @pytest.fixture(autouse=True)
    def _parse_companyfacts(self):
        """Parse the full companyfacts JSON once for all tests in this class."""
        with open(RAW_DATA / "companyfacts_CIK0001045810.json") as f:
            facts = json.load(f)
        self.parser = _make_parser()
        self.df = self.parser.parse_companyfacts(facts)

    def test_fy2025_revenue_from_full_json(self):
        """Parsing the FULL companyfacts JSON must produce FY2025 revenue ≈ $130.5B (±1%)."""
        rev_2025 = self.df[
            (self.df["metric_name"] == "revenue")
            & (self.df["fiscal_year"] == 2025)
            & (self.df["fiscal_period"] == "FY")
        ]

        assert not rev_2025.empty, "FY2025 revenue not found in parsed output"
        value = rev_2025.iloc[0]["value"]
        expected = 130497000000
        tolerance = expected * 0.01

        assert abs(value - expected) <= tolerance, (
            f"FY2025 revenue from full JSON: expected ~$130.5B (±1%), got ${value / 1e9:.3f}B"
        )

    def test_fy2024_revenue_from_full_json(self):
        """FY2024 revenue from full JSON must be $60.922B."""
        rev_2024 = self.df[
            (self.df["metric_name"] == "revenue")
            & (self.df["fiscal_year"] == 2024)
            & (self.df["fiscal_period"] == "FY")
        ]

        assert not rev_2024.empty, "FY2024 revenue not found in parsed output"
        value = rev_2024.iloc[0]["value"]
        expected = 60922000000
        tolerance = expected * 0.01

        assert abs(value - expected) <= tolerance, (
            f"FY2024 revenue from full JSON: expected ~$60.9B (±1%), got ${value / 1e9:.3f}B"
        )

    def test_fy2023_revenue_from_full_json(self):
        """FY2023 revenue from full JSON must be $26.974B."""
        rev_2023 = self.df[
            (self.df["metric_name"] == "revenue")
            & (self.df["fiscal_year"] == 2023)
            & (self.df["fiscal_period"] == "FY")
        ]

        assert not rev_2023.empty, "FY2023 revenue not found in parsed output"
        value = rev_2023.iloc[0]["value"]
        expected = 26974000000
        tolerance = expected * 0.01

        assert abs(value - expected) <= tolerance, (
            f"FY2023 revenue from full JSON: expected ~$27.0B (±1%), got ${value / 1e9:.3f}B"
        )

    def test_no_27b_value_for_fy2025(self):
        """The $26.974B misparse value must NOT appear as FY2025 revenue."""
        rev_2025 = self.df[
            (self.df["metric_name"] == "revenue")
            & (self.df["fiscal_year"] == 2025)
            & (self.df["fiscal_period"] == "FY")
        ]

        assert not rev_2025.empty
        for _, row in rev_2025.iterrows():
            assert row["value"] != 26974000000, (
                "Found $26.974B misparse value tagged as FY2025 revenue"
            )

    def test_published_values_match(self):
        """Cross-check parsed values against published reference for all 3 FYs."""
        published = _load_published_values()

        for fy_label, fy_data in published["values"].items():
            fy_num = int(fy_label.replace("FY", ""))
            expected_revenue = fy_data["metrics"]["revenue"]["value"]

            rev = self.df[
                (self.df["metric_name"] == "revenue")
                & (self.df["fiscal_year"] == fy_num)
                & (self.df["fiscal_period"] == "FY")
            ]

            if rev.empty:
                continue  # Some FYs may not be in the data range

            parsed_value = rev.iloc[0]["value"]
            diff_pct = abs(parsed_value - expected_revenue) / expected_revenue * 100

            assert diff_pct <= 1.0, (
                f"{fy_label} revenue: parsed ${parsed_value / 1e9:.3f}B vs "
                f"published ${expected_revenue / 1e9:.3f}B (diff {diff_pct:.2f}%)"
            )
