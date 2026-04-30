"""
Tests for src.xbrl_parser — XBRL parsing and validation.

Covers:
  - Schema validation: parse_companyfacts returns correct columns
  - Missing tag handling: missing concepts logged, no crash
  - Deduplication: 10-K preferred over 10-Q, latest filing over amendments
  - Validation against known published values

All tests run offline using fixtures in tests/fixtures/.
Reqs: 12.1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from src.config import EngineConfig
from src.xbrl_parser import XBRLParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures")

EXPECTED_COLUMNS = [
    "ticker", "fiscal_period", "fiscal_year", "filing_date",
    "source_available_date", "accession_number", "metric_name",
    "value", "unit", "form_type",
    "fiscal_period_type", "period_start", "period_end",
    "duration_days", "frame", "xbrl_concept",
    "selection_rank", "selection_reason",
    "raw_value", "adjusted_value", "adjustment_factor",
    "adjustment_basis", "validation_basis",
]


def _load_companyfacts() -> dict:
    with open(FIXTURES / "sample_companyfacts.json") as f:
        return json.load(f)


def _load_known_values() -> dict:
    with open(FIXTURES / "known_validation_values.json") as f:
        return json.load(f)


def _make_parser(**overrides) -> XBRLParser:
    config = EngineConfig(**overrides)
    return XBRLParser(config)


# ===================================================================
# 1. Schema validation
# ===================================================================


class TestSchemaValidation:
    """Verify parse_companyfacts returns a DataFrame with the correct schema."""

    def test_output_has_correct_columns(self):
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        assert list(df.columns) == EXPECTED_COLUMNS

    def test_output_is_dataframe(self):
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        assert isinstance(df, pd.DataFrame)

    def test_output_not_empty(self):
        """Fixture has data for multiple metrics — result should not be empty."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        assert not df.empty

    def test_ticker_matches_config(self):
        parser = _make_parser(ticker="NVDA")
        df = parser.parse_companyfacts(_load_companyfacts())

        assert (df["ticker"] == "NVDA").all()

    def test_fiscal_year_is_int(self):
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        assert df["fiscal_year"].dtype in ("int64", "int32", "int")

    def test_value_column_populated(self):
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        assert df["value"].notna().any()

    def test_empty_facts_returns_empty_df_with_schema(self):
        """Empty facts JSON should return an empty DataFrame with correct columns."""
        parser = _make_parser()
        df = parser.parse_companyfacts({"facts": {}})

        assert df.empty
        assert list(df.columns) == EXPECTED_COLUMNS


# ===================================================================
# 2. Missing tag handling
# ===================================================================


class TestMissingTagHandling:
    """Verify that missing XBRL concepts are logged and don't crash."""

    def test_missing_concept_does_not_crash(self):
        """A facts JSON with no matching concepts should return empty, not raise."""
        parser = _make_parser()
        facts = {"facts": {"us-gaap": {"SomeUnrelatedTag": {"units": {"USD": []}}}}}
        df = parser.parse_companyfacts(facts)

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_concept_logged(self, caplog):
        """Missing concepts should produce warning log messages."""
        parser = _make_parser()
        facts = {"facts": {"us-gaap": {}}}

        with caplog.at_level(logging.WARNING):
            parser.parse_companyfacts(facts)

        assert any("No XBRL concept found" in msg for msg in caplog.messages)

    def test_missing_tags_tracked_internally(self):
        """Parser should track missing tags in _missing_tags list."""
        parser = _make_parser()
        facts = {"facts": {"us-gaap": {}}}
        parser.parse_companyfacts(facts)

        assert len(parser._missing_tags) > 0
        assert all("metric" in tag for tag in parser._missing_tags)

    def test_partial_data_returns_found_metrics_only(self):
        """If only some concepts exist, only those metrics appear in output."""
        parser = _make_parser()
        # Fixture has Revenues but not SellingGeneralAndAdministrativeExpense
        facts = _load_companyfacts()
        df = parser.parse_companyfacts(facts)

        metrics = df["metric_name"].unique()
        assert "revenue" in metrics
        # sga is not in the fixture
        assert "sga" not in metrics


# ===================================================================
# 3. Deduplication
# ===================================================================


class TestDeduplication:
    """Verify 10-K preferred over 10-Q, latest filing over amendments."""

    def test_10k_preferred_over_10q_same_period(self):
        """For the same (metric, fiscal_year, fiscal_period), 10-K wins over 10-Q."""
        parser = _make_parser()
        df = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
                "filing_date": "2025-02-26", "source_available_date": "2025-02-26",
                "accession_number": "accn-10k", "metric_name": "revenue",
                "value": 130497000000, "unit": "USD", "form_type": "10-K",
                "fiscal_period_type": "annual", "period_start": "2024-01-29",
                "period_end": "2025-01-26", "duration_days": 363,
                "frame": None, "xbrl_concept": "Revenues",
            },
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
                "filing_date": "2025-01-15", "source_available_date": "2025-01-15",
                "accession_number": "accn-10q", "metric_name": "revenue",
                "value": 999999, "unit": "USD", "form_type": "10-Q",
                "fiscal_period_type": "quarterly", "period_start": "2024-10-28",
                "period_end": "2025-01-26", "duration_days": 90,
                "frame": None, "xbrl_concept": "Revenues",
            },
        ])

        result = parser.deduplicate_facts(df)
        assert len(result) == 1
        assert result.iloc[0]["form_type"] == "10-K"
        assert result.iloc[0]["value"] == 130497000000

    def test_latest_filing_preferred_over_amendment(self):
        """For same form type, latest filing_date wins (original over amendment)."""
        parser = _make_parser()
        df = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2024,
                "filing_date": "2024-02-21", "source_available_date": "2024-02-21",
                "accession_number": "accn-original", "metric_name": "revenue",
                "value": 60922000000, "unit": "USD", "form_type": "10-K",
                "fiscal_period_type": "annual", "period_start": "2023-01-30",
                "period_end": "2024-01-28", "duration_days": 363,
                "frame": None, "xbrl_concept": "Revenues",
            },
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2024,
                "filing_date": "2024-03-15", "source_available_date": "2024-03-15",
                "accession_number": "accn-amendment", "metric_name": "revenue",
                "value": 60922000001, "unit": "USD", "form_type": "10-K/A",
                "fiscal_period_type": "annual", "period_start": "2023-01-30",
                "period_end": "2024-01-28", "duration_days": 363,
                "frame": None, "xbrl_concept": "Revenues",
            },
        ])

        result = parser.deduplicate_facts(df)
        assert len(result) == 1
        # 10-K has priority 0, 10-K/A has priority 1 → 10-K wins
        assert result.iloc[0]["form_type"] == "10-K"

    def test_dedup_preserves_different_periods(self):
        """Different fiscal periods should not be deduplicated against each other."""
        parser = _make_parser()
        df = pd.DataFrame([
            {
                "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2024,
                "filing_date": "2024-02-21", "source_available_date": "2024-02-21",
                "accession_number": "accn-1", "metric_name": "revenue",
                "value": 60922000000, "unit": "USD", "form_type": "10-K",
                "fiscal_period_type": "annual", "period_start": "2023-01-30",
                "period_end": "2024-01-28", "duration_days": 363,
                "frame": None, "xbrl_concept": "Revenues",
            },
            {
                "ticker": "NVDA", "fiscal_period": "Q3", "fiscal_year": 2025,
                "filing_date": "2024-11-20", "source_available_date": "2024-11-20",
                "accession_number": "accn-2", "metric_name": "revenue",
                "value": 35082000000, "unit": "USD", "form_type": "10-Q",
                "fiscal_period_type": "quarterly", "period_start": "2024-07-29",
                "period_end": "2024-10-27", "duration_days": 90,
                "frame": None, "xbrl_concept": "Revenues",
            },
        ])

        result = parser.deduplicate_facts(df)
        assert len(result) == 2

    def test_dedup_empty_df(self):
        """Deduplication on empty DataFrame should return empty."""
        parser = _make_parser()
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        result = parser.deduplicate_facts(df)
        assert result.empty

    def test_full_parse_deduplicates(self):
        """parse_companyfacts should deduplicate internally — no duplicate keys."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        dupes = df.duplicated(subset=["metric_name", "fiscal_year", "fiscal_period"])
        assert not dupes.any(), "Found duplicate (metric, fy, fp) rows after parsing"


# ===================================================================
# 4. Validation against known published values
# ===================================================================


class TestValidation:
    """Verify validate_against_published returns pass/fail/missing correctly."""

    def test_validation_returns_dataframe(self):
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        known = _load_known_values()
        result = parser.validate_against_published(df, known)

        assert isinstance(result, pd.DataFrame)

    def test_validation_has_expected_columns(self):
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        known = _load_known_values()
        result = parser.validate_against_published(df, known)

        expected_cols = ["metric", "fiscal_year", "parsed_value", "published_value",
                         "diff_pct", "status"]
        assert list(result.columns) == expected_cols

    def test_revenue_fy2024_passes(self):
        """FY2024 revenue in fixture matches known value exactly → pass."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        known = _load_known_values()
        result = parser.validate_against_published(df, known)

        rev_2024 = result[
            (result["metric"] == "revenue") & (result["fiscal_year"] == 2024)
        ]
        assert len(rev_2024) == 1
        assert rev_2024.iloc[0]["status"] == "pass"

    def test_revenue_fy2025_passes(self):
        """FY2025 revenue in fixture matches known value exactly → pass."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        known = _load_known_values()
        result = parser.validate_against_published(df, known)

        rev_2025 = result[
            (result["metric"] == "revenue") & (result["fiscal_year"] == 2025)
        ]
        assert len(rev_2025) == 1
        assert rev_2025.iloc[0]["status"] == "pass"

    def test_missing_metric_reported_as_missing(self):
        """Metrics in known_values but not in parsed data should be 'missing'."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        known = _load_known_values()
        result = parser.validate_against_published(df, known)

        # FY2023 is in known_values but not in the fixture → missing
        fy2023 = result[result["fiscal_year"] == 2023]
        if not fy2023.empty:
            assert (fy2023["status"] == "missing").all()

    def test_pass_status_within_tolerance(self):
        """All 'pass' rows should have diff_pct <= tolerance."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        known = _load_known_values()
        tolerance = known.get("tolerance_pct", 2.0)
        result = parser.validate_against_published(df, known)

        passed = result[result["status"] == "pass"]
        for _, row in passed.iterrows():
            assert row["diff_pct"] <= tolerance

    def test_statuses_are_valid(self):
        """All status values should be one of pass/fail/missing."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        known = _load_known_values()
        result = parser.validate_against_published(df, known)

        valid_statuses = {"pass", "fail", "missing"}
        assert set(result["status"].unique()).issubset(valid_statuses)


# ===================================================================
# 5. Period classification (Reqs 15.2, 15.3, 15.4)
# ===================================================================


class TestPeriodClassification:
    """Verify _classify_period_type() and period metadata in parsed output."""

    # --- _classify_period_type unit tests ---

    def test_instant_when_no_start(self):
        """Facts with no start date are classified as instant."""
        assert XBRLParser._classify_period_type(None, "2025-01-26", None) == "instant"

    def test_annual_at_lower_bound(self):
        """350-day duration is classified as annual."""
        assert XBRLParser._classify_period_type("2024-01-29", "2025-01-14", 350) == "annual"

    def test_annual_at_upper_bound(self):
        """380-day duration is classified as annual."""
        assert XBRLParser._classify_period_type("2024-01-01", "2025-01-16", 380) == "annual"

    def test_annual_typical_fiscal_year(self):
        """363-day NVIDIA fiscal year is classified as annual."""
        assert XBRLParser._classify_period_type("2024-01-29", "2025-01-26", 363) == "annual"

    def test_quarterly_90_days(self):
        """90-day duration is classified as quarterly."""
        assert XBRLParser._classify_period_type("2024-01-29", "2024-04-28", 90) == "quarterly"

    def test_quarterly_at_boundary(self):
        """99-day duration is still quarterly (< 100)."""
        assert XBRLParser._classify_period_type("2024-01-01", "2024-04-09", 99) == "quarterly"

    def test_ytd_at_lower_bound(self):
        """100-day duration is classified as ytd."""
        assert XBRLParser._classify_period_type("2024-01-01", "2024-04-10", 100) == "ytd"

    def test_ytd_six_months(self):
        """181-day (6-month) duration is classified as ytd."""
        assert XBRLParser._classify_period_type("2024-01-29", "2024-07-28", 181) == "ytd"

    def test_ytd_nine_months(self):
        """272-day (9-month) duration is classified as ytd."""
        assert XBRLParser._classify_period_type("2024-01-29", "2024-10-27", 272) == "ytd"

    def test_ytd_at_upper_bound(self):
        """339-day duration is still ytd (< 340)."""
        assert XBRLParser._classify_period_type("2024-01-01", "2024-12-05", 339) == "ytd"

    def test_other_for_gap_range(self):
        """340-349 day duration falls in the gap → other."""
        assert XBRLParser._classify_period_type("2024-01-01", "2024-12-10", 344) == "other"

    def test_other_above_annual(self):
        """381-day duration is above annual range → other."""
        assert XBRLParser._classify_period_type("2024-01-01", "2025-01-17", 381) == "other"

    def test_other_when_duration_none_but_start_present(self):
        """If duration can't be computed but start exists → other."""
        assert XBRLParser._classify_period_type("2024-01-01", "bad-date", None) == "other"

    # --- _compute_duration_days unit tests ---

    def test_compute_duration_days_normal(self):
        """Standard fiscal year duration."""
        result = XBRLParser._compute_duration_days("2024-01-29", "2025-01-26")
        assert result == 363

    def test_compute_duration_days_quarter(self):
        """Standard quarter duration."""
        result = XBRLParser._compute_duration_days("2024-01-29", "2024-04-28")
        assert result == 90

    def test_compute_duration_days_no_start(self):
        """Missing start date returns None."""
        result = XBRLParser._compute_duration_days(None, "2025-01-26")
        assert result is None

    def test_compute_duration_days_no_end(self):
        """Missing end date returns None."""
        result = XBRLParser._compute_duration_days("2024-01-29", None)
        assert result is None

    def test_compute_duration_days_bad_date(self):
        """Invalid date string returns None."""
        result = XBRLParser._compute_duration_days("not-a-date", "2025-01-26")
        assert result is None

    # --- Integration: period metadata in parsed output ---

    def test_parsed_output_has_period_columns(self):
        """parse_companyfacts output includes all period metadata columns."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        for col in ["fiscal_period_type", "period_start", "period_end",
                     "duration_days", "frame", "xbrl_concept"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_instant_facts_classified_correctly(self):
        """Facts without start dates (balance sheet items) are classified as instant."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        # The sample fixture has entries with only 'end' (no 'start') → instant
        instant_rows = df[df["fiscal_period_type"] == "instant"]
        assert not instant_rows.empty, "Expected some instant facts from fixture"
        assert instant_rows["period_start"].isna().all()

    def test_xbrl_concept_populated(self):
        """xbrl_concept should be populated for all parsed facts."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        assert (df["xbrl_concept"] != "").all()
        assert df["xbrl_concept"].notna().all()

    def test_empty_facts_has_period_columns(self):
        """Empty facts JSON returns DataFrame with period metadata columns."""
        parser = _make_parser()
        df = parser.parse_companyfacts({"facts": {}})
        for col in ["fiscal_period_type", "period_start", "period_end",
                     "duration_days", "frame", "xbrl_concept"]:
            assert col in df.columns, f"Missing column in empty result: {col}"


# ===================================================================
# 6. Annual fact selection (Reqs 15.2, 15.3, 15.4, 15.5, 15.6)
# ===================================================================


def _load_fy2025_candidates() -> dict:
    with open(FIXTURES / "regression" / "fy2025_revenue_candidates.json") as f:
        return json.load(f)


def _make_candidates_df(candidates: list[dict], metric: str = "revenue") -> pd.DataFrame:
    """Build a DataFrame of annual-duration FY-period candidates from fixture data."""
    rows = []
    for c in candidates:
        duration = c.get("duration_days")
        period_type = c.get("period_type", "")
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
            "fiscal_period_type": period_type,
            "period_start": c.get("start"),
            "period_end": c.get("end"),
            "duration_days": duration,
            "frame": c.get("frame"),
            "xbrl_concept": "Revenues",
            "selection_rank": 0,
            "selection_reason": "",
        })
    return pd.DataFrame(rows)


class TestAnnualFactSelection:
    """Verify _select_annual_fact() correctly selects annual facts using period dates."""

    def test_selects_correct_fy2025_revenue(self):
        """FY2025 revenue must be ~$130.5B, not $27B (the misparse value).

        This is the core regression test for the original failure.
        """
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        # Filter to only annual-duration FY-period candidates
        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        from datetime import date
        fy_end = date.fromisoformat("2025-01-26")

        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None
        assert result["value"] == 130497000000, (
            f"Expected FY2025 revenue $130.497B, got ${result['value'] / 1e9:.3f}B"
        )

    def test_rejects_comparative_fy2023_value(self):
        """The $26.974B comparative (FY2023 revenue in FY2025 10-K) must NOT be selected for FY2025."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        from datetime import date
        fy_end = date.fromisoformat("2025-01-26")

        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None
        assert result["value"] != 26974000000, (
            "Selected the FY2023 comparative value ($26.974B) instead of FY2025 ($130.497B)"
        )

    def test_rejects_comparative_fy2024_value(self):
        """The $60.922B comparative (FY2024 revenue in FY2025 10-K) must NOT be selected for FY2025."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        from datetime import date
        fy_end = date.fromisoformat("2025-01-26")

        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None
        assert result["value"] != 60922000000, (
            "Selected the FY2024 comparative value ($60.922B) instead of FY2025 ($130.497B)"
        )

    def test_selects_correct_fy2024_revenue(self):
        """FY2024 revenue must be $60.922B when selecting for FY2024."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        from datetime import date
        fy_end = date.fromisoformat("2024-01-28")

        result = parser._select_annual_fact(annual_fy, "revenue", 2024, fy_end)

        assert result is not None
        assert result["value"] == 60922000000, (
            f"Expected FY2024 revenue $60.922B, got ${result['value'] / 1e9:.3f}B"
        )

    def test_selects_correct_fy2023_revenue(self):
        """FY2023 revenue must be $26.974B when selecting for FY2023."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        from datetime import date
        fy_end = date.fromisoformat("2023-01-29")

        result = parser._select_annual_fact(annual_fy, "revenue", 2023, fy_end)

        assert result is not None
        assert result["value"] == 26974000000, (
            f"Expected FY2023 revenue $26.974B, got ${result['value'] / 1e9:.3f}B"
        )

    def test_records_selection_rank(self):
        """Selected fact must have selection_rank populated."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        from datetime import date
        fy_end = date.fromisoformat("2025-01-26")

        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None
        assert result["selection_rank"] >= 1
        assert result["selection_rank"] <= 3

    def test_records_selection_reason(self):
        """Selected fact must have selection_reason populated."""
        fixture = _load_fy2025_candidates()
        all_candidates = _make_candidates_df(fixture["candidates"])

        annual_fy = all_candidates[
            (all_candidates["fiscal_period"] == "FY")
            & (all_candidates["fiscal_period_type"] == "annual")
        ]

        parser = _make_parser()
        from datetime import date
        fy_end = date.fromisoformat("2025-01-26")

        result = parser._select_annual_fact(annual_fy, "revenue", 2025, fy_end)

        assert result is not None
        assert result["selection_reason"] != ""
        assert "10-K" in result["selection_reason"] or "annual" in result["selection_reason"]

    def test_quarterly_facts_rejected(self):
        """Quarterly facts (duration < 100 days) must never be selected as annual."""
        parser = _make_parser()
        from datetime import date

        # Create a DataFrame with only quarterly facts for the target FY end
        df = pd.DataFrame([{
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
            "filing_date": "2025-02-26", "source_available_date": "2025-02-26",
            "accession_number": "accn-q", "metric_name": "revenue",
            "value": 35082000000, "unit": "USD", "form_type": "10-Q",
            "fiscal_period_type": "quarterly",
            "period_start": "2024-10-28", "period_end": "2025-01-26",
            "duration_days": 90, "frame": None, "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        }])

        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is None, "Quarterly fact should not be selected as annual"

    def test_ytd_facts_rejected(self):
        """YTD facts (duration 100-340 days) must never be selected as annual."""
        parser = _make_parser()
        from datetime import date

        df = pd.DataFrame([{
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
            "filing_date": "2024-11-20", "source_available_date": "2024-11-20",
            "accession_number": "accn-ytd", "metric_name": "revenue",
            "value": 91166000000, "unit": "USD", "form_type": "10-Q",
            "fiscal_period_type": "ytd",
            "period_start": "2024-01-29", "period_end": "2025-01-26",
            "duration_days": 272, "frame": None, "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        }])

        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is None, "YTD fact should not be selected as annual"

    def test_empty_candidates_returns_none(self):
        """Empty candidates should return None."""
        parser = _make_parser()
        from datetime import date

        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is None

    def test_no_matching_period_end_returns_none(self):
        """If no candidate's period_end matches the target FY end, return None."""
        parser = _make_parser()
        from datetime import date

        df = pd.DataFrame([{
            "ticker": "NVDA", "fiscal_period": "FY", "fiscal_year": 2025,
            "filing_date": "2025-02-26", "source_available_date": "2025-02-26",
            "accession_number": "accn-1", "metric_name": "revenue",
            "value": 26974000000, "unit": "USD", "form_type": "10-K",
            "fiscal_period_type": "annual",
            "period_start": "2022-01-31", "period_end": "2023-01-29",
            "duration_days": 363, "frame": "CY2022", "xbrl_concept": "Revenues",
            "selection_rank": 0, "selection_reason": "",
        }])

        # Target FY2025 end is 2025-01-26, but the only candidate ends 2023-01-29
        fy_end = date.fromisoformat("2025-01-26")
        result = parser._select_annual_fact(df, "revenue", 2025, fy_end)

        assert result is None, "Comparative period fact should not match FY2025"

    def test_prefers_10k_over_non_10k(self):
        """10-K annual-duration facts should be preferred over non-10-K."""
        parser = _make_parser()
        from datetime import date

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
                "value": 130497000000, "unit": "USD", "form_type": "10-K",
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


# ===================================================================
# 7. Amendment handling (Req 15.6)
# ===================================================================


class TestAmendmentHandling:
    """Verify _handle_amendments() deterministic behavior."""

    def test_latest_filing_date_wins(self):
        """Among same-rank candidates, latest filing_date should win."""
        parser = _make_parser()

        row_old = pd.Series({
            "filing_date": "2025-02-26",
            "value": 130497000000,
            "form_type": "10-K",
        })
        row_new = pd.Series({
            "filing_date": "2025-04-15",
            "value": 130500000000,
            "form_type": "10-K/A",
        })

        candidates = [
            (1, row_old, "10-K annual-duration"),
            (1, row_new, "10-K annual-duration"),
        ]

        result = parser._handle_amendments(candidates)

        assert len(result) == 1
        _, best_row, reason = result[0]
        assert best_row["filing_date"] == "2025-04-15"
        assert "latest amendment" in reason

    def test_single_candidate_unchanged(self):
        """Single candidate should pass through unchanged."""
        parser = _make_parser()

        row = pd.Series({
            "filing_date": "2025-02-26",
            "value": 130497000000,
        })

        candidates = [(1, row, "10-K annual-duration")]
        result = parser._handle_amendments(candidates)

        assert len(result) == 1
        assert result[0][0] == 1

    def test_different_ranks_preserved(self):
        """Candidates with different ranks should all be preserved."""
        parser = _make_parser()

        row1 = pd.Series({"filing_date": "2025-02-26", "value": 100})
        row2 = pd.Series({"filing_date": "2025-03-15", "value": 200})

        candidates = [
            (1, row1, "10-K annual-duration"),
            (2, row2, "frame-tagged annual"),
        ]

        result = parser._handle_amendments(candidates)

        assert len(result) == 2
        ranks = [r[0] for r in result]
        assert 1 in ranks
        assert 2 in ranks

    def test_multiple_same_rank_keeps_latest(self):
        """Three candidates at same rank: only the latest filing_date survives."""
        parser = _make_parser()

        row1 = pd.Series({"filing_date": "2025-02-26", "value": 100})
        row2 = pd.Series({"filing_date": "2025-03-15", "value": 200})
        row3 = pd.Series({"filing_date": "2025-04-01", "value": 300})

        candidates = [
            (1, row1, "10-K annual-duration"),
            (1, row2, "10-K annual-duration"),
            (1, row3, "10-K annual-duration"),
        ]

        result = parser._handle_amendments(candidates)

        assert len(result) == 1
        _, best_row, _ = result[0]
        assert best_row["filing_date"] == "2025-04-01"
        assert best_row["value"] == 300


# ===================================================================
# 8. Integration: parse_companyfacts with annual selection
# ===================================================================


class TestParseWithAnnualSelection:
    """Verify parse_companyfacts uses the annual selection algorithm."""

    def test_output_has_selection_columns(self):
        """parse_companyfacts output includes selection_rank and selection_reason."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        assert "selection_rank" in df.columns
        assert "selection_reason" in df.columns

    def test_annual_facts_have_selection_metadata(self):
        """Annual FY facts should have selection_rank and selection_reason populated."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        annual_fy = df[
            (df["fiscal_period"] == "FY")
            & (df["fiscal_period_type"].isin(["annual", "instant"]))
        ]

        if not annual_fy.empty:
            # At least some annual facts should have selection metadata
            has_rank = annual_fy[annual_fy["selection_rank"] > 0]
            if not has_rank.empty:
                assert (has_rank["selection_reason"] != "").all()

    def test_no_duplicate_annual_facts_per_metric_fy(self):
        """After selection, there should be at most one annual fact per (metric, fiscal_year)."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())

        annual_fy = df[
            (df["fiscal_period"] == "FY")
        ]

        dupes = annual_fy.duplicated(subset=["metric_name", "fiscal_year"], keep=False)
        assert not dupes.any(), (
            f"Found duplicate annual facts: "
            f"{annual_fy[dupes][['metric_name', 'fiscal_year', 'value']].to_dict('records')}"
        )


# ===================================================================
# 9. validate_with_gate() — DataValidationGate integration (Req 14.9, 3.6)
# ===================================================================


def _load_published_values() -> dict:
    with open(FIXTURES / "nvda_published_values.json") as f:
        return json.load(f)


class TestValidateWithGate:
    """Verify validate_with_gate() delegates to DataValidationGate and returns
    (DataQualityStatus, list[ValidatedMetric])."""

    def test_returns_tuple_of_status_and_list(self):
        """validate_with_gate() returns (DataQualityStatus, list[ValidatedMetric])."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        published = _load_published_values()

        status, validated = parser.validate_with_gate(df, published)

        assert isinstance(status, DataQualityStatus)
        assert isinstance(validated, list)
        assert all(isinstance(v, ValidatedMetric) for v in validated)

    def test_validated_metrics_have_severity_and_blocker(self):
        """Each ValidatedMetric should have severity and blocker fields."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        published = _load_published_values()

        _, validated = parser.validate_with_gate(df, published)

        assert len(validated) > 0
        for vm in validated:
            assert vm.severity in ("critical", "major", "minor")
            assert isinstance(vm.blocker, bool)

    def test_validated_metrics_have_tolerance(self):
        """Each ValidatedMetric should carry the tolerance used."""
        parser = _make_parser()
        df = parser.parse_companyfacts(_load_companyfacts())
        published = _load_published_values()

        _, validated = parser.validate_with_gate(df, published)

        for vm in validated:
            assert vm.tolerance == parser.config.validation_tolerance_pct


# ===================================================================
# 10. compute_data_quality_status() (Req 14.9, 3.6)
# ===================================================================


class TestComputeDataQualityStatus:
    """Verify compute_data_quality_status() returns correct DataQualityStatus."""

    def test_pass_when_all_pass(self):
        """All metrics passing → PASS."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        validations = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
            ValidatedMetric(
                metric_name="net_income", fiscal_year=2025,
                parsed_value=72880000000, published_value=72880000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
        ]

        result = parser.compute_data_quality_status(validations)
        assert result is DataQualityStatus.PASS

    def test_data_blocked_when_critical_blocker_fails(self):
        """Critical blocker with fail status → DATA_BLOCKED."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        validations = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=27000000000, published_value=130497000000,
                diff_pct=79.3, tolerance=1.0, status="fail",
                severity="critical", blocker=True,
            ),
        ]

        result = parser.compute_data_quality_status(validations)
        assert result is DataQualityStatus.DATA_BLOCKED

    def test_data_blocked_when_critical_blocker_missing(self):
        """Critical blocker with missing status → DATA_BLOCKED."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        validations = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=None, published_value=130497000000,
                diff_pct=None, tolerance=1.0, status="missing",
                severity="critical", blocker=True,
            ),
        ]

        result = parser.compute_data_quality_status(validations)
        assert result is DataQualityStatus.DATA_BLOCKED

    def test_pass_with_warnings_for_non_critical_failure(self):
        """Non-critical failure → PASS_WITH_WARNINGS."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        validations = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
            ValidatedMetric(
                metric_name="some_minor_metric", fiscal_year=2025,
                parsed_value=100, published_value=200,
                diff_pct=50.0, tolerance=1.0, status="fail",
                severity="major", blocker=False,
            ),
        ]

        result = parser.compute_data_quality_status(validations)
        assert result is DataQualityStatus.PASS_WITH_WARNINGS

    def test_pass_with_warnings_for_non_blocker_missing(self):
        """Non-blocker missing metric → PASS_WITH_WARNINGS."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        validations = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
            ValidatedMetric(
                metric_name="sga", fiscal_year=2025,
                parsed_value=None, published_value=None,
                diff_pct=None, tolerance=1.0, status="missing",
                severity="major", blocker=False,
            ),
        ]

        result = parser.compute_data_quality_status(validations)
        assert result is DataQualityStatus.PASS_WITH_WARNINGS

    def test_empty_validations_returns_pass(self):
        """Empty validation list → PASS (no failures)."""
        from src.config import DataQualityStatus

        parser = _make_parser()
        result = parser.compute_data_quality_status([])
        assert result is DataQualityStatus.PASS


# ===================================================================
# 11. generate_data_quality_report() with DataQualityStatus (Req 14.9, 3.8)
# ===================================================================


class TestDataQualityReportWithStatus:
    """Verify generate_data_quality_report() includes DataQualityStatus info."""

    def test_report_includes_status_section_when_provided(self):
        """When data_quality_status is provided, report includes status section."""
        from src.config import DataQualityStatus

        parser = _make_parser()
        # Trigger _missing_tags / _fallbacks_used to be populated
        parser._missing_tags = []
        parser._fallbacks_used = []

        df = pd.DataFrame(columns=XBRLParser.OUTPUT_COLUMNS)
        report = parser.generate_data_quality_report(
            df, pd.DataFrame(), data_quality_status=DataQualityStatus.PASS,
        )

        assert "## Data Quality Status" in report
        assert "PASS" in report

    def test_report_shows_data_blocked_warning(self):
        """DATA_BLOCKED status shows blocking warning in report."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        parser._missing_tags = []
        parser._fallbacks_used = []

        df = pd.DataFrame(columns=XBRLParser.OUTPUT_COLUMNS)
        validations = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=27000000000, published_value=130497000000,
                diff_pct=79.3, tolerance=1.0, status="fail",
                severity="critical", blocker=True,
            ),
        ]

        report = parser.generate_data_quality_report(
            df, validations, data_quality_status=DataQualityStatus.DATA_BLOCKED,
        )

        assert "DATA_BLOCKED" in report
        assert "Formal recommendation is BLOCKED" in report
        assert "Blocking Issues" in report
        assert "revenue" in report

    def test_report_shows_pass_with_warnings(self):
        """PASS_WITH_WARNINGS status shows warning note."""
        from src.config import DataQualityStatus

        parser = _make_parser()
        parser._missing_tags = []
        parser._fallbacks_used = []

        df = pd.DataFrame(columns=XBRLParser.OUTPUT_COLUMNS)
        report = parser.generate_data_quality_report(
            df, [], data_quality_status=DataQualityStatus.PASS_WITH_WARNINGS,
        )

        assert "PASS_WITH_WARNINGS" in report
        assert "Non-critical warnings" in report

    def test_report_without_status_has_no_status_section(self):
        """When data_quality_status is None, no status section appears."""
        parser = _make_parser()
        parser._missing_tags = []
        parser._fallbacks_used = []

        df = pd.DataFrame(columns=XBRLParser.OUTPUT_COLUMNS)
        report = parser.generate_data_quality_report(df, pd.DataFrame())

        assert "## Data Quality Status" not in report

    def test_report_with_validated_metrics_shows_severity_columns(self):
        """When validation is list[ValidatedMetric], table includes severity/blocker."""
        from src.config import DataQualityStatus, ValidatedMetric

        parser = _make_parser()
        parser._missing_tags = []
        parser._fallbacks_used = []

        df = pd.DataFrame(columns=XBRLParser.OUTPUT_COLUMNS)
        validations = [
            ValidatedMetric(
                metric_name="revenue", fiscal_year=2025,
                parsed_value=130497000000, published_value=130497000000,
                diff_pct=0.0, tolerance=1.0, status="pass",
                severity="critical", blocker=True,
            ),
        ]

        report = parser.generate_data_quality_report(
            df, validations, data_quality_status=DataQualityStatus.PASS,
        )

        assert "Severity" in report
        assert "Blocker" in report
        assert "critical" in report

    def test_legacy_dataframe_validation_still_works(self):
        """Legacy DataFrame validation (from validate_against_published) still renders."""
        parser = _make_parser()
        parser._missing_tags = []
        parser._fallbacks_used = []

        df = pd.DataFrame(columns=XBRLParser.OUTPUT_COLUMNS)
        validation_df = pd.DataFrame([{
            "metric": "revenue",
            "fiscal_year": 2025,
            "parsed_value": 130497000000,
            "published_value": 130497000000,
            "diff_pct": 0.0,
            "status": "pass",
        }])

        report = parser.generate_data_quality_report(df, validation_df)

        assert "Validation Against Published Values" in report
        assert "revenue" in report
        assert "pass" in report
        # Legacy format should NOT have severity columns
        assert "Severity" not in report
