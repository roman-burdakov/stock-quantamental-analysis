"""
Tests for src.split_adjuster — Stock split adjustment utility.

Covers:
  - Raw values are preserved (raw_value never modified)
  - Adjusted values are reproducible from raw + factor
  - Pre-split EPS and shares are adjusted to post-split basis
  - Post-split facts are left unchanged
  - Non-per-share metrics are unaffected by split adjustment
  - Consistent basis across split boundary

Reqs: 3.1, 3.2
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import EngineConfig
from src.split_adjuster import PER_SHARE_METRICS, SplitAdjuster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adjuster(**config_overrides) -> SplitAdjuster:
    """Create a SplitAdjuster with default NVIDIA config."""
    config = EngineConfig(**config_overrides)
    return SplitAdjuster(config)


def _make_fact_row(
    metric_name: str,
    value: float | None,
    period_end: str,
    *,
    fiscal_year: int = 2024,
    fiscal_period: str = "FY",
    form_type: str = "10-K",
    fiscal_period_type: str = "annual",
    period_start: str = "2023-01-30",
    duration_days: int = 363,
) -> dict:
    """Build a single fact row matching XBRLParser output schema."""
    return {
        "ticker": "NVDA",
        "fiscal_period": fiscal_period,
        "fiscal_year": fiscal_year,
        "filing_date": "2024-02-21",
        "source_available_date": "2024-02-21",
        "accession_number": "0001045810-24-000029",
        "metric_name": metric_name,
        "value": value,
        "unit": "USD/shares" if metric_name == "diluted_eps" else (
            "shares" if metric_name == "diluted_shares" else "USD"
        ),
        "form_type": form_type,
        "fiscal_period_type": fiscal_period_type,
        "period_start": period_start,
        "period_end": period_end,
        "duration_days": duration_days,
        "frame": None,
        "xbrl_concept": "EarningsPerShareDiluted" if metric_name == "diluted_eps" else (
            "WeightedAverageNumberOfDilutedSharesOutstanding"
            if metric_name == "diluted_shares" else "Revenues"
        ),
        "selection_rank": 1,
        "selection_reason": "10-K annual duration",
    }


def _build_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from row dicts."""
    return pd.DataFrame(rows)


# ===================================================================
# 1. SplitAdjuster initialisation and config parsing
# ===================================================================


class TestSplitAdjusterInit:
    """Verify SplitAdjuster correctly parses split_history from config."""

    def test_default_config_has_nvidia_split(self):
        """Default EngineConfig includes the NVIDIA 10:1 split."""
        adjuster = _make_adjuster()
        assert len(adjuster._splits) == 1
        assert adjuster._splits[0]["ratio"] == 10
        assert adjuster._splits[0]["date"].isoformat() == "2024-06-10"

    def test_empty_split_history(self):
        """No splits → empty internal list."""
        adjuster = _make_adjuster(split_history=[])
        assert adjuster._splits == []

    def test_invalid_split_entry_skipped(self):
        """Invalid split entries are skipped with a warning."""
        adjuster = _make_adjuster(split_history=[
            {"date": "not-a-date", "ratio": 10},
            {"date": "2024-06-10", "ratio": 10, "description": "valid"},
        ])
        assert len(adjuster._splits) == 1
        assert adjuster._splits[0]["description"] == "valid"

    def test_ratio_lte_1_skipped(self):
        """Split with ratio <= 1 is skipped."""
        adjuster = _make_adjuster(split_history=[
            {"date": "2024-06-10", "ratio": 1},
            {"date": "2024-06-10", "ratio": 0.5},
        ])
        assert adjuster._splits == []


# ===================================================================
# 2. Adjustment factor computation
# ===================================================================


class TestComputeAdjustmentFactor:
    """Verify compute_adjustment_factor for pre-split and post-split dates."""

    def test_pre_split_period_gets_factor_10(self):
        """Period ending before June 10, 2024 → factor = 10."""
        adjuster = _make_adjuster()
        factor = adjuster.compute_adjustment_factor("2024-01-28")
        assert factor == 10.0

    def test_post_split_period_gets_factor_1(self):
        """Period ending on or after June 10, 2024 → factor = 1."""
        adjuster = _make_adjuster()
        factor = adjuster.compute_adjustment_factor("2025-01-26")
        assert factor == 1.0

    def test_split_date_itself_gets_factor_1(self):
        """Period ending exactly on split date → factor = 1 (not pre-split)."""
        adjuster = _make_adjuster()
        factor = adjuster.compute_adjustment_factor("2024-06-10")
        assert factor == 1.0

    def test_day_before_split_gets_factor_10(self):
        """Period ending one day before split → factor = 10."""
        adjuster = _make_adjuster()
        factor = adjuster.compute_adjustment_factor("2024-06-09")
        assert factor == 10.0

    def test_none_period_end_returns_1(self):
        """None period_end → no adjustment."""
        adjuster = _make_adjuster()
        assert adjuster.compute_adjustment_factor(None) == 1.0

    def test_invalid_date_returns_1(self):
        """Invalid date string → no adjustment."""
        adjuster = _make_adjuster()
        assert adjuster.compute_adjustment_factor("bad-date") == 1.0

    def test_no_splits_returns_1(self):
        """No split history → always 1.0."""
        adjuster = _make_adjuster(split_history=[])
        assert adjuster.compute_adjustment_factor("2020-01-01") == 1.0

    def test_multiple_splits_cumulative(self):
        """Multiple splits accumulate multiplicatively."""
        adjuster = _make_adjuster(split_history=[
            {"date": "2020-07-20", "ratio": 4, "description": "4:1 split"},
            {"date": "2024-06-10", "ratio": 10, "description": "10:1 split"},
        ])
        # Before both splits
        factor = adjuster.compute_adjustment_factor("2020-01-01")
        assert factor == 40.0  # 4 * 10

        # Between splits
        factor = adjuster.compute_adjustment_factor("2022-01-01")
        assert factor == 10.0  # only the 2024 split applies

        # After both splits
        factor = adjuster.compute_adjustment_factor("2025-01-01")
        assert factor == 1.0


# ===================================================================
# 3. Value adjustment logic
# ===================================================================


class TestAdjustValue:
    """Verify adjust_value computes correct split-adjusted values."""

    def test_diluted_eps_pre_split_divided(self):
        """Pre-split EPS is divided by the split ratio."""
        adjuster = _make_adjuster()
        # FY2024 EPS was $11.93 pre-split → $1.193 post-split
        result = adjuster.adjust_value("diluted_eps", 11.93, 10.0)
        assert result == pytest.approx(1.193)

    def test_diluted_shares_pre_split_multiplied(self):
        """Pre-split shares are multiplied by the split ratio."""
        adjuster = _make_adjuster()
        # FY2024 shares were 2.494B pre-split → 24.94B post-split
        result = adjuster.adjust_value("diluted_shares", 2494000000, 10.0)
        assert result == pytest.approx(24940000000)

    def test_post_split_eps_unchanged(self):
        """Post-split EPS (factor=1.0) is unchanged."""
        adjuster = _make_adjuster()
        result = adjuster.adjust_value("diluted_eps", 2.94, 1.0)
        assert result == 2.94

    def test_post_split_shares_unchanged(self):
        """Post-split shares (factor=1.0) are unchanged."""
        adjuster = _make_adjuster()
        result = adjuster.adjust_value("diluted_shares", 24804000000, 1.0)
        assert result == 24804000000

    def test_non_per_share_metric_unchanged(self):
        """Revenue (non-per-share) is never adjusted regardless of factor."""
        adjuster = _make_adjuster()
        result = adjuster.adjust_value("revenue", 60922000000, 10.0)
        assert result == 60922000000

    def test_none_value_returns_none(self):
        """None input → None output."""
        adjuster = _make_adjuster()
        assert adjuster.adjust_value("diluted_eps", None, 10.0) is None


# ===================================================================
# 4. DataFrame-level split adjustment
# ===================================================================


class TestApplySplitAdjustments:
    """Verify apply_split_adjustments on a full DataFrame."""

    def test_raw_value_always_preserved(self):
        """raw_value must equal the original value column, never overwritten."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_eps", 11.93, "2024-01-28", fiscal_year=2024),
            _make_fact_row("diluted_eps", 2.94, "2025-01-26", fiscal_year=2025),
            _make_fact_row("revenue", 60922000000, "2024-01-28", fiscal_year=2024),
        ])

        result = adjuster.apply_split_adjustments(df)

        assert (result["raw_value"] == result["value"]).all(), (
            "raw_value must always equal the original value"
        )

    def test_pre_split_eps_adjusted(self):
        """Pre-split EPS gets adjusted_value = raw / 10."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_eps", 11.93, "2024-01-28", fiscal_year=2024),
        ])

        result = adjuster.apply_split_adjustments(df)
        row = result.iloc[0]

        assert row["raw_value"] == 11.93
        assert row["adjusted_value"] == pytest.approx(1.193)
        assert row["adjustment_factor"] == 10.0
        assert row["adjustment_basis"] == "split_adjusted_to_current"

    def test_post_split_eps_unchanged(self):
        """Post-split EPS has adjusted_value = raw_value, factor = 1.0."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_eps", 2.94, "2025-01-26", fiscal_year=2025),
        ])

        result = adjuster.apply_split_adjustments(df)
        row = result.iloc[0]

        assert row["raw_value"] == 2.94
        assert row["adjusted_value"] == 2.94
        assert row["adjustment_factor"] == 1.0
        assert row["adjustment_basis"] == "as_reported"

    def test_pre_split_shares_adjusted(self):
        """Pre-split diluted_shares gets adjusted_value = raw * 10."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_shares", 2494000000, "2024-01-28", fiscal_year=2024),
        ])

        result = adjuster.apply_split_adjustments(df)
        row = result.iloc[0]

        assert row["raw_value"] == 2494000000
        assert row["adjusted_value"] == pytest.approx(24940000000)
        assert row["adjustment_factor"] == 10.0
        assert row["adjustment_basis"] == "split_adjusted_to_current"

    def test_revenue_not_adjusted(self):
        """Revenue (non-per-share) is never split-adjusted."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("revenue", 60922000000, "2024-01-28", fiscal_year=2024),
        ])

        result = adjuster.apply_split_adjustments(df)
        row = result.iloc[0]

        assert row["raw_value"] == 60922000000
        assert row["adjusted_value"] == 60922000000
        assert row["adjustment_factor"] == 1.0
        assert row["adjustment_basis"] == "as_reported"

    def test_empty_df_returns_empty(self):
        """Empty DataFrame returns empty with no errors."""
        adjuster = _make_adjuster()
        df = pd.DataFrame()

        result = adjuster.apply_split_adjustments(df)
        assert result.empty

    def test_no_splits_all_as_reported(self):
        """With no split history, all facts are as_reported."""
        adjuster = _make_adjuster(split_history=[])
        df = _build_df([
            _make_fact_row("diluted_eps", 11.93, "2024-01-28", fiscal_year=2024),
            _make_fact_row("diluted_shares", 2494000000, "2024-01-28", fiscal_year=2024),
        ])

        result = adjuster.apply_split_adjustments(df)

        assert (result["adjustment_factor"] == 1.0).all()
        assert (result["adjustment_basis"] == "as_reported").all()
        assert (result["raw_value"] == result["value"]).all()
        assert (result["adjusted_value"] == result["value"]).all()

    def test_validation_basis_set_for_pre_split(self):
        """Pre-split per-share facts get validation_basis = post_split_restated."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_eps", 1.74, "2023-01-29", fiscal_year=2023),
        ])

        result = adjuster.apply_split_adjustments(df)
        assert result.iloc[0]["validation_basis"] == "post_split_restated"

    def test_validation_basis_as_reported_for_post_split(self):
        """Post-split facts get validation_basis = as_reported."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_eps", 2.94, "2025-01-26", fiscal_year=2025),
        ])

        result = adjuster.apply_split_adjustments(df)
        assert result.iloc[0]["validation_basis"] == "as_reported"


# ===================================================================
# 5. Cross-split consistency verification
# ===================================================================


class TestCrossSplitConsistency:
    """Verify pre-split and post-split EPS/shares are on consistent basis after adjustment."""

    def test_eps_consistent_across_split_boundary(self):
        """After adjustment, all EPS values should be on post-split basis.

        FY2023 EPS: $1.74 pre-split → $0.174 post-split
        FY2024 EPS: $11.93 pre-split → $1.193 post-split
        FY2025 EPS: $2.94 post-split (native)

        The adjusted values should show the real earnings growth trajectory.
        """
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_eps", 1.74, "2023-01-29", fiscal_year=2023,
                           period_start="2022-01-31"),
            _make_fact_row("diluted_eps", 11.93, "2024-01-28", fiscal_year=2024,
                           period_start="2023-01-30"),
            _make_fact_row("diluted_eps", 2.94, "2025-01-26", fiscal_year=2025,
                           period_start="2024-01-29"),
        ])

        result = adjuster.apply_split_adjustments(df)

        # All adjusted EPS should be on post-split basis
        adj_eps = result["adjusted_value"].tolist()
        assert adj_eps[0] == pytest.approx(0.174)   # FY2023: 1.74 / 10
        assert adj_eps[1] == pytest.approx(1.193)    # FY2024: 11.93 / 10
        assert adj_eps[2] == pytest.approx(2.94)     # FY2025: native post-split

        # Verify growth trajectory makes sense (EPS grew significantly)
        assert adj_eps[1] > adj_eps[0]  # FY2024 > FY2023
        assert adj_eps[2] > adj_eps[1]  # FY2025 > FY2024

    def test_shares_consistent_across_split_boundary(self):
        """After adjustment, all share counts should be on post-split basis.

        FY2023 shares: 2.507B pre-split → 25.07B post-split
        FY2024 shares: 2.494B pre-split → 24.94B post-split
        FY2025 shares: 24.804B post-split (native)
        """
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_shares", 2507000000, "2023-01-29",
                           fiscal_year=2023, period_start="2022-01-31"),
            _make_fact_row("diluted_shares", 2494000000, "2024-01-28",
                           fiscal_year=2024, period_start="2023-01-30"),
            _make_fact_row("diluted_shares", 24804000000, "2025-01-26",
                           fiscal_year=2025, period_start="2024-01-29"),
        ])

        result = adjuster.apply_split_adjustments(df)

        adj_shares = result["adjusted_value"].tolist()
        assert adj_shares[0] == pytest.approx(25070000000)  # FY2023: 2.507B * 10
        assert adj_shares[1] == pytest.approx(24940000000)  # FY2024: 2.494B * 10
        assert adj_shares[2] == pytest.approx(24804000000)  # FY2025: native

        # All should be in the same ~24-25B range (post-split)
        for s in adj_shares:
            assert 20_000_000_000 < s < 30_000_000_000, (
                f"Adjusted shares {s:,.0f} outside expected post-split range"
            )

    def test_adjusted_reproducible_from_raw_and_factor(self):
        """adjusted_value must be exactly reproducible from raw_value and adjustment_factor."""
        adjuster = _make_adjuster()
        df = _build_df([
            _make_fact_row("diluted_eps", 1.74, "2023-01-29", fiscal_year=2023),
            _make_fact_row("diluted_eps", 11.93, "2024-01-28", fiscal_year=2024),
            _make_fact_row("diluted_eps", 2.94, "2025-01-26", fiscal_year=2025),
            _make_fact_row("diluted_shares", 2507000000, "2023-01-29", fiscal_year=2023),
            _make_fact_row("diluted_shares", 2494000000, "2024-01-28", fiscal_year=2024),
            _make_fact_row("diluted_shares", 24804000000, "2025-01-26", fiscal_year=2025),
            _make_fact_row("revenue", 26974000000, "2023-01-29", fiscal_year=2023),
        ])

        result = adjuster.apply_split_adjustments(df)

        for _, row in result.iterrows():
            metric = row["metric_name"]
            raw = row["raw_value"]
            factor = row["adjustment_factor"]
            adjusted = row["adjusted_value"]

            if raw is None:
                assert adjusted is None
                continue

            # Reproduce the adjusted value
            expected = adjuster.adjust_value(metric, raw, factor)
            assert adjusted == pytest.approx(expected), (
                f"{metric} FY{row['fiscal_year']}: "
                f"adjusted={adjusted} != expected={expected} "
                f"(raw={raw}, factor={factor})"
            )


# ===================================================================
# 6. Integration with XBRLParser
# ===================================================================


class TestXBRLParserIntegration:
    """Verify split adjustment is applied during parse_companyfacts."""

    def test_parsed_output_has_split_columns(self):
        """parse_companyfacts output includes all split adjustment columns."""
        import json
        from pathlib import Path

        from src.xbrl_parser import XBRLParser

        with open(Path("tests/fixtures/sample_companyfacts.json")) as f:
            facts = json.load(f)

        parser = XBRLParser(EngineConfig())
        df = parser.parse_companyfacts(facts)

        for col in ["raw_value", "adjusted_value", "adjustment_factor",
                     "adjustment_basis", "validation_basis"]:
            assert col in df.columns, f"Missing split column: {col}"

    def test_raw_value_equals_value_in_parsed_output(self):
        """raw_value should always equal the original value column."""
        import json
        from pathlib import Path

        from src.xbrl_parser import XBRLParser

        with open(Path("tests/fixtures/sample_companyfacts.json")) as f:
            facts = json.load(f)

        parser = XBRLParser(EngineConfig())
        df = parser.parse_companyfacts(facts)

        assert (df["raw_value"] == df["value"]).all(), (
            "raw_value must always equal the original value"
        )

    def test_per_share_metrics_have_adjustment_metadata(self):
        """Per-share metrics should have adjustment_factor and basis populated."""
        import json
        from pathlib import Path

        from src.xbrl_parser import XBRLParser

        with open(Path("tests/fixtures/sample_companyfacts.json")) as f:
            facts = json.load(f)

        parser = XBRLParser(EngineConfig())
        df = parser.parse_companyfacts(facts)

        per_share = df[df["metric_name"].isin(PER_SHARE_METRICS)]
        if not per_share.empty:
            assert (per_share["adjustment_factor"] >= 1.0).all()
            assert per_share["adjustment_basis"].isin(
                ["as_reported", "split_adjusted_to_current"]
            ).all()
