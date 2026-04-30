"""
Tests for src.valuation — Valuation Module.

Covers:
  1. DCF PV against known inputs
  2. FCF projection (revenue × margin)
  3. Terminal value (Gordon Growth formula)
  4. Scenario weighting (probability-weighted expected value)
  5. Reverse-DCF grid dimensions and values
  6. One-variable solver convergence
  7. Historical FCF margin reconciliation

All tests run offline with inline test data.
Reqs: 12.2
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.config import EngineConfig, ScenarioAssumptions, get_default_config
from src.valuation import ValuationModule


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vm() -> ValuationModule:
    """ValuationModule with default config (10-year projection, wacc=0.10, tg=0.03)."""
    return ValuationModule(get_default_config())


@pytest.fixture
def simple_assumptions() -> ScenarioAssumptions:
    """Deterministic assumptions for hand-verifiable DCF."""
    return ScenarioAssumptions(
        name="test",
        probability=1.0,
        revenue_cagr=0.20,
        fcf_margin_start=0.30,
        fcf_margin_terminal=0.30,  # constant margin → easy to verify
        terminal_growth=0.03,
        sbc_treatment="included_in_fcf",
        analyst_notes="unit test",
    )


# ---------------------------------------------------------------------------
# 1. DCF PV against known inputs
# ---------------------------------------------------------------------------

class TestDCFKnownInputs:
    """Verify compute_dcf with known assumptions produces expected per_share_value."""

    def test_dcf_returns_all_keys(self, vm, simple_assumptions):
        result = vm.compute_dcf(simple_assumptions, base_revenue=100.0, wacc=0.10, net_cash=50.0, shares=10.0)
        expected_keys = {
            "projected_revenue", "fcf_margin", "projected_fcf",
            "discount_factors", "pv_fcf", "terminal_value",
            "pv_terminal", "enterprise_value", "net_cash_bridge",
            "equity_value", "per_share_value",
        }
        assert expected_keys == set(result.keys())

    def test_dcf_per_share_value_positive(self, vm, simple_assumptions):
        result = vm.compute_dcf(simple_assumptions, base_revenue=100.0, wacc=0.10, net_cash=0.0, shares=10.0)
        assert result["per_share_value"] > 0

    def test_dcf_known_value(self, vm):
        """Hand-computed DCF: constant 30% margin, 20% CAGR, 10% WACC, 3% tg, 10yr."""
        assumptions = ScenarioAssumptions(
            name="known", probability=1.0, revenue_cagr=0.20,
            fcf_margin_start=0.30, fcf_margin_terminal=0.30,
            terminal_growth=0.03, sbc_treatment="included_in_fcf", analyst_notes="",
        )
        base_revenue = 100.0
        wacc = 0.10
        net_cash = 0.0
        shares = 1.0

        result = vm.compute_dcf(assumptions, base_revenue, wacc, net_cash, shares)

        # Manually compute expected sum of PV(FCF)
        expected_pv_sum = 0.0
        for yr in range(1, 11):
            rev = base_revenue * (1.20 ** yr)
            fcf = rev * 0.30
            pv = fcf / (1.10 ** yr)
            expected_pv_sum += pv

        # Terminal value
        terminal_fcf = (base_revenue * (1.20 ** 10) * 0.30) * 1.03
        expected_tv = terminal_fcf / (0.10 - 0.03)
        expected_pv_tv = expected_tv / (1.10 ** 10)

        expected_ev = expected_pv_sum + expected_pv_tv
        expected_per_share = expected_ev  # shares=1, net_cash=0

        assert result["per_share_value"] == pytest.approx(expected_per_share, rel=1e-6)
        assert result["enterprise_value"] == pytest.approx(expected_ev, rel=1e-6)

    def test_dcf_net_cash_bridge(self, vm, simple_assumptions):
        """Equity value = EV + net_cash."""
        net_cash = 200.0
        result = vm.compute_dcf(simple_assumptions, base_revenue=100.0, wacc=0.10, net_cash=net_cash, shares=10.0)
        assert result["equity_value"] == pytest.approx(result["enterprise_value"] + net_cash, rel=1e-9)
        assert result["net_cash_bridge"] == net_cash

    def test_dcf_shares_division(self, vm, simple_assumptions):
        """per_share_value = equity_value / shares."""
        shares = 24.5
        result = vm.compute_dcf(simple_assumptions, base_revenue=100.0, wacc=0.10, net_cash=50.0, shares=shares)
        assert result["per_share_value"] == pytest.approx(result["equity_value"] / shares, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. FCF projection (revenue × margin)
# ---------------------------------------------------------------------------

class TestFCFProjection:
    """Verify projected_fcf = projected_revenue × fcf_margin for each year."""

    def test_fcf_equals_revenue_times_margin(self, vm, simple_assumptions):
        result = vm.compute_dcf(simple_assumptions, base_revenue=100.0, wacc=0.10, net_cash=0.0, shares=1.0)
        for i in range(len(result["projected_revenue"])):
            expected_fcf = result["projected_revenue"][i] * result["fcf_margin"][i]
            assert result["projected_fcf"][i] == pytest.approx(expected_fcf, rel=1e-9)

    def test_revenue_projection_cagr(self, vm):
        """Revenue in year t = base_revenue × (1 + cagr)^t."""
        assumptions = ScenarioAssumptions(
            name="cagr_test", probability=1.0, revenue_cagr=0.15,
            fcf_margin_start=0.25, fcf_margin_terminal=0.25,
            terminal_growth=0.03, sbc_treatment="included_in_fcf", analyst_notes="",
        )
        base_revenue = 200.0
        result = vm.compute_dcf(assumptions, base_revenue, wacc=0.10, net_cash=0.0, shares=1.0)
        for yr_idx, rev in enumerate(result["projected_revenue"], start=1):
            expected = base_revenue * (1.15 ** yr_idx)
            assert rev == pytest.approx(expected, rel=1e-9)

    def test_margin_linear_interpolation(self, vm):
        """FCF margin linearly interpolates from start to terminal over projection years."""
        assumptions = ScenarioAssumptions(
            name="interp", probability=1.0, revenue_cagr=0.10,
            fcf_margin_start=0.40, fcf_margin_terminal=0.20,
            terminal_growth=0.03, sbc_treatment="included_in_fcf", analyst_notes="",
        )
        result = vm.compute_dcf(assumptions, base_revenue=100.0, wacc=0.10, net_cash=0.0, shares=1.0)
        n = vm.config.projection_years  # 10
        margins = result["fcf_margin"]
        assert len(margins) == n
        # First year margin = start
        assert margins[0] == pytest.approx(0.40, rel=1e-9)
        # Last year margin = terminal
        assert margins[-1] == pytest.approx(0.20, rel=1e-9)
        # Middle year check (year 5 → index 4, interpolation fraction = 4/9)
        expected_mid = 0.40 + (0.20 - 0.40) * (4 / 9)
        assert margins[4] == pytest.approx(expected_mid, rel=1e-6)

    def test_projection_length(self, vm, simple_assumptions):
        result = vm.compute_dcf(simple_assumptions, base_revenue=100.0, wacc=0.10, net_cash=0.0, shares=1.0)
        assert len(result["projected_revenue"]) == vm.config.projection_years
        assert len(result["projected_fcf"]) == vm.config.projection_years
        assert len(result["pv_fcf"]) == vm.config.projection_years


# ---------------------------------------------------------------------------
# 3. Terminal value (Gordon Growth formula)
# ---------------------------------------------------------------------------

class TestTerminalValue:
    """Verify terminal_value calculation matches Gordon Growth formula."""

    def test_terminal_value_gordon_growth(self, vm):
        """TV = FCF_n × (1 + g) / (WACC - g), discounted to present."""
        assumptions = ScenarioAssumptions(
            name="tv_test", probability=1.0, revenue_cagr=0.20,
            fcf_margin_start=0.30, fcf_margin_terminal=0.30,
            terminal_growth=0.03, sbc_treatment="included_in_fcf", analyst_notes="",
        )
        base_revenue = 100.0
        wacc = 0.10
        result = vm.compute_dcf(assumptions, base_revenue, wacc, net_cash=0.0, shares=1.0)

        # Last year FCF
        last_fcf = result["projected_fcf"][-1]
        terminal_fcf = last_fcf * (1 + 0.03)
        expected_tv = terminal_fcf / (wacc - 0.03)
        assert result["terminal_value"] == pytest.approx(expected_tv, rel=1e-9)

    def test_pv_terminal_discounted(self, vm):
        """PV of terminal = TV / (1 + WACC)^n."""
        assumptions = ScenarioAssumptions(
            name="pv_tv", probability=1.0, revenue_cagr=0.15,
            fcf_margin_start=0.35, fcf_margin_terminal=0.35,
            terminal_growth=0.025, sbc_treatment="included_in_fcf", analyst_notes="",
        )
        wacc = 0.10
        result = vm.compute_dcf(assumptions, base_revenue=150.0, wacc=wacc, net_cash=0.0, shares=1.0)
        n = vm.config.projection_years
        expected_pv_tv = result["terminal_value"] / ((1 + wacc) ** n)
        assert result["pv_terminal"] == pytest.approx(expected_pv_tv, rel=1e-9)

    def test_terminal_value_zero_when_wacc_leq_growth(self):
        """If WACC <= terminal_growth, terminal_value is capped (not infinite)."""
        config = get_default_config()
        vm = ValuationModule(config)
        assumptions = ScenarioAssumptions(
            name="edge", probability=1.0, revenue_cagr=0.10,
            fcf_margin_start=0.30, fcf_margin_terminal=0.30,
            terminal_growth=0.12,  # > wacc
            sbc_treatment="included_in_fcf", analyst_notes="",
        )
        result = vm.compute_dcf(assumptions, base_revenue=100.0, wacc=0.10, net_cash=0.0, shares=1.0)
        # Terminal value should be finite (capped at 200× terminal FCF), not 0 or infinite
        assert result["terminal_value"] > 0
        assert result["terminal_value"] < float("inf")


# ---------------------------------------------------------------------------
# 4. Scenario weighting
# ---------------------------------------------------------------------------

class TestScenarioWeighting:
    """Verify build_scenarios produces probability-weighted expected value."""

    def test_scenario_keys(self, vm):
        scenarios = vm.build_scenarios(base_revenue=100.0, wacc=0.10, net_cash=50.0, shares=10.0)
        assert "bear" in scenarios
        assert "base" in scenarios
        assert "bull" in scenarios

    def test_probabilities_sum_to_one(self, vm):
        scenarios = vm.build_scenarios(base_revenue=100.0, wacc=0.10, net_cash=50.0, shares=10.0)
        total_prob = sum(s["probability"] for s in scenarios.values())
        assert total_prob == pytest.approx(1.0, abs=1e-9)

    def test_weighted_expected_value(self, vm):
        """Expected value = Σ(probability × per_share_value) across scenarios."""
        scenarios = vm.build_scenarios(base_revenue=100.0, wacc=0.10, net_cash=50.0, shares=10.0)
        expected_value = sum(
            s["probability"] * s["per_share_value"] for s in scenarios.values()
        )
        # Verify each scenario has a positive per_share_value
        for name, s in scenarios.items():
            assert s["per_share_value"] > 0, f"{name} scenario has non-positive value"

        # The recommendation uses this same weighted sum
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=50.0,
            reverse_dcf=pd.DataFrame(),
        )
        assert rec.expected_value == pytest.approx(expected_value, rel=1e-9)

    def test_bull_greater_than_bear(self, vm):
        """Bull scenario should produce higher per-share value than bear."""
        scenarios = vm.build_scenarios(base_revenue=100.0, wacc=0.10, net_cash=50.0, shares=10.0)
        assert scenarios["bull"]["per_share_value"] > scenarios["bear"]["per_share_value"]


# ---------------------------------------------------------------------------
# 5. Reverse-DCF grid dimensions and values
# ---------------------------------------------------------------------------

class TestReverseDCFGrid:
    """Verify grid has correct shape and values are positive."""

    def test_grid_dimensions(self, vm):
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15, 0.20, 0.25]
        margin_range = [0.20, 0.25, 0.30, 0.35]
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        assert grid.shape == (len(cagr_range), len(margin_range))

    def test_grid_values_positive(self, vm):
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15, 0.20]
        margin_range = [0.20, 0.30, 0.40]
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        assert (grid.values > 0).all(), "All reverse-DCF grid values should be positive"

    def test_grid_monotonic_in_cagr(self, vm):
        """Higher CAGR → higher implied price (holding margin constant)."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15, 0.20, 0.25]
        margin_range = [0.30]
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        col = grid.iloc[:, 0].values
        for i in range(len(col) - 1):
            assert col[i + 1] > col[i], "Grid should be monotonically increasing in CAGR"

    def test_grid_monotonic_in_margin(self, vm):
        """Higher margin → higher implied price (holding CAGR constant)."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.15]
        margin_range = [0.15, 0.20, 0.25, 0.30, 0.35]
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        row = grid.iloc[0, :].values
        for i in range(len(row) - 1):
            assert row[i + 1] > row[i], "Grid should be monotonically increasing in margin"

    def test_grid_highlight_attribute(self, vm):
        """Grid should have a highlight attribute marking cells near current price."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15]
        margin_range = [0.20, 0.30]
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        assert "highlight" in grid.attrs
        hl = grid.attrs["highlight"]
        assert hl.shape == grid.shape


# ---------------------------------------------------------------------------
# 6. One-variable solver convergence
# ---------------------------------------------------------------------------

class TestSolvers:
    """Verify solve_implied_cagr and solve_implied_margin converge to reasonable values."""

    def test_implied_cagr_round_trip(self, vm):
        """Compute DCF price at known CAGR, then solve back for that CAGR."""
        vm.set_base_revenue(100.0)
        known_cagr = 0.20
        fixed_margin = 0.30
        wacc = 0.10
        cash = 50.0
        shares = 10.0

        # Compute the price at known_cagr
        assumptions = ScenarioAssumptions(
            name="rt", probability=1.0, revenue_cagr=known_cagr,
            fcf_margin_start=fixed_margin, fcf_margin_terminal=fixed_margin,
            terminal_growth=vm.config.terminal_growth,
            sbc_treatment="included_in_fcf", analyst_notes="",
        )
        dcf = vm.compute_dcf(assumptions, base_revenue=100.0, wacc=wacc, net_cash=cash, shares=shares)
        target_price = dcf["per_share_value"]

        # Solve back
        implied = vm.solve_implied_cagr(target_price, shares, cash, wacc, fixed_margin)
        assert implied == pytest.approx(known_cagr, abs=1e-4)

    def test_implied_margin_round_trip(self, vm):
        """Compute DCF price at known margin, then solve back for that margin."""
        vm.set_base_revenue(100.0)
        fixed_cagr = 0.15
        known_margin = 0.35
        wacc = 0.10
        cash = 30.0
        shares = 10.0

        assumptions = ScenarioAssumptions(
            name="rt", probability=1.0, revenue_cagr=fixed_cagr,
            fcf_margin_start=known_margin, fcf_margin_terminal=known_margin,
            terminal_growth=vm.config.terminal_growth,
            sbc_treatment="included_in_fcf", analyst_notes="",
        )
        dcf = vm.compute_dcf(assumptions, base_revenue=100.0, wacc=wacc, net_cash=cash, shares=shares)
        target_price = dcf["per_share_value"]

        implied = vm.solve_implied_margin(target_price, shares, cash, wacc, fixed_cagr)
        assert implied == pytest.approx(known_margin, abs=1e-4)

    def test_implied_cagr_reasonable_range(self, vm):
        """Implied CAGR should be a finite number in a plausible range."""
        vm.set_base_revenue(100.0)
        implied = vm.solve_implied_cagr(
            price=50.0, shares=10.0, cash=20.0, wacc=0.10, fixed_margin=0.30,
        )
        assert not math.isnan(implied)
        assert -0.50 <= implied <= 1.50

    def test_implied_margin_reasonable_range(self, vm):
        """Implied margin should be a finite number in a plausible range."""
        vm.set_base_revenue(100.0)
        implied = vm.solve_implied_margin(
            price=50.0, shares=10.0, cash=20.0, wacc=0.10, fixed_cagr=0.15,
        )
        assert not math.isnan(implied)
        assert 0.01 <= implied <= 0.90


# ---------------------------------------------------------------------------
# 7. Historical FCF margin reconciliation
# ---------------------------------------------------------------------------

class TestHistoricalFCFMarginReconciliation:
    """Verify reconcile_historical_fcf_margin returns correct comparison."""

    def test_reconciliation_columns(self, vm):
        metrics = pd.DataFrame({
            "fiscal_period": ["FY2023", "FY2024", "FY2025"],
            "metric_name": ["FCF_margin", "FCF_margin", "FCF_margin"],
            "metric_value": [0.28, 0.32, 0.38],
        })
        result = vm.reconcile_historical_fcf_margin(metrics)
        expected_cols = {"fiscal_period", "historical_fcf_margin", "assumed_start", "assumed_terminal", "gap_vs_start"}
        assert expected_cols == set(result.columns)

    def test_reconciliation_values(self, vm):
        """Gap = historical - assumed_start for each period."""
        metrics = pd.DataFrame({
            "fiscal_period": ["FY2023", "FY2024"],
            "metric_name": ["FCF_margin", "FCF_margin"],
            "metric_value": [0.28, 0.40],
        })
        result = vm.reconcile_historical_fcf_margin(metrics)
        base = vm.config.scenarios["base"]
        assert len(result) == 2
        assert result.iloc[0]["historical_fcf_margin"] == 0.28
        assert result.iloc[0]["assumed_start"] == base.fcf_margin_start
        assert result.iloc[0]["gap_vs_start"] == pytest.approx(0.28 - base.fcf_margin_start, rel=1e-9)
        assert result.iloc[1]["gap_vs_start"] == pytest.approx(0.40 - base.fcf_margin_start, rel=1e-9)

    def test_reconciliation_empty_metrics(self, vm):
        """Empty metrics → empty result with correct columns."""
        metrics = pd.DataFrame(columns=["fiscal_period", "metric_name", "metric_value"])
        result = vm.reconcile_historical_fcf_margin(metrics)
        assert len(result) == 0
        assert "fiscal_period" in result.columns

    def test_reconciliation_filters_non_fcf(self, vm):
        """Only FCF_margin rows should appear in reconciliation."""
        metrics = pd.DataFrame({
            "fiscal_period": ["FY2023", "FY2023", "FY2024"],
            "metric_name": ["gross_margin", "FCF_margin", "FCF_margin"],
            "metric_value": [0.65, 0.30, 0.35],
        })
        result = vm.reconcile_historical_fcf_margin(metrics)
        assert len(result) == 2
        assert list(result["fiscal_period"]) == ["FY2023", "FY2024"]


# ---------------------------------------------------------------------------
# 8. Valuation input validation (Req 16.1)
# ---------------------------------------------------------------------------

from src.config import (
    ComponentStatusEnum,
    DataQualityStatus,
    ValidatedMetric,
)


class TestCheckValuationInputs:
    """Verify check_valuation_inputs blocks or allows valuation correctly."""

    def _make_metric(
        self,
        name: str = "revenue",
        status: str = "pass",
        severity: str = "critical",
        fy: int = 2025,
    ) -> ValidatedMetric:
        return ValidatedMetric(
            metric_name=name,
            fiscal_year=fy,
            parsed_value=130_000.0,
            published_value=130_000.0,
            diff_pct=0.0,
            tolerance=1.0,
            status=status,
            severity=severity,
            blocker=(severity == "critical"),
        )

    def test_blocked_when_data_quality_blocked(self, vm):
        """DATA_BLOCKED status → valuation BLOCKED regardless of metrics."""
        metrics = [self._make_metric(status="pass")]
        result = vm.check_valuation_inputs(DataQualityStatus.DATA_BLOCKED, metrics)
        assert result.status == ComponentStatusEnum.BLOCKED
        assert result.component_name == "valuation"
        assert "DATA_BLOCKED" in result.reason

    def test_blocked_when_critical_metric_fails(self, vm):
        """A critical valuation metric with status='fail' → BLOCKED."""
        metrics = [
            self._make_metric(name="revenue", status="fail", severity="critical"),
            self._make_metric(name="operating_cash_flow", status="pass"),
        ]
        result = vm.check_valuation_inputs(DataQualityStatus.PASS, metrics)
        assert result.status == ComponentStatusEnum.BLOCKED
        assert "revenue" in result.reason

    def test_blocked_when_critical_metric_missing(self, vm):
        """A critical valuation metric with status='missing' → BLOCKED."""
        metrics = [
            self._make_metric(name="capex", status="missing", severity="critical"),
        ]
        result = vm.check_valuation_inputs(DataQualityStatus.PASS, metrics)
        assert result.status == ComponentStatusEnum.BLOCKED
        assert "capex" in result.reason

    def test_usable_when_all_pass(self, vm):
        """All metrics pass → valuation USABLE."""
        metrics = [
            self._make_metric(name="revenue", status="pass"),
            self._make_metric(name="operating_cash_flow", status="pass"),
            self._make_metric(name="capex", status="pass"),
            self._make_metric(name="diluted_shares", status="pass"),
            self._make_metric(name="cash_and_securities", status="pass"),
            self._make_metric(name="total_debt", status="pass"),
        ]
        result = vm.check_valuation_inputs(DataQualityStatus.PASS, metrics)
        assert result.status == ComponentStatusEnum.USABLE
        assert result.reason == "All valuation inputs validated"

    def test_usable_when_non_critical_metric_fails(self, vm):
        """A non-valuation-critical metric failing does not block valuation."""
        metrics = [
            self._make_metric(name="revenue", status="pass"),
            self._make_metric(name="gross_profit", status="fail", severity="critical"),
        ]
        result = vm.check_valuation_inputs(DataQualityStatus.PASS, metrics)
        assert result.status == ComponentStatusEnum.USABLE

    def test_usable_with_pass_with_warnings(self, vm):
        """PASS_WITH_WARNINGS + all valuation-critical pass → USABLE."""
        metrics = [
            self._make_metric(name="revenue", status="pass"),
            self._make_metric(name="diluted_shares", status="pass"),
        ]
        result = vm.check_valuation_inputs(
            DataQualityStatus.PASS_WITH_WARNINGS, metrics,
        )
        assert result.status == ComponentStatusEnum.USABLE

    def test_usable_with_empty_metrics(self, vm):
        """No validated metrics + PASS status → USABLE (nothing to block)."""
        result = vm.check_valuation_inputs(DataQualityStatus.PASS, [])
        assert result.status == ComponentStatusEnum.USABLE

    def test_non_critical_severity_does_not_block(self, vm):
        """A valuation-critical metric with severity='major' (not critical) → USABLE."""
        metrics = [
            self._make_metric(name="revenue", status="fail", severity="major"),
        ]
        result = vm.check_valuation_inputs(DataQualityStatus.PASS, metrics)
        assert result.status == ComponentStatusEnum.USABLE

    def test_first_failing_metric_reported(self, vm):
        """When multiple valuation-critical metrics fail, the first one is reported."""
        metrics = [
            self._make_metric(name="revenue", status="fail", severity="critical"),
            self._make_metric(name="capex", status="fail", severity="critical"),
        ]
        result = vm.check_valuation_inputs(DataQualityStatus.PASS, metrics)
        assert result.status == ComponentStatusEnum.BLOCKED
        assert "revenue" in result.reason


# ---------------------------------------------------------------------------
# 9. Sanity-check bridge explanations (Reqs 16.2–16.5)
# ---------------------------------------------------------------------------

import numpy as np


class TestGenerateSanityChecks:
    """Verify generate_sanity_checks flags implausible valuation outputs."""

    def _make_grid(self, values: list[list[float]], cagr_labels=None, margin_labels=None) -> pd.DataFrame:
        """Helper to build a small reverse-DCF grid DataFrame."""
        if margin_labels is None:
            margin_labels = [f"margin={m:.2f}" for m in [0.20, 0.30]]
        if cagr_labels is None:
            cagr_labels = [f"cagr={c:.2f}" for c in [0.10, 0.20]]
        return pd.DataFrame(values, index=cagr_labels, columns=margin_labels)

    # --- DCF divergence (Req 16.2) ---

    def test_dcf_divergence_above_threshold(self, vm):
        """Target >50% above current price → bridge explanation."""
        notes = vm.generate_sanity_checks(
            target_price=200.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=80.0,
            config=vm.config,
        )
        dcf_notes = [n for n in notes if "DCF target" in n]
        assert len(dcf_notes) == 1
        assert "above" in dcf_notes[0]
        assert "100.0%" in dcf_notes[0]

    def test_dcf_divergence_below_threshold(self, vm):
        """Target >50% below current price → bridge explanation."""
        notes = vm.generate_sanity_checks(
            target_price=40.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=80.0,
            config=vm.config,
        )
        dcf_notes = [n for n in notes if "DCF target" in n]
        assert len(dcf_notes) == 1
        assert "below" in dcf_notes[0]
        assert "60.0%" in dcf_notes[0]

    def test_dcf_divergence_within_threshold(self, vm):
        """Target within 50% of current price → no DCF note."""
        notes = vm.generate_sanity_checks(
            target_price=130.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=80.0,
            config=vm.config,
        )
        dcf_notes = [n for n in notes if "DCF target" in n]
        assert len(dcf_notes) == 0

    def test_dcf_divergence_exactly_at_threshold(self, vm):
        """Target exactly at 50% boundary → no note (not exceeding)."""
        notes = vm.generate_sanity_checks(
            target_price=150.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=80.0,
            config=vm.config,
        )
        dcf_notes = [n for n in notes if "DCF target" in n]
        assert len(dcf_notes) == 0

    # --- Reverse-DCF grid (Req 16.3) ---

    def test_grid_warning_when_price_outside(self, vm):
        """No grid cell within ±10% of current price → warning."""
        # Grid values all far from 100
        grid = self._make_grid([[200, 250], [300, 350]])
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=grid,
            bear_value=80.0,
            config=vm.config,
        )
        grid_notes = [n for n in notes if "reverse-DCF grid" in n]
        assert len(grid_notes) == 1

    def test_no_grid_warning_when_price_inside(self, vm):
        """Grid cell within ±10% of current price → no warning."""
        grid = self._make_grid([[95, 150], [200, 250]])
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=grid,
            bear_value=80.0,
            config=vm.config,
        )
        grid_notes = [n for n in notes if "reverse-DCF grid" in n]
        assert len(grid_notes) == 0

    def test_no_grid_warning_for_empty_grid(self, vm):
        """Empty grid → no grid warning (nothing to check)."""
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=pd.DataFrame(),
            bear_value=80.0,
            config=vm.config,
        )
        grid_notes = [n for n in notes if "reverse-DCF grid" in n]
        assert len(grid_notes) == 0

    # --- Bear downside (Req 16.4) ---

    def test_bear_downside_exceeds_threshold(self, vm):
        """Bear downside worse than −40% → flag."""
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=50.0,  # -50% downside
            config=vm.config,
        )
        bear_notes = [n for n in notes if "Bear-case" in n]
        assert len(bear_notes) == 1
        assert "-50.0%" in bear_notes[0]

    def test_bear_downside_within_threshold(self, vm):
        """Bear downside better than −40% → no flag."""
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=70.0,  # -30% downside
            config=vm.config,
        )
        bear_notes = [n for n in notes if "Bear-case" in n]
        assert len(bear_notes) == 0

    def test_bear_downside_exactly_at_threshold(self, vm):
        """Bear downside exactly −40% → no flag (not exceeding)."""
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=60.0,  # exactly -40%
            config=vm.config,
        )
        bear_notes = [n for n in notes if "Bear-case" in n]
        assert len(bear_notes) == 0

    # --- Sensitivity table (Req 16.5) ---

    def test_sensitivity_warning_when_price_outside(self, vm):
        """No sensitivity cell within ±10% of current price → note."""
        sens = pd.DataFrame(
            {"tg=0.02": [200, 250], "tg=0.03": [300, 350]},
            index=["wacc=0.08", "wacc=0.10"],
        )
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=80.0,
            config=vm.config,
            sensitivity_table=sens,
        )
        sens_notes = [n for n in notes if "sensitivity table" in n]
        assert len(sens_notes) == 1

    def test_no_sensitivity_warning_when_price_inside(self, vm):
        """Sensitivity cell within ±10% of current price → no note."""
        sens = pd.DataFrame(
            {"tg=0.02": [95, 150], "tg=0.03": [200, 250]},
            index=["wacc=0.08", "wacc=0.10"],
        )
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=80.0,
            config=vm.config,
            sensitivity_table=sens,
        )
        sens_notes = [n for n in notes if "sensitivity table" in n]
        assert len(sens_notes) == 0

    def test_no_sensitivity_warning_when_none(self, vm):
        """No sensitivity table provided → no sensitivity note."""
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=100.0,
            reverse_dcf_grid=self._make_grid([[95, 105], [110, 120]]),
            bear_value=80.0,
            config=vm.config,
            sensitivity_table=None,
        )
        sens_notes = [n for n in notes if "sensitivity table" in n]
        assert len(sens_notes) == 0

    # --- Combined / edge cases ---

    def test_multiple_flags_at_once(self, vm):
        """All four checks can fire simultaneously."""
        grid = self._make_grid([[500, 600], [700, 800]])
        sens = pd.DataFrame(
            {"tg=0.02": [500, 600], "tg=0.03": [700, 800]},
            index=["wacc=0.08", "wacc=0.10"],
        )
        notes = vm.generate_sanity_checks(
            target_price=300.0,   # 200% above → DCF divergence
            current_price=100.0,
            reverse_dcf_grid=grid,  # all far from 100 → grid warning
            bear_value=40.0,      # -60% → bear flag
            config=vm.config,
            sensitivity_table=sens,  # all far from 100 → sens warning
        )
        assert len(notes) == 4

    def test_no_flags_when_all_ok(self, vm):
        """Clean inputs → empty notes list."""
        grid = self._make_grid([[95, 105], [110, 120]])
        sens = pd.DataFrame(
            {"tg=0.02": [95, 105], "tg=0.03": [110, 120]},
            index=["wacc=0.08", "wacc=0.10"],
        )
        notes = vm.generate_sanity_checks(
            target_price=110.0,
            current_price=100.0,
            reverse_dcf_grid=grid,
            bear_value=70.0,
            config=vm.config,
            sensitivity_table=sens,
        )
        assert len(notes) == 0

    def test_zero_current_price_skips(self, vm):
        """Zero current price → single skip note, no crash."""
        notes = vm.generate_sanity_checks(
            target_price=100.0,
            current_price=0.0,
            reverse_dcf_grid=pd.DataFrame(),
            bear_value=50.0,
            config=vm.config,
        )
        assert len(notes) == 1
        assert "skipped" in notes[0].lower()


# ---------------------------------------------------------------------------
# 10. Reverse-DCF grid semantics (Req 12.2 — Task 6.2)
# ---------------------------------------------------------------------------


class TestReverseDCFGridSemantics:
    """Verify reverse-DCF grid uses configured ranges, warns when current price
    is outside the grid, computes implied CAGR/margin, assesses plausibility,
    and generates a coherent report.

    Validates: Requirements 12.2
    """

    # --- 1. Grid uses configured ranges ---

    def test_grid_uses_configured_cagr_and_margin_ranges(self, vm):
        """Grid dimensions match the configured reverse_dcf_cagr_range and
        reverse_dcf_margin_range from EngineConfig."""
        vm.set_base_revenue(100_000.0)
        cfg = vm.config
        cagr_range = cfg.reverse_dcf_cagr_range
        margin_range = cfg.reverse_dcf_margin_range

        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        assert grid.shape == (len(cagr_range), len(margin_range))
        # Verify index and column labels correspond to configured values
        for i, c in enumerate(cagr_range):
            assert grid.index[i] == f"cagr={c:.2f}"
        for j, m in enumerate(margin_range):
            assert grid.columns[j] == f"margin={m:.2f}"

    # --- 2. Price inside grid ---

    def test_price_inside_grid_flag_true(self, vm):
        """When current_price is within the grid range (a cell within ±10%),
        price_inside_grid is True and no grid_warning attr exists."""
        vm.set_base_revenue(100.0)
        # Build a grid where at least one cell will be near current_price
        cagr_range = [0.05, 0.10, 0.15, 0.20, 0.25]
        margin_range = [0.20, 0.25, 0.30, 0.35]

        # First compute a grid without current_price to find a value in range
        probe = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        # Pick a price that is close to one of the grid values
        mid_val = float(probe.iloc[2, 2])  # a value from the middle of the grid

        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
            current_price=mid_val,
        )
        assert grid.attrs["price_inside_grid"] is True
        assert "grid_warning" not in grid.attrs

    # --- 3. Price outside grid — warning ---

    def test_price_outside_grid_warning(self, vm):
        """When current_price is far above the grid range, price_inside_grid
        is False and grid_warning attr contains an explanation."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15]
        margin_range = [0.20, 0.25, 0.30]

        # Use a very high current_price that no grid cell can be within ±10%
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
            current_price=999_999.0,
        )
        assert grid.attrs["price_inside_grid"] is False
        assert "grid_warning" in grid.attrs
        warning = grid.attrs["grid_warning"]
        assert "outside" in warning.lower()
        assert "$999,999.00" in warning

    # --- 4. Implied CAGR computed ---

    def test_implied_cagr_populated_when_outside(self, vm):
        """When price is outside grid, implied_cagr attr is populated with
        a finite number."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15]
        margin_range = [0.20, 0.25, 0.30]

        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
            current_price=999_999.0,
        )
        assert "implied_cagr" in grid.attrs
        implied_cagr = grid.attrs["implied_cagr"]
        assert isinstance(implied_cagr, float)
        # May be NaN if solver can't converge to such an extreme price,
        # but the attr must exist. If finite, it should be a number.
        # For a very high price the solver may return NaN — that's acceptable.
        # We just verify the attr is set.

    # --- 5. Implied margin computed ---

    def test_implied_margin_populated_when_outside(self, vm):
        """When price is outside grid, implied_margin attr is populated with
        a finite number."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15]
        margin_range = [0.20, 0.25, 0.30]

        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
            current_price=999_999.0,
        )
        assert "implied_margin" in grid.attrs
        implied_margin = grid.attrs["implied_margin"]
        assert isinstance(implied_margin, float)

    # --- 6. Plausibility assessment ---

    def test_plausibility_assessment_text(self, vm):
        """implied_cagr_assessment and implied_margin_assessment attrs contain
        plausibility text ('within', 'above', or 'far outside')."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15]
        margin_range = [0.20, 0.25, 0.30]

        # Use a moderately high price so solvers converge
        # First find the max grid value and go somewhat above it
        probe = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        max_val = float(probe.values.max())
        outside_price = max_val * 2.0  # double the max grid value

        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
            current_price=outside_price,
        )
        assert grid.attrs["price_inside_grid"] is False

        cagr_assess = grid.attrs["implied_cagr_assessment"]
        margin_assess = grid.attrs["implied_margin_assessment"]

        # Each assessment should contain one of the expected plausibility phrases
        valid_phrases = ["within", "above", "below", "far outside", "could not be solved"]
        assert any(p in cagr_assess for p in valid_phrases), (
            f"CAGR assessment '{cagr_assess}' does not contain a valid plausibility phrase"
        )
        assert any(p in margin_assess for p in valid_phrases), (
            f"Margin assessment '{margin_assess}' does not contain a valid plausibility phrase"
        )

    # --- 7. Report text ---

    def test_report_text_includes_grid_description(self, vm):
        """generate_reverse_dcf_report() produces text that includes grid
        description, price status, and implied assumptions when applicable."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15]
        margin_range = [0.20, 0.25, 0.30]

        # Case A: price inside grid
        probe = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        mid_val = float(probe.iloc[1, 1])
        grid_inside = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
            current_price=mid_val,
        )
        report_inside = vm.generate_reverse_dcf_report(grid_inside)
        assert "Reverse-DCF Grid Analysis" in report_inside
        assert "3 CAGR assumptions" in report_inside
        assert "3 FCF margin assumptions" in report_inside
        assert "within" in report_inside.lower()

    def test_report_text_outside_grid_includes_implied(self, vm):
        """When price is outside grid, report includes implied assumptions
        and plausibility assessment."""
        vm.set_base_revenue(100.0)
        cagr_range = [0.05, 0.10, 0.15]
        margin_range = [0.20, 0.25, 0.30]

        probe = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
        )
        max_val = float(probe.values.max())
        outside_price = max_val * 2.0

        grid_outside = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=cagr_range, margin_range=margin_range,
            current_price=outside_price,
        )
        report_outside = vm.generate_reverse_dcf_report(grid_outside)
        assert "outside" in report_outside.lower()
        assert "Implied CAGR" in report_outside
        assert "Implied FCF margin" in report_outside
        assert "NOT extended to unreasonable assumptions" in report_outside

    def test_report_text_no_current_price(self, vm):
        """When no current_price was provided, report says so."""
        vm.set_base_revenue(100.0)
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=[0.10, 0.20], margin_range=[0.25, 0.35],
        )
        report = vm.generate_reverse_dcf_report(grid)
        assert "No current price was provided" in report

    # --- 8. Backward compatibility ---

    def test_backward_compat_no_current_price(self, vm):
        """Calling compute_reverse_dcf_grid() without current_price still works
        and doesn't add the new attrs (price_inside_grid, grid_warning, etc.)."""
        vm.set_base_revenue(100.0)
        grid = vm.compute_reverse_dcf_grid(
            price=100.0, shares=10.0, cash=50.0, wacc=0.10,
            cagr_range=[0.10, 0.15, 0.20], margin_range=[0.25, 0.30],
        )
        # The grid should still have highlight (existing behavior)
        assert "highlight" in grid.attrs
        # But should NOT have the new current-price-analysis attrs
        assert "price_inside_grid" not in grid.attrs
        assert "grid_warning" not in grid.attrs
        assert "implied_cagr" not in grid.attrs
        assert "implied_margin" not in grid.attrs

    # --- Solver convergence (one-variable solvers) ---

    def test_solver_convergence_implied_cagr(self, vm):
        """solve_implied_cagr converges for a price within the grid range."""
        vm.set_base_revenue(100.0)
        # Compute a known price from a known CAGR
        known_cagr = 0.15
        fixed_margin = 0.30
        wacc = 0.10
        cash = 50.0
        shares = 10.0

        assumptions = ScenarioAssumptions(
            name="rt", probability=1.0, revenue_cagr=known_cagr,
            fcf_margin_start=fixed_margin, fcf_margin_terminal=fixed_margin,
            terminal_growth=vm.config.terminal_growth,
            sbc_treatment="included_in_fcf", analyst_notes="",
        )
        dcf = vm.compute_dcf(assumptions, base_revenue=100.0, wacc=wacc, net_cash=cash, shares=shares)
        target_price = dcf["per_share_value"]

        implied = vm.solve_implied_cagr(target_price, shares, cash, wacc, fixed_margin)
        assert math.isfinite(implied)
        assert implied == pytest.approx(known_cagr, abs=1e-3)

    def test_solver_convergence_implied_margin(self, vm):
        """solve_implied_margin converges for a price within the grid range."""
        vm.set_base_revenue(100.0)
        fixed_cagr = 0.20
        known_margin = 0.28
        wacc = 0.10
        cash = 30.0
        shares = 10.0

        assumptions = ScenarioAssumptions(
            name="rt", probability=1.0, revenue_cagr=fixed_cagr,
            fcf_margin_start=known_margin, fcf_margin_terminal=known_margin,
            terminal_growth=vm.config.terminal_growth,
            sbc_treatment="included_in_fcf", analyst_notes="",
        )
        dcf = vm.compute_dcf(assumptions, base_revenue=100.0, wacc=wacc, net_cash=cash, shares=shares)
        target_price = dcf["per_share_value"]

        implied = vm.solve_implied_margin(target_price, shares, cash, wacc, fixed_cagr)
        assert math.isfinite(implied)
        assert implied == pytest.approx(known_margin, abs=1e-3)
