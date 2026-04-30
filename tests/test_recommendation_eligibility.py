"""
Tests for recommendation eligibility gate (Req 17).

Validates:
- formal_rating when all blocking gates pass (even if ML is diagnostic_only)
- diagnostic_not_rated when DATA_BLOCKED
- ML diagnostic_only does NOT block formal rating
- NLP diagnostic_only does NOT block formal rating
- Scorecard never contains None status
- Rating thresholds only evaluated after eligibility gate passes
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.audit_utils import AuditModule
from src.config import (
    ComponentStatus,
    ComponentStatusEnum,
    DataQualityStatus,
    EngineConfig,
    RecommendationStatus,
    ReportMode,
    get_default_config,
)
from src.valuation import ValuationModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_component(
    name: str,
    status: ComponentStatusEnum = ComponentStatusEnum.USABLE,
    reason: str = "OK",
) -> ComponentStatus:
    return ComponentStatus(component_name=name, status=status, reason=reason)


def _all_blocking_usable() -> list[ComponentStatus]:
    """Return a list of ComponentStatus where every blocking gate is usable."""
    return [
        _make_component("data_validation"),
        _make_component("market_price"),
        _make_component("dcf_assumptions"),
        _make_component("split_basis"),
        _make_component("lookahead"),
        _make_component("audit_consistency"),
    ]


def _valuation_usable() -> ComponentStatus:
    return _make_component("valuation")


def _build_scenarios(
    bear_val: float = 80.0,
    base_val: float = 120.0,
    bull_val: float = 180.0,
) -> dict:
    """Build a minimal scenarios dict matching ValuationModule.build_scenarios output."""
    return {
        "bear": {"per_share_value": bear_val, "probability": 0.25},
        "base": {"per_share_value": base_val, "probability": 0.50},
        "bull": {"per_share_value": bull_val, "probability": 0.25},
    }


def _empty_reverse_dcf() -> pd.DataFrame:
    """Return a minimal reverse-DCF grid DataFrame."""
    df = pd.DataFrame({"margin=0.30": [100.0, 120.0]}, index=["cagr=0.10", "cagr=0.20"])
    hl = pd.DataFrame({"margin=0.30": [False, True]}, index=["cagr=0.10", "cagr=0.20"])
    df.attrs["highlight"] = hl
    return df


# ---------------------------------------------------------------------------
# Task 7.1 / 7.2: run_pre_report_audit tests
# ---------------------------------------------------------------------------

class TestRunPreReportAudit:
    """Tests for AuditModule.run_pre_report_audit()."""

    def test_formal_rating_all_blocking_pass(self):
        """formal_rating when all blocking gates pass, even if ML is diagnostic_only."""
        audit = AuditModule()
        components = _all_blocking_usable() + [
            _make_component("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY, "degenerate model"),
            _make_component("nlp_signal", ComponentStatusEnum.USABLE),
            _make_component("peer_multiples", ComponentStatusEnum.USABLE),
            _make_component("segment_chart", ComponentStatusEnum.USABLE),
        ]
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.PASS,
            component_statuses=components,
            valuation_input_status=_valuation_usable(),
        )
        assert result.eligibility_status == ReportMode.FORMAL_RATING
        assert result.blocking_issues == []

    def test_diagnostic_not_rated_when_data_blocked(self):
        """diagnostic_not_rated when DataQualityStatus is DATA_BLOCKED."""
        audit = AuditModule()
        components = _all_blocking_usable()
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            component_statuses=components,
            valuation_input_status=_valuation_usable(),
        )
        assert result.eligibility_status == ReportMode.DIAGNOSTIC_NOT_RATED
        assert any("DATA_BLOCKED" in issue for issue in result.blocking_issues)

    def test_diagnostic_not_rated_when_valuation_blocked(self):
        """diagnostic_not_rated when valuation inputs are blocked."""
        audit = AuditModule()
        components = _all_blocking_usable()
        val_blocked = _make_component(
            "valuation", ComponentStatusEnum.BLOCKED, "Valuation-critical input failed",
        )
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.PASS,
            component_statuses=components,
            valuation_input_status=val_blocked,
        )
        assert result.eligibility_status == ReportMode.DIAGNOSTIC_NOT_RATED
        assert any("valuation" in issue for issue in result.blocking_issues)

    def test_ml_diagnostic_only_does_not_block(self):
        """ML diagnostic_only does NOT block formal rating."""
        audit = AuditModule()
        components = _all_blocking_usable() + [
            _make_component("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY, "underperforms baselines"),
        ]
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.PASS,
            component_statuses=components,
            valuation_input_status=_valuation_usable(),
        )
        assert result.eligibility_status == ReportMode.FORMAL_RATING
        assert result.blocking_issues == []

    def test_nlp_diagnostic_only_does_not_block(self):
        """NLP diagnostic_only does NOT block formal rating."""
        audit = AuditModule()
        components = _all_blocking_usable() + [
            _make_component("nlp_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY, "poor extraction"),
        ]
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.PASS,
            component_statuses=components,
            valuation_input_status=_valuation_usable(),
        )
        assert result.eligibility_status == ReportMode.FORMAL_RATING
        assert result.blocking_issues == []

    def test_pass_with_warnings_allows_formal_rating(self):
        """PASS_WITH_WARNINGS still allows formal_rating."""
        audit = AuditModule()
        components = _all_blocking_usable()
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.PASS_WITH_WARNINGS,
            component_statuses=components,
            valuation_input_status=_valuation_usable(),
        )
        assert result.eligibility_status == ReportMode.FORMAL_RATING

    def test_recommendation_status_has_timestamp(self):
        """RecommendationStatus includes a timestamp."""
        audit = AuditModule()
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.PASS,
            component_statuses=_all_blocking_usable(),
            valuation_input_status=_valuation_usable(),
        )
        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_component_statuses_preserved(self):
        """All component statuses are preserved in the output."""
        audit = AuditModule()
        components = _all_blocking_usable() + [
            _make_component("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY, "degenerate"),
        ]
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.PASS,
            component_statuses=components,
            valuation_input_status=_valuation_usable(),
        )
        names = {cs.component_name for cs in result.component_statuses}
        assert "ml_signal" in names
        assert "valuation" in names

    def test_multiple_blocking_failures(self):
        """Multiple blocking failures are all reported."""
        audit = AuditModule()
        components = [
            _make_component("data_validation", ComponentStatusEnum.BLOCKED, "data failed"),
            _make_component("market_price", ComponentStatusEnum.UNAVAILABLE, "no price"),
            _make_component("dcf_assumptions"),
            _make_component("split_basis"),
            _make_component("lookahead"),
            _make_component("audit_consistency"),
        ]
        result = audit.run_pre_report_audit(
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            component_statuses=components,
            valuation_input_status=_valuation_usable(),
        )
        assert result.eligibility_status == ReportMode.DIAGNOSTIC_NOT_RATED
        # Should have at least 3 blocking issues: DATA_BLOCKED + data_validation + market_price
        assert len(result.blocking_issues) >= 3


# ---------------------------------------------------------------------------
# Task 7.3: generate_recommendation() respects eligibility
# ---------------------------------------------------------------------------

class TestGenerateRecommendationEligibility:
    """Tests for ValuationModule.generate_recommendation() with eligibility gate."""

    def _make_formal_status(
        self,
        ml_status: ComponentStatusEnum = ComponentStatusEnum.USABLE,
        nlp_status: ComponentStatusEnum = ComponentStatusEnum.USABLE,
    ) -> RecommendationStatus:
        """Build a formal_rating RecommendationStatus."""
        return RecommendationStatus(
            eligibility_status=ReportMode.FORMAL_RATING,
            data_quality_status=DataQualityStatus.PASS,
            component_statuses=[
                _make_component("data_validation"),
                _make_component("valuation"),
                _make_component("ml_signal", ml_status, "ML status"),
                _make_component("nlp_signal", nlp_status, "NLP status"),
            ],
            blocking_issues=[],
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def _make_blocked_status(self) -> RecommendationStatus:
        """Build a diagnostic_not_rated RecommendationStatus."""
        return RecommendationStatus(
            eligibility_status=ReportMode.DIAGNOSTIC_NOT_RATED,
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            component_statuses=[
                _make_component("data_validation", ComponentStatusEnum.BLOCKED, "DATA_BLOCKED"),
                _make_component("valuation", ComponentStatusEnum.BLOCKED, "blocked"),
            ],
            blocking_issues=["Data validation gate: DataQualityStatus is DATA_BLOCKED"],
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def test_formal_rating_produces_buy_hold_sell(self):
        """When eligibility is formal_rating, rating is Buy/Hold/Sell."""
        vm = ValuationModule()
        scenarios = _build_scenarios(bear_val=80, base_val=120, bull_val=180)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            recommendation_status=self._make_formal_status(),
        )
        assert rec.rating in ("Buy", "Hold", "Sell")
        assert rec.eligibility_status == "formal_rating"
        assert rec.not_rated_reason is None

    def test_diagnostic_not_rated_forces_not_rated(self):
        """When eligibility is diagnostic_not_rated, rating = 'Not Rated'."""
        vm = ValuationModule()
        # Even with strong upside, should be Not Rated
        scenarios = _build_scenarios(bear_val=200, base_val=300, bull_val=400)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            recommendation_status=self._make_blocked_status(),
        )
        assert rec.rating == "Not Rated"
        assert rec.eligibility_status == "diagnostic_not_rated"
        assert rec.not_rated_reason is not None
        assert "DATA_BLOCKED" in rec.not_rated_reason

    def test_thresholds_not_evaluated_when_blocked(self):
        """Rating thresholds (Req 9.11) are NOT evaluated when eligibility fails.

        Even with massive upside that would normally trigger Buy, the
        rating must be 'Not Rated' when the eligibility gate fails.
        """
        vm = ValuationModule()
        # Extreme upside scenario that would be Buy if thresholds were evaluated
        scenarios = _build_scenarios(bear_val=500, base_val=1000, bull_val=2000)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            recommendation_status=self._make_blocked_status(),
        )
        assert rec.rating == "Not Rated"

    def test_scorecard_never_contains_none_status(self):
        """Every scorecard component has a status field, never None (Req 17.4)."""
        vm = ValuationModule()
        scenarios = _build_scenarios()
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            ml_signal=None,
            narrative_signal=None,
            recommendation_status=self._make_formal_status(),
        )
        for component_name, component_data in rec.scorecard.items():
            assert isinstance(component_data, dict), (
                f"Scorecard component '{component_name}' should be a dict"
            )
            assert "status" in component_data, (
                f"Scorecard component '{component_name}' missing 'status' key"
            )
            assert component_data["status"] is not None, (
                f"Scorecard component '{component_name}' has None status"
            )

    def test_ml_diagnostic_only_reflected_in_scorecard(self):
        """ML diagnostic_only status is reflected in the scorecard."""
        vm = ValuationModule()
        scenarios = _build_scenarios()
        status = self._make_formal_status(ml_status=ComponentStatusEnum.DIAGNOSTIC_ONLY)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            ml_signal=0.5,
            recommendation_status=status,
        )
        assert rec.scorecard["ml_signal"]["status"] == "diagnostic_only"
        # But rating should still be formal (Buy/Hold/Sell)
        assert rec.rating in ("Buy", "Hold", "Sell")

    def test_nlp_diagnostic_only_reflected_in_scorecard(self):
        """NLP diagnostic_only status is reflected in the scorecard."""
        vm = ValuationModule()
        scenarios = _build_scenarios()
        status = self._make_formal_status(nlp_status=ComponentStatusEnum.DIAGNOSTIC_ONLY)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            narrative_signal=0.3,
            recommendation_status=status,
        )
        assert rec.scorecard["narrative_signal"]["status"] == "diagnostic_only"
        assert rec.rating in ("Buy", "Hold", "Sell")

    def test_backward_compatible_without_recommendation_status(self):
        """generate_recommendation() still works without recommendation_status."""
        vm = ValuationModule()
        scenarios = _build_scenarios(bear_val=80, base_val=120, bull_val=180)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
        )
        # Should produce a normal rating
        assert rec.rating in ("Buy", "Hold", "Sell")
        assert rec.eligibility_status == "formal_rating"

    def test_buy_rating_with_strong_upside(self):
        """Buy rating when upside exceeds thresholds and eligibility passes."""
        vm = ValuationModule()
        # Expected value = 0.25*150 + 0.50*200 + 0.25*300 = 37.5 + 100 + 75 = 212.5
        # Upside = (212.5 - 100) / 100 = 112.5% > 15%
        # Base upside = (200 - 100) / 100 = 100% > 15%
        # Bear downside = (150 - 100) / 100 = 50% > -25%
        scenarios = _build_scenarios(bear_val=150, base_val=200, bull_val=300)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            recommendation_status=self._make_formal_status(),
        )
        assert rec.rating == "Buy"

    def test_sell_rating_with_strong_downside(self):
        """Sell rating when downside exceeds threshold and eligibility passes."""
        vm = ValuationModule()
        # Expected value = 0.25*30 + 0.50*50 + 0.25*70 = 7.5 + 25 + 17.5 = 50
        # Upside = (50 - 100) / 100 = -50% < -10%
        scenarios = _build_scenarios(bear_val=30, base_val=50, bull_val=70)
        rec = vm.generate_recommendation(
            scenarios=scenarios,
            current_price=100.0,
            reverse_dcf=_empty_reverse_dcf(),
            recommendation_status=self._make_formal_status(),
        )
        assert rec.rating == "Sell"
