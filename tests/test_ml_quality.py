"""
Tests for ML quality assessment (Tasks 8.1–8.3).

Covers:
- assess_ml_quality() degenerate model detection
- assess_ml_quality() baseline underperformance detection
- assess_ml_quality() usable model detection
- get_coefficient_summary()
- generate_model_audit() annual-only caveat
- generate_model_audit() training summary and coefficient values
- generate_model_audit() walk-forward MAE vs baselines
- generate_model_audit() honest predictive value assessment

Reqs: 18.1, 18.2, 18.3, 18.4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ComponentStatus, ComponentStatusEnum, EngineConfig
from src.ml_models import MLDriverModel


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig()


@pytest.fixture
def ml(config) -> MLDriverModel:
    return MLDriverModel(config=config)


# ------------------------------------------------------------------
# Tests: get_coefficient_summary
# ------------------------------------------------------------------

class TestGetCoefficientSummary:
    def test_returns_dict_for_trained_model(self, ml):
        X = pd.DataFrame({"feat_a": [1, 2, 3, 4], "feat_b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = ml.train_primary_model(X, y)
        summary = ml.get_coefficient_summary(fitted)
        assert isinstance(summary, dict)
        assert len(summary) == 2

    def test_returns_empty_for_none_model(self, ml):
        assert ml.get_coefficient_summary(None) == {}

    def test_returns_empty_for_model_without_coef(self, ml):
        class FakeModel:
            pass
        assert ml.get_coefficient_summary(FakeModel()) == {}


# ------------------------------------------------------------------
# Tests: assess_ml_quality — degenerate model (Req 18.1)
# ------------------------------------------------------------------

class TestDegenerateModelDetection:
    def test_all_coefs_near_zero_returns_diagnostic_only(self, ml):
        """When max(|coef|) < 0.001, status should be diagnostic_only
        with reason 'degenerate model'."""
        model_results = {
            "coefficients": {"feat_a": 0.0001, "feat_b": -0.0005, "feat_c": 0.0},
            "walk_forward": {"predictions": [1.0, 2.0], "actuals": [1.1, 2.1]},
            "training_obs": 5,
        }
        baseline_results = {
            "last_period": {"mae": 0.5},
        }
        status = ml.assess_ml_quality(model_results, baseline_results)
        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY
        assert status.reason == "degenerate model"
        assert status.component_name == "ml_signal"

    def test_exactly_at_threshold_is_degenerate(self, ml):
        """max(|coef|) = 0.0009 < 0.001 → degenerate."""
        model_results = {
            "coefficients": {"feat_a": 0.0009},
            "walk_forward": {"predictions": [1.0], "actuals": [1.0]},
            "training_obs": 3,
        }
        status = ml.assess_ml_quality(model_results, {})
        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY
        assert status.reason == "degenerate model"

    def test_above_threshold_not_degenerate(self, ml):
        """max(|coef|) = 0.002 >= 0.001 → not degenerate (may still
        fail baseline check)."""
        model_results = {
            "coefficients": {"feat_a": 0.002},
            "walk_forward": {"predictions": [1.0, 2.0], "actuals": [1.0, 2.0]},
            "training_obs": 5,
        }
        # Model MAE = 0, baseline MAE = 0.5 → model beats baseline
        baseline_results = {"last_period": {"mae": 0.5}}
        status = ml.assess_ml_quality(model_results, baseline_results)
        assert status.status == ComponentStatusEnum.USABLE

    def test_empty_coefficients_returns_diagnostic_only(self, ml):
        """No coefficients at all → diagnostic_only."""
        model_results = {
            "coefficients": {},
            "walk_forward": {"predictions": [], "actuals": []},
            "training_obs": 0,
        }
        status = ml.assess_ml_quality(model_results, {})
        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY


# ------------------------------------------------------------------
# Tests: assess_ml_quality — baseline underperformance (Req 18.2)
# ------------------------------------------------------------------

class TestBaselineUnderperformanceDetection:
    def test_model_worse_than_all_baselines_returns_diagnostic_only(self, ml):
        """When model MAE > every baseline MAE → diagnostic_only
        with reason 'underperforms baselines'."""
        model_results = {
            "coefficients": {"feat_a": 1.5, "feat_b": -0.8},
            "walk_forward": {
                "predictions": [1.0, 2.0, 3.0],
                "actuals": [2.0, 4.0, 6.0],  # MAE = 2.0
            },
            "training_obs": 6,
        }
        baseline_results = {
            "last_period": {"mae": 1.0},
            "trailing_4q_avg": {"mae": 1.5},
            "three_year_avg": {"mae": 1.8},
            "linear_trend": {"mae": 0.9},
        }
        status = ml.assess_ml_quality(model_results, baseline_results)
        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY
        assert status.reason == "underperforms baselines"

    def test_model_beats_one_baseline_is_usable(self, ml):
        """Model MAE < at least one baseline → usable."""
        model_results = {
            "coefficients": {"feat_a": 1.5},
            "walk_forward": {
                "predictions": [1.0, 2.0, 3.0],
                "actuals": [1.1, 2.1, 3.1],  # MAE = 0.1
            },
            "training_obs": 6,
        }
        baseline_results = {
            "last_period": {"mae": 0.05},       # better than model
            "trailing_4q_avg": {"mae": 0.5},     # worse than model
        }
        status = ml.assess_ml_quality(model_results, baseline_results)
        assert status.status == ComponentStatusEnum.USABLE

    def test_no_walk_forward_data_still_usable_if_coefs_ok(self, ml):
        """If walk-forward wasn't run but coefficients are non-degenerate,
        model is still usable (can't fail baseline check without data)."""
        model_results = {
            "coefficients": {"feat_a": 2.0},
            "walk_forward": {"predictions": [], "actuals": []},
            "training_obs": 3,
        }
        baseline_results = {"last_period": {"mae": 0.5}}
        status = ml.assess_ml_quality(model_results, baseline_results)
        assert status.status == ComponentStatusEnum.USABLE

    def test_baselines_with_nan_mae_are_skipped(self, ml):
        """Baselines with NaN MAE should be ignored in the comparison."""
        model_results = {
            "coefficients": {"feat_a": 1.5},
            "walk_forward": {
                "predictions": [1.0, 2.0],
                "actuals": [2.0, 4.0],  # MAE = 1.5
            },
            "training_obs": 5,
        }
        baseline_results = {
            "last_period": {"mae": float("nan")},
            "trailing_4q_avg": {"mae": 1.0},  # model worse
        }
        status = ml.assess_ml_quality(model_results, baseline_results)
        assert status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY
        assert status.reason == "underperforms baselines"


# ------------------------------------------------------------------
# Tests: assess_ml_quality — ComponentStatus structure
# ------------------------------------------------------------------

class TestAssessMlQualityStructure:
    def test_returns_component_status_type(self, ml):
        model_results = {
            "coefficients": {"feat_a": 1.0},
            "walk_forward": {"predictions": [1.0], "actuals": [1.0]},
            "training_obs": 5,
        }
        status = ml.assess_ml_quality(model_results, {})
        assert isinstance(status, ComponentStatus)
        assert status.component_name == "ml_signal"

    def test_details_include_training_observations(self, ml):
        model_results = {
            "coefficients": {"feat_a": 0.0001},
            "walk_forward": {"predictions": [], "actuals": []},
            "training_obs": 7,
        }
        status = ml.assess_ml_quality(model_results, {})
        assert status.details["training_observations"] == 7


# ------------------------------------------------------------------
# Tests: generate_model_audit — annual-only caveat (Req 18.3)
# ------------------------------------------------------------------

class TestAnnualOnlyCaveat:
    def test_caveat_present_when_few_observations(self, ml, tmp_path):
        """When training observations < 12, audit should include
        the annual-only caveat."""
        ml.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y = pd.Series([10, 20, 30, 40, 50])
        fitted, _ = ml.train_primary_model(X, y)
        baselines = ml.compute_baselines(y)

        audit = ml.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert "annual-only caveat" in audit.lower()
        assert "exploratory and underpowered" in audit.lower()

    def test_caveat_absent_when_enough_observations(self, ml, tmp_path):
        """When training observations >= 12, no annual-only caveat."""
        ml.config.outputs_dir = tmp_path
        n = 15
        X = pd.DataFrame({"a": range(n), "b": range(n, 0, -1)})
        y = pd.Series(np.random.randn(n))
        fitted, _ = ml.train_primary_model(X, y)
        baselines = ml.compute_baselines(y)

        audit = ml.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert "annual-only caveat" not in audit.lower()


# ------------------------------------------------------------------
# Tests: generate_model_audit — training summary (Req 18.4)
# ------------------------------------------------------------------

class TestAuditTrainingSummary:
    def test_audit_includes_training_obs_count(self, ml, tmp_path):
        ml.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = ml.train_primary_model(X, y)
        baselines = ml.compute_baselines(y)

        audit = ml.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert "Training observations" in audit
        assert "4" in audit  # 4 observations

    def test_audit_includes_feature_count(self, ml, tmp_path):
        ml.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [0, 1, 0]})
        y = pd.Series([10, 20, 30])
        fitted, _ = ml.train_primary_model(X, y)
        baselines = ml.compute_baselines(y)

        audit = ml.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert "Feature count" in audit
        assert "3" in audit  # 3 features

    def test_audit_includes_coefficient_values(self, ml, tmp_path):
        ml.config.outputs_dir = tmp_path
        X = pd.DataFrame({"feat_x": [1, 2, 3, 4], "feat_y": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = ml.train_primary_model(X, y)
        baselines = ml.compute_baselines(y)

        audit = ml.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert "feat_x" in audit
        assert "feat_y" in audit
        assert "Coefficient" in audit

    def test_audit_includes_walk_forward_vs_baselines(self, ml, tmp_path):
        """Audit should show walk-forward MAE alongside baseline MAEs."""
        ml.config.outputs_dir = tmp_path
        ml.config.min_train_years = 3
        X = pd.DataFrame({"a": range(6), "b": range(6, 0, -1)})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fitted, _ = ml.train_primary_model(X, y)
        baselines = ml.compute_baselines(y)
        wf = ml.walk_forward_validate(X, y)

        audit = ml.generate_model_audit(
            model=fitted, baselines=baselines, features=X,
            walk_forward_results=wf, target=y,
        )
        assert "Baseline Comparison" in audit
        assert "last_period" in audit
        assert "ML model (walk-forward)" in audit

    def test_audit_includes_predictive_value_assessment(self, ml, tmp_path):
        ml.config.outputs_dir = tmp_path
        ml.config.min_train_years = 3
        X = pd.DataFrame({"a": range(6)})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fitted, _ = ml.train_primary_model(X, y)
        baselines = ml.compute_baselines(y)
        wf = ml.walk_forward_validate(X, y)

        audit = ml.generate_model_audit(
            model=fitted, baselines=baselines, features=X,
            walk_forward_results=wf, target=y,
        )
        assert "Predictive Value Assessment" in audit
