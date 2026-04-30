"""
Tests for tree-based models (RF / GBT) with SHAP feature importance.

Covers: train_tree_model, compute_feature_importances, compute_shap_values,
walk_forward_validate_tree, generate_tree_model_audit, and integration with
the overall model audit.

Task 17.4 — Optional enhancement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig
from src.ml_models import MLDriverModel


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig()


@pytest.fixture
def model(config) -> MLDriverModel:
    return MLDriverModel(config=config)


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Synthetic feature matrix with 10 observations and 3 features."""
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        "revenue_growth": np.random.uniform(-0.1, 0.5, n),
        "margin": np.random.uniform(0.2, 0.6, n),
        "sentiment": np.random.uniform(-1, 1, n),
    })


@pytest.fixture
def synthetic_target(synthetic_features) -> pd.Series:
    """Target correlated with features for meaningful model training."""
    np.random.seed(42)
    X = synthetic_features
    # Target is a noisy linear combination of features
    target = 0.5 * X["revenue_growth"] + 0.3 * X["margin"] + 0.1 * np.random.randn(len(X))
    return target


@pytest.fixture
def large_synthetic_data():
    """Larger dataset (20 rows) for walk-forward validation tests."""
    np.random.seed(123)
    n = 20
    X = pd.DataFrame({
        "feat_a": np.random.uniform(0, 1, n),
        "feat_b": np.random.uniform(0, 1, n),
        "feat_c": np.random.uniform(0, 1, n),
    })
    y = 2.0 * X["feat_a"] - 1.0 * X["feat_b"] + 0.5 * np.random.randn(n)
    return X, y


# ------------------------------------------------------------------
# Tests: train_tree_model — Random Forest
# ------------------------------------------------------------------

class TestTrainRandomForest:
    def test_rf_returns_fitted_model(self, model, synthetic_features, synthetic_target):
        fitted, importances = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        assert fitted is not None
        assert hasattr(fitted, "predict")
        assert hasattr(fitted, "feature_importances_")

    def test_rf_importances_match_features(self, model, synthetic_features, synthetic_target):
        fitted, importances = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        assert set(importances.keys()) == set(synthetic_features.columns)
        assert all(v >= 0 for v in importances.values())

    def test_rf_importances_sum_to_one(self, model, synthetic_features, synthetic_target):
        fitted, importances = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)

    def test_rf_can_predict(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        preds = fitted.predict(synthetic_features.fillna(0.0))
        assert len(preds) == len(synthetic_features)
        assert not np.any(np.isnan(preds))


# ------------------------------------------------------------------
# Tests: train_tree_model — Gradient Boosted Trees
# ------------------------------------------------------------------

class TestTrainGradientBoosting:
    def test_gbt_returns_fitted_model(self, model, synthetic_features, synthetic_target):
        fitted, importances = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        assert fitted is not None
        assert hasattr(fitted, "predict")
        assert hasattr(fitted, "feature_importances_")

    def test_gbt_importances_match_features(self, model, synthetic_features, synthetic_target):
        fitted, importances = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        assert set(importances.keys()) == set(synthetic_features.columns)

    def test_gbt_importances_sum_to_one(self, model, synthetic_features, synthetic_target):
        fitted, importances = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)

    def test_gbt_deterministic_with_seed(self, model, synthetic_features, synthetic_target):
        """GBT with same random_state should produce identical results."""
        fitted1, imp1 = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        fitted2, imp2 = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        preds1 = fitted1.predict(synthetic_features.fillna(0.0))
        preds2 = fitted2.predict(synthetic_features.fillna(0.0))
        np.testing.assert_array_almost_equal(preds1, preds2)


# ------------------------------------------------------------------
# Tests: compute_feature_importances
# ------------------------------------------------------------------

class TestComputeFeatureImportancesTree:
    def test_returns_sorted_dataframe(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        imp_df = model.compute_feature_importances(fitted, synthetic_features)
        assert list(imp_df.columns) == ["feature", "importance"]
        assert len(imp_df) == 3
        # Should be sorted descending
        assert imp_df["importance"].is_monotonic_decreasing

    def test_none_model_returns_empty_df(self, model, synthetic_features):
        imp_df = model.compute_feature_importances(None, synthetic_features)
        assert imp_df.empty
        assert list(imp_df.columns) == ["feature", "importance"]

    def test_model_without_importances_returns_empty(self, model, synthetic_features):
        """A model without feature_importances_ should return empty."""

        class FakeModel:
            pass

        imp_df = model.compute_feature_importances(FakeModel(), synthetic_features)
        assert imp_df.empty


# ------------------------------------------------------------------
# Tests: compute_shap_values
# ------------------------------------------------------------------

class TestComputeShapValues:
    def test_shap_returns_values_and_importance(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        shap_df, importance_df = model.compute_shap_values(fitted, synthetic_features)
        assert shap_df is not None
        assert importance_df is not None
        # SHAP values should have same shape as features
        assert shap_df.shape[1] == synthetic_features.shape[1]
        # Importance should have feature and mean_abs_shap columns
        assert list(importance_df.columns) == ["feature", "mean_abs_shap"]
        assert len(importance_df) == 3

    def test_shap_importance_sorted_descending(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        _, importance_df = model.compute_shap_values(fitted, synthetic_features)
        assert importance_df is not None
        assert importance_df["mean_abs_shap"].is_monotonic_decreasing

    def test_shap_values_nonnegative_importance(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        _, importance_df = model.compute_shap_values(fitted, synthetic_features)
        assert importance_df is not None
        assert (importance_df["mean_abs_shap"] >= 0).all()

    def test_shap_none_model_returns_none(self, model, synthetic_features):
        shap_df, importance_df = model.compute_shap_values(None, synthetic_features)
        assert shap_df is None
        assert importance_df is None

    def test_shap_with_nan_features(self, model, synthetic_features, synthetic_target):
        """SHAP should handle features with some NaN values gracefully."""
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        # Introduce NaN into features
        features_with_nan = synthetic_features.copy()
        features_with_nan.iloc[0, 0] = np.nan
        shap_df, importance_df = model.compute_shap_values(fitted, features_with_nan)
        # Should still work (NaN rows dropped internally)
        assert shap_df is not None
        assert importance_df is not None

    def test_shap_top_features_present(self, model, synthetic_features, synthetic_target):
        """SHAP importance summary should include all feature names."""
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        _, importance_df = model.compute_shap_values(fitted, synthetic_features)
        assert importance_df is not None
        shap_features = set(importance_df["feature"].tolist())
        assert shap_features == set(synthetic_features.columns)

    def test_shap_with_gbt_model(self, model, synthetic_features, synthetic_target):
        """SHAP should work with GradientBoostingRegressor."""
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        shap_df, importance_df = model.compute_shap_values(fitted, synthetic_features)
        assert shap_df is not None
        assert importance_df is not None
        assert len(importance_df) == 3


# ------------------------------------------------------------------
# Tests: walk_forward_validate_tree
# ------------------------------------------------------------------

class TestWalkForwardValidateTree:
    def test_returns_predictions_and_actuals(self, model, large_synthetic_data):
        X, y = large_synthetic_data
        model.config.min_train_years = 3
        result = model.walk_forward_validate_tree(X, y, model_type="random_forest")
        assert "predictions" in result
        assert "actuals" in result
        assert "folds" in result
        assert len(result["predictions"]) == len(y) - 3

    def test_gbt_walk_forward(self, model, large_synthetic_data):
        X, y = large_synthetic_data
        model.config.min_train_years = 3
        result = model.walk_forward_validate_tree(X, y, model_type="gradient_boosting")
        assert len(result["folds"]) == len(y) - 3

    def test_fold_structure(self, model, large_synthetic_data):
        X, y = large_synthetic_data
        model.config.min_train_years = 3
        result = model.walk_forward_validate_tree(X, y, model_type="random_forest")
        for fold in result["folds"]:
            assert "train_size" in fold
            assert "test_index" in fold
            assert "predicted" in fold
            assert "actual" in fold

    def test_insufficient_data_returns_empty(self, model):
        X = pd.DataFrame({"a": [1, 2]})
        y = pd.Series([1.0, 2.0])
        model.config.min_train_years = 3
        result = model.walk_forward_validate_tree(X, y)
        assert result["predictions"] == []
        assert result["actuals"] == []
        assert result["folds"] == []

    def test_min_train_respected(self, model, large_synthetic_data):
        X, y = large_synthetic_data
        model.config.min_train_years = 10
        result = model.walk_forward_validate_tree(X, y, model_type="random_forest")
        assert len(result["folds"]) == len(y) - 10


# ------------------------------------------------------------------
# Tests: generate_tree_model_audit
# ------------------------------------------------------------------

class TestGenerateTreeModelAudit:
    def test_audit_contains_model_type(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        audit = model.generate_tree_model_audit(
            tree_model=fitted,
            tree_walk_forward=None,
            baselines={},
            features=synthetic_features,
            target=synthetic_target,
        )
        assert "RandomForestRegressor" in audit
        assert "Tree Model Audit" in audit

    def test_audit_contains_hyperparameters(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="gradient_boosting",
        )
        audit = model.generate_tree_model_audit(
            tree_model=fitted,
            tree_walk_forward=None,
            baselines={},
            features=synthetic_features,
        )
        assert "n_estimators" in audit
        assert "max_depth" in audit

    def test_audit_with_walk_forward(self, model, large_synthetic_data):
        X, y = large_synthetic_data
        model.config.min_train_years = 3
        fitted, _ = model.train_tree_model(X, y, model_type="gradient_boosting")
        wf = model.walk_forward_validate_tree(X, y, model_type="gradient_boosting")
        baselines = model.compute_baselines(y)

        audit = model.generate_tree_model_audit(
            tree_model=fitted,
            tree_walk_forward=wf,
            baselines=baselines,
            features=X,
            target=y,
        )
        assert "MAE" in audit
        assert "RMSE" in audit
        assert "Baseline Comparison" in audit

    def test_audit_with_shap_importance(self, model, synthetic_features, synthetic_target):
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        _, shap_imp = model.compute_shap_values(fitted, synthetic_features)

        audit = model.generate_tree_model_audit(
            tree_model=fitted,
            tree_walk_forward=None,
            baselines={},
            features=synthetic_features,
            shap_importance=shap_imp,
        )
        assert "SHAP" in audit
        assert "Feature Importance" in audit

    def test_audit_with_builtin_importance_fallback(self, model, synthetic_features, synthetic_target):
        """When SHAP is not available, built-in importance should be used."""
        fitted, _ = model.train_tree_model(
            synthetic_features, synthetic_target, model_type="random_forest",
        )
        builtin_imp = model.compute_feature_importances(fitted, synthetic_features)

        audit = model.generate_tree_model_audit(
            tree_model=fitted,
            tree_walk_forward=None,
            baselines={},
            features=synthetic_features,
            shap_importance=None,
            builtin_importance=builtin_imp,
        )
        assert "Built-in feature importance" in audit or "Gini" in audit

    def test_audit_none_model(self, model, synthetic_features):
        audit = model.generate_tree_model_audit(
            tree_model=None,
            tree_walk_forward=None,
            baselines={},
            features=synthetic_features,
        )
        assert "not trained" in audit.lower()

    def test_audit_primary_model_comparison(self, model, large_synthetic_data):
        """Audit should compare tree model with primary model when both available."""
        X, y = large_synthetic_data
        model.config.min_train_years = 3
        fitted_tree, _ = model.train_tree_model(X, y, model_type="gradient_boosting")
        tree_wf = model.walk_forward_validate_tree(X, y, model_type="gradient_boosting")
        primary_wf = model.walk_forward_validate(X, y)

        audit = model.generate_tree_model_audit(
            tree_model=fitted_tree,
            tree_walk_forward=tree_wf,
            baselines={},
            features=X,
            target=y,
            primary_walk_forward=primary_wf,
        )
        assert "Comparison with Primary Model" in audit
        assert "Primary model MAE" in audit
        assert "Tree model MAE" in audit


# ------------------------------------------------------------------
# Tests: Edge cases
# ------------------------------------------------------------------

class TestTreeModelEdgeCases:
    def test_too_few_samples(self, model):
        """Training with only 1 sample should return None."""
        X = pd.DataFrame({"a": [1.0]})
        y = pd.Series([10.0])
        fitted, importances = model.train_tree_model(X, y)
        assert fitted is None
        assert importances == {}

    def test_single_feature(self, model):
        """Training with a single feature should work."""
        X = pd.DataFrame({"only_feature": [1, 2, 3, 4, 5]})
        y = pd.Series([10, 20, 30, 40, 50])
        fitted, importances = model.train_tree_model(X, y, model_type="random_forest")
        assert fitted is not None
        assert "only_feature" in importances
        assert importances["only_feature"] == pytest.approx(1.0, abs=1e-6)

    def test_all_zero_target(self, model):
        """Training with all-zero target should still produce a model."""
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        fitted, importances = model.train_tree_model(X, y, model_type="gradient_boosting")
        assert fitted is not None
        # Predictions should be near zero
        preds = fitted.predict(X.fillna(0.0))
        assert np.allclose(preds, 0.0, atol=0.01)

    def test_nan_in_features(self, model):
        """NaN in features should be handled (filled with 0)."""
        X = pd.DataFrame({"a": [1, np.nan, 3, 4, 5], "b": [5, 4, np.nan, 2, 1]})
        y = pd.Series([10, 20, 30, 40, 50])
        fitted, importances = model.train_tree_model(X, y, model_type="random_forest")
        assert fitted is not None
        assert len(importances) == 2

    def test_all_nan_target(self, model):
        """All-NaN target should return None (no valid rows)."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([np.nan, np.nan, np.nan])
        fitted, importances = model.train_tree_model(X, y)
        assert fitted is None
        assert importances == {}


# ------------------------------------------------------------------
# Tests: Integration with overall model audit
# ------------------------------------------------------------------

class TestTreeModelAuditIntegration:
    def test_full_audit_includes_tree_section(self, model, tmp_path):
        """generate_model_audit with tree model should include tree audit section."""
        model.config.outputs_dir = tmp_path
        X = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [7, 6, 5, 4, 3, 2, 1],
        })
        y = pd.Series([10, 20, 30, 40, 50, 60, 70])

        # Train primary model
        primary_fitted, _ = model.train_primary_model(X, y)
        primary_wf = model.walk_forward_validate(X, y)
        baselines = model.compute_baselines(y)

        # Train tree model
        tree_fitted, _ = model.train_tree_model(X, y, model_type="gradient_boosting")
        tree_imp = model.compute_feature_importances(tree_fitted, X)
        tree_wf = model.walk_forward_validate_tree(X, y, model_type="gradient_boosting")
        _, shap_imp = model.compute_shap_values(tree_fitted, X)

        audit = model.generate_model_audit(
            model=primary_fitted,
            baselines=baselines,
            features=X,
            walk_forward_results=primary_wf,
            target=y,
            tree_model=tree_fitted,
            tree_importances=tree_imp,
            tree_walk_forward=tree_wf,
            shap_importance=shap_imp,
        )

        # Verify both primary and tree sections exist
        assert "Model Audit — ML Driver Model" in audit
        assert "Tree Model Audit (Supplementary)" in audit
        assert "GradientBoostingRegressor" in audit

        # Verify file was saved
        audit_path = tmp_path / "model_audit.md"
        assert audit_path.exists()
        saved_content = audit_path.read_text()
        assert "Tree Model Audit" in saved_content

    def test_audit_without_tree_model_has_no_tree_section(self, model, tmp_path):
        """generate_model_audit without tree model should not include tree section."""
        model.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = model.train_primary_model(X, y)
        baselines = model.compute_baselines(y)

        audit = model.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert "Tree Model Audit" not in audit

    def test_tree_importances_in_main_audit(self, model, tmp_path):
        """Tree feature importances table should appear in main audit body."""
        model.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = model.train_primary_model(X, y)
        tree_fitted, _ = model.train_tree_model(X, y, model_type="random_forest")
        tree_imp = model.compute_feature_importances(tree_fitted, X)
        baselines = model.compute_baselines(y)

        audit = model.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
            tree_model=tree_fitted, tree_importances=tree_imp,
        )
        assert "Tree Model Feature Importances" in audit

    def test_shap_summary_includes_top_features(self, model, tmp_path):
        """SHAP importance in audit should list top features."""
        model.config.outputs_dir = tmp_path
        np.random.seed(42)
        n = 15
        X = pd.DataFrame({
            "revenue_growth": np.random.uniform(-0.1, 0.5, n),
            "margin": np.random.uniform(0.2, 0.6, n),
            "sentiment": np.random.uniform(-1, 1, n),
        })
        y = 0.5 * X["revenue_growth"] + 0.3 * X["margin"] + 0.1 * np.random.randn(n)

        fitted, _ = model.train_primary_model(X, y)
        tree_fitted, _ = model.train_tree_model(X, y, model_type="gradient_boosting")
        tree_imp = model.compute_feature_importances(tree_fitted, X)
        _, shap_imp = model.compute_shap_values(tree_fitted, X)
        baselines = model.compute_baselines(y)

        audit = model.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
            tree_model=tree_fitted, tree_importances=tree_imp,
            shap_importance=shap_imp,
        )
        # SHAP section should mention top features
        assert "SHAP" in audit
        assert "revenue_growth" in audit
        assert "margin" in audit
        assert "sentiment" in audit
