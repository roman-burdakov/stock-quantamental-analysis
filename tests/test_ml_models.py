"""
Tests for src/ml_models.py — MLDriverModel.

Covers: feature/target matrix construction, no-lookahead validation,
model training, walk-forward validation, baselines, evaluation, and
model audit generation.
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
def sample_metrics() -> pd.DataFrame:
    """Minimal metrics DataFrame spanning 5 annual periods."""
    rows = []
    periods = [
        ("FY2020", "2020-02-20", "2020-02-20", "acc-2020"),
        ("FY2021", "2021-02-24", "2021-02-24", "acc-2021"),
        ("FY2022", "2022-02-23", "2022-02-23", "acc-2022"),
        ("FY2023", "2023-02-22", "2023-02-22", "acc-2023"),
        ("FY2024", "2024-02-21", "2024-02-21", "acc-2024"),
    ]
    metric_values = {
        "gross_margin": [0.62, 0.65, 0.66, 0.57, 0.73],
        "operating_margin": [0.30, 0.35, 0.38, 0.17, 0.54],
        "FCF_margin": [0.25, 0.30, 0.33, 0.15, 0.44],
        "revenue_growth_YoY": [None, 0.53, 0.61, -0.21, 1.26],
    }
    for i, (fp, fd, sad, acc) in enumerate(periods):
        for metric_name, values in metric_values.items():
            rows.append({
                "ticker": "NVDA",
                "fiscal_period": fp,
                "filing_date": fd,
                "source_available_date": sad,
                "source_accession": acc,
                "metric_name": metric_name,
                "metric_value": values[i],
                "unit": "ratio",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_nlp() -> pd.DataFrame:
    """Minimal NLP features DataFrame."""
    rows = []
    periods = [
        ("2020-02-20", "2020-02-20", "acc-2020"),
        ("2021-02-24", "2021-02-24", "acc-2021"),
        ("2022-02-23", "2022-02-23", "acc-2022"),
        ("2023-02-22", "2023-02-22", "acc-2023"),
        ("2024-02-21", "2024-02-21", "acc-2024"),
    ]
    for fd, sad, acc in periods:
        rows.append({
            "filing_date": fd,
            "source_available_date": sad,
            "source_accession": acc,
            "section": "mda",
            "feature_type": "keyword_score",
            "feature_name": "ai_accelerated_computing",
            "value": np.random.uniform(0.001, 0.05),
            "prev_filing_date": None,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Tests: build_feature_target_matrix
# ------------------------------------------------------------------

class TestBuildFeatureTargetMatrix:
    def test_output_has_required_columns(self, model, sample_metrics, sample_nlp):
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        required = [
            "feature_period", "feature_available_date", "prediction_date",
            "target_period", "target_available_date", "target_name",
            "target_value", "source_accessions",
        ]
        for col in required:
            assert col in matrix.columns, f"Missing column: {col}"

    def test_target_shifted_forward(self, model, sample_metrics, sample_nlp):
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        # For all rows except the last, target_period should be the next period
        for i in range(len(matrix) - 1):
            assert matrix.iloc[i]["target_period"] == matrix.iloc[i + 1]["feature_period"]

    def test_last_row_has_empty_target(self, model, sample_metrics, sample_nlp):
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        last = matrix.iloc[-1]
        assert last["target_period"] == ""
        assert last["target_available_date"] == ""

    def test_saves_csv(self, model, sample_metrics, sample_nlp, tmp_path):
        model.config.processed_dir = tmp_path
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        csv_path = tmp_path / "ml_feature_target_matrix.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(matrix)

    def test_empty_nlp_still_works(self, model, sample_metrics, tmp_path):
        model.config.processed_dir = tmp_path
        empty_nlp = pd.DataFrame()
        matrix = model.build_feature_target_matrix(sample_metrics, empty_nlp)
        assert len(matrix) > 0
        assert "feature_period" in matrix.columns


# ------------------------------------------------------------------
# Tests: validate_no_lookahead_matrix
# ------------------------------------------------------------------

class TestValidateNoLookahead:
    def test_valid_matrix_passes(self, model, sample_metrics, sample_nlp, tmp_path):
        model.config.processed_dir = tmp_path
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        assert model.validate_no_lookahead_matrix(matrix) is True

    def test_feature_after_prediction_fails(self, model):
        matrix = pd.DataFrame([{
            "feature_available_date": "2024-03-01",
            "prediction_date": "2024-02-01",
            "target_available_date": "2025-02-01",
        }])
        assert model.validate_no_lookahead_matrix(matrix) is False

    def test_target_before_prediction_fails(self, model):
        matrix = pd.DataFrame([{
            "feature_available_date": "2024-01-01",
            "prediction_date": "2024-02-01",
            "target_available_date": "2024-01-15",
        }])
        assert model.validate_no_lookahead_matrix(matrix) is False

    def test_empty_matrix_passes(self, model):
        assert model.validate_no_lookahead_matrix(pd.DataFrame()) is True


# ------------------------------------------------------------------
# Tests: train_primary_model
# ------------------------------------------------------------------

class TestTrainPrimaryModel:
    def test_ridge_returns_model_and_coefficients(self, model):
        model.config.primary_model = "ridge"
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, coeffs = model.train_primary_model(X, y)
        assert fitted is not None
        assert "a" in coeffs
        assert "b" in coeffs

    def test_elasticnet_returns_model_and_coefficients(self, model):
        model.config.primary_model = "elasticnet"
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, coeffs = model.train_primary_model(X, y)
        assert fitted is not None
        assert len(coeffs) == 2

    def test_insufficient_data_returns_none(self, model):
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([10])
        fitted, coeffs = model.train_primary_model(X, y)
        assert fitted is None
        assert coeffs == {}


# ------------------------------------------------------------------
# Tests: walk_forward_validate
# ------------------------------------------------------------------

class TestWalkForwardValidate:
    def test_returns_predictions_and_actuals(self, model):
        model.config.min_train_years = 3
        X = pd.DataFrame({"a": range(6), "b": range(6, 0, -1)})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = model.walk_forward_validate(X, y)
        assert "predictions" in result
        assert "actuals" in result
        assert "folds" in result
        assert len(result["predictions"]) == 3  # 6 - 3 = 3 folds

    def test_min_train_respected(self, model):
        model.config.min_train_years = 4
        X = pd.DataFrame({"a": range(6)})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = model.walk_forward_validate(X, y)
        assert len(result["folds"]) == 2  # 6 - 4 = 2 folds


# ------------------------------------------------------------------
# Tests: compute_baselines
# ------------------------------------------------------------------

class TestComputeBaselines:
    def test_returns_all_four_baselines(self, model):
        target = pd.Series([0.1, 0.2, 0.15, 0.3, 0.25])
        baselines = model.compute_baselines(target)
        assert set(baselines.keys()) == {
            "last_period", "trailing_4q_avg", "three_year_avg", "linear_trend",
        }

    def test_last_period_uses_previous_value(self, model):
        target = pd.Series([0.1, 0.2, 0.3])
        baselines = model.compute_baselines(target)
        assert np.isnan(baselines["last_period"][0])
        assert baselines["last_period"][1] == pytest.approx(0.1)
        assert baselines["last_period"][2] == pytest.approx(0.2)

    def test_baseline_lengths_match_target(self, model):
        target = pd.Series([0.1, 0.2, 0.15, 0.3, 0.25])
        baselines = model.compute_baselines(target)
        for bl_name, bl_preds in baselines.items():
            assert len(bl_preds) == len(target), f"{bl_name} length mismatch"


# ------------------------------------------------------------------
# Tests: evaluate
# ------------------------------------------------------------------

class TestEvaluate:
    def test_perfect_predictions(self, model):
        result = model.evaluate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0

    def test_known_errors(self, model):
        result = model.evaluate([1.0, 2.0], [2.0, 4.0])
        assert result["mae"] == pytest.approx(1.5)
        assert result["rmse"] == pytest.approx(np.sqrt(2.5), rel=1e-4)

    def test_empty_returns_nan(self, model):
        result = model.evaluate([], [])
        assert np.isnan(result["mae"])
        assert result["n_evaluated"] == 0

    def test_nan_excluded(self, model):
        result = model.evaluate([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
        assert result["n_evaluated"] == 2


# ------------------------------------------------------------------
# Tests: generate_model_audit
# ------------------------------------------------------------------

class TestGenerateModelAudit:
    def test_audit_saved_to_file(self, model, tmp_path):
        model.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = model.train_primary_model(X, y)
        baselines = model.compute_baselines(y)

        audit = model.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert (tmp_path / "model_audit.md").exists()
        assert "Model Audit" in audit

    def test_audit_contains_honest_disclosure(self, model, tmp_path):
        model.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([10, 20, 30])
        fitted, _ = model.train_primary_model(X, y)
        baselines = model.compute_baselines(y)

        audit = model.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
        )
        assert "Limitations" in audit
        assert "WARNING" in audit or "caveat" in audit.lower()

    def test_audit_with_no_model(self, model, tmp_path):
        model.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([10])
        baselines = model.compute_baselines(y)

        audit = model.generate_model_audit(
            model=None, baselines=baselines, features=X, target=y,
        )
        assert "unavailable" in audit.lower() or "not trained" in audit.lower()


# ------------------------------------------------------------------
# Tests: train_tree_model
# ------------------------------------------------------------------

class TestTrainTreeModel:
    def test_random_forest_returns_model_and_importances(self, model):
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40, 50])
        fitted, importances = model.train_tree_model(X, y, model_type="random_forest")
        assert fitted is not None
        assert "a" in importances
        assert "b" in importances
        assert all(v >= 0 for v in importances.values())

    def test_gradient_boosting_returns_model_and_importances(self, model):
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40, 50])
        fitted, importances = model.train_tree_model(X, y, model_type="gradient_boosting")
        assert fitted is not None
        assert len(importances) == 2

    def test_insufficient_data_returns_none(self, model):
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([10])
        fitted, importances = model.train_tree_model(X, y)
        assert fitted is None
        assert importances == {}

    def test_handles_nan_target(self, model):
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, np.nan, 30, 40])
        fitted, importances = model.train_tree_model(X, y)
        assert fitted is not None
        assert len(importances) == 2


# ------------------------------------------------------------------
# Tests: compute_feature_importances
# ------------------------------------------------------------------

class TestComputeFeatureImportances:
    def test_returns_dataframe_with_correct_columns(self, model):
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = model.train_tree_model(X, y)
        imp_df = model.compute_feature_importances(fitted, X)
        assert list(imp_df.columns) == ["feature", "importance"]
        assert len(imp_df) == 2

    def test_importances_sum_to_one(self, model):
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = model.train_tree_model(X, y)
        imp_df = model.compute_feature_importances(fitted, X)
        assert imp_df["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_sorted_descending(self, model):
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = model.train_tree_model(X, y)
        imp_df = model.compute_feature_importances(fitted, X)
        assert imp_df["importance"].is_monotonic_decreasing

    def test_none_model_returns_empty(self, model):
        X = pd.DataFrame({"a": [1, 2]})
        imp_df = model.compute_feature_importances(None, X)
        assert imp_df.empty
        assert list(imp_df.columns) == ["feature", "importance"]


# ------------------------------------------------------------------
# Tests: model audit with tree model
# ------------------------------------------------------------------

class TestModelAuditWithTreeModel:
    def test_audit_includes_tree_importances(self, model, tmp_path):
        model.config.outputs_dir = tmp_path
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([10, 20, 30, 40])
        fitted, _ = model.train_primary_model(X, y)
        tree_fitted, _ = model.train_tree_model(X, y)
        tree_imp = model.compute_feature_importances(tree_fitted, X)
        baselines = model.compute_baselines(y)

        audit = model.generate_model_audit(
            model=fitted, baselines=baselines, features=X, target=y,
            tree_model=tree_fitted, tree_importances=tree_imp,
        )
        assert "Tree Model Feature Importances" in audit
        assert "RandomForestRegressor" in audit
