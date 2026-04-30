"""
Tests for no-lookahead validation of the ML feature/target matrix.

Validates Requirement 12.3:
  - All feature_available_date <= prediction_date for every row
  - All target_available_date > prediction_date for every row
    (excluding last row which has no target)

Uses inline test data and MLDriverModel.build_feature_target_matrix().
Runs fully offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import EngineConfig
from src.ml_models import MLDriverModel


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def config(tmp_path) -> EngineConfig:
    cfg = EngineConfig()
    cfg.processed_dir = tmp_path
    return cfg


@pytest.fixture
def model(config) -> MLDriverModel:
    return MLDriverModel(config=config)


@pytest.fixture
def sample_metrics() -> pd.DataFrame:
    """Five annual periods with realistic filing dates."""
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
    """Minimal NLP features aligned to the same filing dates."""
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
            "value": 0.02,
            "prev_filing_date": None,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Tests: built matrix satisfies no-lookahead constraints
# ------------------------------------------------------------------

class TestBuiltMatrixNoLookahead:
    """Verify the matrix produced by build_feature_target_matrix
    satisfies point-in-time constraints (Req 8.6, 12.3)."""

    def test_feature_available_date_lte_prediction_date(
        self, model, sample_metrics, sample_nlp,
    ):
        """Every row must have feature_available_date <= prediction_date."""
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        for idx, row in matrix.iterrows():
            feat = str(row["feature_available_date"])
            pred = str(row["prediction_date"])
            assert feat <= pred, (
                f"Row {idx}: feature_available_date ({feat}) > "
                f"prediction_date ({pred})"
            )

    def test_target_available_date_gt_prediction_date(
        self, model, sample_metrics, sample_nlp,
    ):
        """For all rows except the last, target_available_date > prediction_date."""
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        # Last row has no target — exclude it
        rows_with_target = matrix[matrix["target_available_date"] != ""]
        for idx, row in rows_with_target.iterrows():
            tgt = str(row["target_available_date"])
            pred = str(row["prediction_date"])
            assert tgt > pred, (
                f"Row {idx}: target_available_date ({tgt}) <= "
                f"prediction_date ({pred})"
            )

    def test_validate_no_lookahead_passes_on_built_matrix(
        self, model, sample_metrics, sample_nlp,
    ):
        """validate_no_lookahead_matrix() returns True on a correctly built matrix."""
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        assert model.validate_no_lookahead_matrix(matrix) is True

    def test_last_row_excluded_from_target_check(
        self, model, sample_metrics, sample_nlp,
    ):
        """The last row has empty target_available_date and should not
        cause a validation failure."""
        matrix = model.build_feature_target_matrix(sample_metrics, sample_nlp)
        last = matrix.iloc[-1]
        assert last["target_available_date"] == ""
        assert last["target_period"] == ""
        # Validation still passes
        assert model.validate_no_lookahead_matrix(matrix) is True


# ------------------------------------------------------------------
# Tests: validate_no_lookahead_matrix catches violations
# ------------------------------------------------------------------

class TestValidateNoLookaheadCatchesViolations:
    """Confirm that deliberately invalid data is caught."""

    def test_detects_feature_after_prediction(self, model):
        """feature_available_date > prediction_date → violation."""
        bad_matrix = pd.DataFrame([{
            "feature_available_date": "2024-06-01",
            "prediction_date": "2024-02-01",
            "target_available_date": "2025-02-01",
        }])
        assert model.validate_no_lookahead_matrix(bad_matrix) is False

    def test_detects_target_on_prediction_date(self, model):
        """target_available_date == prediction_date → violation
        (must be strictly greater)."""
        bad_matrix = pd.DataFrame([{
            "feature_available_date": "2024-01-01",
            "prediction_date": "2024-02-01",
            "target_available_date": "2024-02-01",
        }])
        assert model.validate_no_lookahead_matrix(bad_matrix) is False

    def test_detects_target_before_prediction(self, model):
        """target_available_date < prediction_date → violation."""
        bad_matrix = pd.DataFrame([{
            "feature_available_date": "2024-01-01",
            "prediction_date": "2024-06-01",
            "target_available_date": "2024-03-01",
        }])
        assert model.validate_no_lookahead_matrix(bad_matrix) is False

    def test_multiple_violations_all_detected(self, model):
        """Matrix with multiple bad rows still returns False."""
        bad_matrix = pd.DataFrame([
            {
                "feature_available_date": "2024-05-01",
                "prediction_date": "2024-02-01",
                "target_available_date": "2025-02-01",
            },
            {
                "feature_available_date": "2024-01-01",
                "prediction_date": "2024-06-01",
                "target_available_date": "2024-03-01",
            },
        ])
        assert model.validate_no_lookahead_matrix(bad_matrix) is False

    def test_mixed_valid_and_invalid_rows(self, model):
        """One bad row among valid rows → still fails."""
        matrix = pd.DataFrame([
            {
                "feature_available_date": "2023-02-20",
                "prediction_date": "2023-02-20",
                "target_available_date": "2024-02-21",
            },
            {
                "feature_available_date": "2024-02-21",
                "prediction_date": "2024-02-21",
                "target_available_date": "2024-01-01",  # violation
            },
        ])
        assert model.validate_no_lookahead_matrix(matrix) is False

    def test_valid_handcrafted_matrix_passes(self, model):
        """A manually constructed valid matrix passes validation."""
        matrix = pd.DataFrame([
            {
                "feature_available_date": "2022-02-23",
                "prediction_date": "2022-02-23",
                "target_available_date": "2023-02-22",
            },
            {
                "feature_available_date": "2023-02-22",
                "prediction_date": "2023-02-22",
                "target_available_date": "2024-02-21",
            },
            {
                "feature_available_date": "2024-02-21",
                "prediction_date": "2024-02-21",
                "target_available_date": "",  # last row, no target
            },
        ])
        assert model.validate_no_lookahead_matrix(matrix) is True

    def test_empty_matrix_passes(self, model):
        """An empty matrix has no violations."""
        assert model.validate_no_lookahead_matrix(pd.DataFrame()) is True
