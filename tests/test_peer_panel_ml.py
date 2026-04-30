"""
Tests for semiconductor peer-panel ML model (Task 17.5).

Covers:
- Panel dataset construction from NVDA + peer data
- Model training with synthetic peer panel data
- Walk-forward validation on panel data
- Graceful handling when peer data is insufficient (< 10 years)
- Missing peers are skipped with logging
- Audit report generation for peer-panel model
- Edge cases: no peer data, single peer, all NaN peer data
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.config import EngineConfig
from src.ml_models import MLDriverModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nvda_features(n: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic NVDA feature matrix and target."""
    np.random.seed(42)
    features = pd.DataFrame({
        "revenue_growth_yoy": np.random.uniform(-0.1, 0.5, n),
        "gross_margin": np.random.uniform(0.55, 0.75, n),
        "operating_margin": np.random.uniform(0.20, 0.50, n),
        "fcf_margin": np.random.uniform(0.15, 0.40, n),
    })
    target = pd.Series(np.random.uniform(-0.05, 0.6, n), name="target")
    return features, target


def _make_peer_data(
    tickers: list[str],
    rows_per_peer: int = 10,
    include_target: bool = True,
    all_nan_features: bool = False,
    nan_target: bool = False,
) -> pd.DataFrame:
    """Create synthetic peer data matching NVDA feature columns."""
    np.random.seed(123)
    frames = []
    for ticker in tickers:
        n = rows_per_peer
        if all_nan_features:
            df = pd.DataFrame({
                "revenue_growth_yoy": [np.nan] * n,
                "gross_margin": [np.nan] * n,
                "operating_margin": [np.nan] * n,
                "fcf_margin": [np.nan] * n,
                "ticker": [ticker] * n,
            })
        else:
            df = pd.DataFrame({
                "revenue_growth_yoy": np.random.uniform(-0.2, 0.4, n),
                "gross_margin": np.random.uniform(0.40, 0.65, n),
                "operating_margin": np.random.uniform(0.10, 0.35, n),
                "fcf_margin": np.random.uniform(0.05, 0.30, n),
                "ticker": [ticker] * n,
            })
        if include_target:
            if nan_target:
                df["target_value"] = [np.nan] * n
            else:
                df["target_value"] = np.random.uniform(-0.1, 0.5, n)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: Panel dataset construction
# ---------------------------------------------------------------------------

class TestBuildPeerPanelDataset:
    """Test build_peer_panel_dataset constructs panel data correctly."""

    def test_no_peer_data_returns_nvda_only(self):
        """Without peer data, pooled dataset equals NVDA data."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=None,
        )
        assert len(pooled_X) == 10
        assert len(pooled_y) == 10
        assert peer_info == {}

    def test_empty_peer_data_returns_nvda_only(self):
        """Empty peer DataFrame returns NVDA-only dataset."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=pd.DataFrame(),
        )
        assert len(pooled_X) == 10
        assert peer_info == {}

    def test_peers_with_sufficient_history_included(self):
        """Peers with ≥ 10 rows are included in the panel."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(["AMD", "INTC"], rows_per_peer=10)

        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        # 10 NVDA + 10 AMD + 10 INTC = 30
        assert len(pooled_X) == 30
        assert len(pooled_y) == 30
        assert peer_info["AMD"]["included"] is True
        assert peer_info["INTC"]["included"] is True

    def test_peers_with_insufficient_history_excluded(self):
        """Peers with < 10 rows are excluded from the panel."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=5)

        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        # Only NVDA rows
        assert len(pooled_X) == 10
        assert peer_info["AMD"]["included"] is False
        assert "insufficient history" in peer_info["AMD"]["reason"]

    def test_mixed_peer_history_lengths(self):
        """Some peers included, some excluded based on history length."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        # AMD has 10 rows (included), QCOM has 5 rows (excluded)
        amd_data = _make_peer_data(["AMD"], rows_per_peer=10)
        qcom_data = _make_peer_data(["QCOM"], rows_per_peer=5)
        peer_data = pd.concat([amd_data, qcom_data], ignore_index=True)

        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        # 10 NVDA + 10 AMD = 20 (QCOM excluded)
        assert len(pooled_X) == 20
        assert peer_info["AMD"]["included"] is True
        assert peer_info["QCOM"]["included"] is False

    def test_all_nan_peer_features_excluded(self):
        """Peers with all NaN features are excluded."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD"], rows_per_peer=10, all_nan_features=True,
        )

        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        assert len(pooled_X) == 10  # NVDA only
        assert peer_info["AMD"]["included"] is False
        assert "NaN" in peer_info["AMD"]["reason"]

    def test_no_common_features_returns_nvda_only(self):
        """If peer features don't overlap with NVDA, return NVDA-only."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = pd.DataFrame({
            "completely_different_feature": np.random.uniform(0, 1, 10),
            "target_value": np.random.uniform(0, 0.5, 10),
            "ticker": ["AMD"] * 10,
        })

        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        assert len(pooled_X) == 10  # NVDA only

    def test_peer_data_missing_target_column(self):
        """Peer data without target_value returns NVDA-only."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD"], rows_per_peer=10, include_target=False,
        )

        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        assert len(pooled_X) == 10  # NVDA only

    def test_peer_with_nan_targets_excluded(self):
        """Peers with all NaN target values are excluded."""
        model = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD"], rows_per_peer=10, nan_target=True,
        )

        pooled_X, pooled_y, peer_info = model.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        assert len(pooled_X) == 10  # NVDA only
        assert peer_info["AMD"]["included"] is False
        assert "no valid target" in peer_info["AMD"]["reason"]


# ---------------------------------------------------------------------------
# Tests: Model training
# ---------------------------------------------------------------------------

class TestTrainPeerPanelModel:
    """Test train_peer_panel_model with various data scenarios."""

    def test_no_peer_data_returns_none(self):
        """Without peer data, should return (None, {})."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=None,
        )
        assert model is None
        assert coeffs == {}

    def test_empty_peer_data_returns_none(self):
        """Empty peer DataFrame should return (None, {})."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=pd.DataFrame(),
        )
        assert model is None
        assert coeffs == {}

    def test_valid_peer_data_trains_model(self):
        """With valid peer data, should return a fitted model."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD", "INTC"], rows_per_peer=10,
        )

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        assert model is not None
        assert len(coeffs) == 4  # 4 common features
        assert "revenue_growth_yoy" in coeffs
        assert "gross_margin" in coeffs

    def test_peer_data_missing_target_returns_none(self):
        """Peer data without target_value returns (None, {})."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD"], rows_per_peer=10, include_target=False,
        )

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        assert model is None
        assert coeffs == {}

    def test_no_common_features_returns_none(self):
        """If NVDA and peer features have no overlap, returns (None, {})."""
        model_mgr = MLDriverModel()
        features = pd.DataFrame({"nvda_only": [1, 2, 3]})
        target = pd.Series([0.1, 0.2, 0.3])
        peer_data = pd.DataFrame({
            "peer_only": [4, 5, 6],
            "target_value": [0.4, 0.5, 0.6],
            "ticker": ["AMD"] * 3,
        })

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        assert model is None
        assert coeffs == {}

    def test_single_peer_trains_model(self):
        """A single peer with sufficient data should train successfully."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=10)

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        assert model is not None
        assert len(coeffs) == 4

    def test_model_coefficients_are_numeric(self):
        """All coefficients should be finite numbers."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD", "INTC", "AVGO"], rows_per_peer=10,
        )

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        assert model is not None
        for name, val in coeffs.items():
            assert np.isfinite(val), f"Coefficient {name} is not finite: {val}"


# ---------------------------------------------------------------------------
# Tests: Walk-forward validation on panel data
# ---------------------------------------------------------------------------

class TestWalkForwardValidatePanel:
    """Test walk-forward validation on pooled panel data."""

    def test_walk_forward_with_sufficient_data(self):
        """Walk-forward should produce predictions with enough data."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(15)
        peer_data = _make_peer_data(
            ["AMD", "INTC"], rows_per_peer=15,
        )

        pooled_X, pooled_y, _ = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )

        results = model_mgr.walk_forward_validate_panel(pooled_X, pooled_y)
        assert len(results["predictions"]) > 0
        assert len(results["actuals"]) > 0
        assert len(results["folds"]) > 0
        assert len(results["predictions"]) == len(results["actuals"])

    def test_walk_forward_insufficient_data(self):
        """Walk-forward with too few rows returns empty results."""
        model_mgr = MLDriverModel()
        features = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        })
        target = pd.Series([0.1, 0.2, 0.3])

        results = model_mgr.walk_forward_validate_panel(features, target)
        assert results["predictions"] == []
        assert results["actuals"] == []
        assert results["folds"] == []

    def test_walk_forward_predictions_are_finite(self):
        """All walk-forward predictions should be finite numbers."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(20)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=20)

        pooled_X, pooled_y, _ = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )

        results = model_mgr.walk_forward_validate_panel(pooled_X, pooled_y)
        for pred in results["predictions"]:
            assert np.isfinite(pred), f"Non-finite prediction: {pred}"

    def test_walk_forward_folds_have_correct_structure(self):
        """Each fold should have train_size, test_index, predicted, actual."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(12)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=12)

        pooled_X, pooled_y, _ = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )

        results = model_mgr.walk_forward_validate_panel(pooled_X, pooled_y)
        for fold in results["folds"]:
            assert "train_size" in fold
            assert "test_index" in fold
            assert "predicted" in fold
            assert "actual" in fold
            assert fold["train_size"] >= 5  # min_train


# ---------------------------------------------------------------------------
# Tests: Graceful handling of insufficient peer data
# ---------------------------------------------------------------------------

class TestInsufficientPeerData:
    """Test graceful handling when peer data is insufficient."""

    def test_all_peers_below_10_year_threshold(self):
        """When all peers have < 10 years, panel is NVDA-only."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD", "INTC", "AVGO"], rows_per_peer=5,
        )

        pooled_X, pooled_y, peer_info = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        assert len(pooled_X) == 10  # NVDA only
        for ticker in ["AMD", "INTC", "AVGO"]:
            assert peer_info[ticker]["included"] is False

    def test_logging_on_skipped_peers(self, caplog):
        """Skipped peers should produce log messages."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=5)

        with caplog.at_level(logging.INFO):
            model_mgr.build_peer_panel_dataset(
                features, target, peer_data=peer_data,
            )
        assert any("Skipping peer AMD" in msg for msg in caplog.messages)

    def test_logging_on_no_peer_data(self, caplog):
        """No peer data should produce a warning."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)

        with caplog.at_level(logging.WARNING):
            model_mgr.build_peer_panel_dataset(
                features, target, peer_data=None,
            )
        assert any("No peer data" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Tests: Audit report generation
# ---------------------------------------------------------------------------

class TestPeerPanelAudit:
    """Test audit report generation for peer-panel model."""

    def test_audit_when_model_not_trained(self):
        """Audit should report model was not trained."""
        model_mgr = MLDriverModel()
        audit = model_mgr.generate_peer_panel_audit(
            panel_model=None,
            panel_coefficients={},
            panel_walk_forward={},
            peer_info={},
            nvda_obs=10,
        )
        assert "not trained" in audit.lower()
        assert "Peer-Panel ML Model Audit" in audit

    def test_audit_with_trained_model(self):
        """Audit should include coefficients and walk-forward results."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(15)
        peer_data = _make_peer_data(["AMD", "INTC"], rows_per_peer=15)

        pooled_X, pooled_y, peer_info = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        wf = model_mgr.walk_forward_validate_panel(pooled_X, pooled_y)

        audit = model_mgr.generate_peer_panel_audit(
            panel_model=model,
            panel_coefficients=coeffs,
            panel_walk_forward=wf,
            peer_info=peer_info,
            nvda_obs=len(target),
        )
        assert "Peer-Panel ML Model Audit" in audit
        assert "ElasticNet" in audit
        assert "Panel Model Coefficients" in audit
        assert "Panel Walk-Forward Validation" in audit
        assert "MAE" in audit
        assert "Peer Data Availability" in audit
        assert "AMD" in audit
        assert "INTC" in audit

    def test_audit_shows_skipped_peers(self):
        """Audit should list peers that were skipped."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        # AMD has 10 rows (included), QCOM has 3 rows (excluded)
        amd_data = _make_peer_data(["AMD"], rows_per_peer=10)
        qcom_data = _make_peer_data(["QCOM"], rows_per_peer=3)
        peer_data = pd.concat([amd_data, qcom_data], ignore_index=True)

        pooled_X, pooled_y, peer_info = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )

        audit = model_mgr.generate_peer_panel_audit(
            panel_model=model,
            panel_coefficients=coeffs,
            panel_walk_forward={},
            peer_info=peer_info,
            nvda_obs=len(target),
        )
        assert "QCOM" in audit
        assert "✗" in audit  # excluded marker
        assert "✓" in audit  # included marker

    def test_audit_includes_limitations(self):
        """Audit should include a limitations section."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=10)

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )

        audit = model_mgr.generate_peer_panel_audit(
            panel_model=model,
            panel_coefficients=coeffs,
            panel_walk_forward={},
            peer_info={"AMD": {"rows": 10, "included": True, "reason": "included"}},
            nvda_obs=10,
        )
        assert "Limitations" in audit
        assert "supplementary" in audit.lower()

    def test_audit_with_no_walk_forward(self):
        """Audit handles missing walk-forward results gracefully."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=10)

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )

        audit = model_mgr.generate_peer_panel_audit(
            panel_model=model,
            panel_coefficients=coeffs,
            panel_walk_forward={},
            peer_info={"AMD": {"rows": 10, "included": True, "reason": "included"}},
            nvda_obs=10,
        )
        assert "not performed" in audit.lower() or "insufficient" in audit.lower()


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestPeerPanelEdgeCases:
    """Test edge cases for peer-panel ML."""

    def test_single_peer_single_feature(self):
        """Minimal case: one peer, one feature."""
        model_mgr = MLDriverModel()
        features = pd.DataFrame({"feat_a": np.random.uniform(0, 1, 10)})
        target = pd.Series(np.random.uniform(0, 0.5, 10))
        peer_data = pd.DataFrame({
            "feat_a": np.random.uniform(0, 1, 10),
            "target_value": np.random.uniform(0, 0.5, 10),
            "ticker": ["AMD"] * 10,
        })

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        assert model is not None
        assert "feat_a" in coeffs

    def test_all_nan_peer_data(self):
        """All NaN peer features should be excluded gracefully."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(
            ["AMD", "INTC"], rows_per_peer=10, all_nan_features=True,
        )

        pooled_X, pooled_y, peer_info = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        assert len(pooled_X) == 10  # NVDA only
        assert peer_info["AMD"]["included"] is False
        assert peer_info["INTC"]["included"] is False

    def test_many_peers_all_included(self):
        """All 5 core semiconductor peers with sufficient data."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(12)
        peer_data = _make_peer_data(
            ["AMD", "AVGO", "INTC", "QCOM", "MRVL"],
            rows_per_peer=12,
        )

        pooled_X, pooled_y, peer_info = model_mgr.build_peer_panel_dataset(
            features, target, peer_data=peer_data,
        )
        # 12 NVDA + 5 * 12 peers = 72
        assert len(pooled_X) == 72
        for ticker in ["AMD", "AVGO", "INTC", "QCOM", "MRVL"]:
            assert peer_info[ticker]["included"] is True

    def test_peer_data_with_extra_columns_ignored(self):
        """Extra columns in peer data that aren't in NVDA are ignored."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        peer_data = _make_peer_data(["AMD"], rows_per_peer=10)
        peer_data["extra_column"] = np.random.uniform(0, 1, 10)

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        assert model is not None
        assert "extra_column" not in coeffs

    def test_pooled_target_nan_rows_dropped(self):
        """Rows with NaN target in pooled data are dropped during training."""
        model_mgr = MLDriverModel()
        features, target = _make_nvda_features(10)
        # Create peer data where some targets are NaN
        peer_data = _make_peer_data(["AMD"], rows_per_peer=10)
        peer_data.loc[0:2, "target_value"] = np.nan

        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=peer_data,
        )
        # Should still train (enough valid rows)
        assert model is not None
