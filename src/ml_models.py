"""
NVDA Quantamental Engine — ML Driver Model.

Builds a feature/target matrix from financial metrics and NLP features,
trains Ridge or ElasticNet with walk-forward validation, computes
baselines, evaluates, and generates an honest model audit.

Requirements: 8.1–8.11
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge

from src.config import EngineConfig, FeatureTargetRecord

logger = logging.getLogger(__name__)


class MLDriverModel:
    """Interpretable ML model forecasting a financial driver."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------
    # 1. Feature / target matrix
    # ------------------------------------------------------------------

    def build_feature_target_matrix(
        self,
        metrics: pd.DataFrame,
        nlp: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge financial metrics + NLP features, shift target forward.

        Each row carries ``feature_available_date`` and
        ``target_available_date`` so that no-lookahead validation is
        possible.  Saves to ``data/processed/ml_feature_target_matrix.csv``.

        Parameters
        ----------
        metrics : pd.DataFrame
            Output of ``FinancialMetricsCalculator.compute_all_metrics()``.
            Expected columns: fiscal_period, filing_date,
            source_available_date, metric_name, metric_value,
            source_accession.
        nlp : pd.DataFrame
            Output of ``NLPFeatureExtractor.compute_narrative_drift()``.
            Expected columns: filing_date, source_available_date,
            source_accession, section, feature_type, feature_name, value.

        Returns
        -------
        pd.DataFrame
            FeatureTargetRecord schema + dynamic feature columns.
        """
        target_name = self.config.ml_target  # e.g. "revenue_growth_yoy"

        # --- Pivot metrics into wide format keyed by fiscal_period ---
        metrics_wide = self._pivot_metrics(metrics)

        # --- Pivot NLP features into wide format keyed by fiscal_period ---
        nlp_wide = self._pivot_nlp(nlp, metrics)

        # --- Merge on fiscal_period ---
        if nlp_wide is not None and not nlp_wide.empty:
            feature_df = metrics_wide.merge(
                nlp_wide, on="fiscal_period", how="left", suffixes=("", "_nlp"),
            )
            # Resolve duplicate date columns from merge
            for col in ["filing_date_nlp", "source_available_date_nlp",
                        "source_accession_nlp"]:
                if col in feature_df.columns:
                    feature_df.drop(columns=[col], inplace=True)
        else:
            feature_df = metrics_wide.copy()

        # --- Sort chronologically by actual date (not string period) ---
        # Using source_available_date ensures correct temporal ordering
        # regardless of fiscal_period string format (e.g., "FY2024" vs "FY2024Q1").
        if "source_available_date" in feature_df.columns:
            feature_df = feature_df.sort_values("source_available_date").reset_index(drop=True)
        else:
            logger.warning(
                "source_available_date not available for sorting; "
                "falling back to fiscal_period string sort — verify temporal order"
            )
            feature_df = feature_df.sort_values("fiscal_period").reset_index(drop=True)

        # --- Identify target column ---
        target_col = self._resolve_target_column(feature_df, target_name)
        if target_col is None:
            available_cols = [c for c in feature_df.columns if c not in {
                "fiscal_period", "filing_date", "source_available_date", "source_accession"
            }]
            logger.warning(
                "Target '%s' not found in feature columns. Matrix will have null targets. "
                "ML training will be skipped. Available metric columns: %s",
                target_name,
                ", ".join(available_cols[:20]),
            )

        # --- Build rows with shifted target ---
        rows: list[dict] = []
        for i in range(len(feature_df)):
            row = feature_df.iloc[i]
            feature_period = row["fiscal_period"]
            feature_available_date = row["source_available_date"]
            source_accession = row.get("source_accession", "")

            # Prediction date = feature_available_date (the day we make
            # the prediction, using only data available up to that point)
            prediction_date = feature_available_date

            # Target = next period's value of the target metric
            target_value = None
            target_period = ""
            target_available_date = ""
            if i + 1 < len(feature_df):
                next_row = feature_df.iloc[i + 1]
                target_period = next_row["fiscal_period"]
                target_available_date = next_row["source_available_date"]
                if target_col is not None and target_col in next_row.index:
                    val = next_row[target_col]
                    target_value = val if pd.notna(val) else None

            # Collect feature columns (everything except metadata)
            meta_cols = {
                "fiscal_period", "filing_date", "source_available_date",
                "source_accession",
            }
            feature_dict: dict = {
                "feature_period": feature_period,
                "feature_available_date": feature_available_date,
                "prediction_date": prediction_date,
                "target_period": target_period,
                "target_available_date": target_available_date,
                "target_name": target_name,
                "target_value": target_value,
                "source_accessions": source_accession,
            }
            for col in feature_df.columns:
                if col not in meta_cols:
                    val = row[col]
                    feature_dict[col] = val if pd.notna(val) else None

            rows.append(feature_dict)

        matrix = pd.DataFrame(rows)

        # Persist
        out_dir = Path(self.config.processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ml_feature_target_matrix.csv"
        matrix.to_csv(out_path, index=False)
        logger.info("Saved ML feature/target matrix to %s (%d rows)", out_path, len(matrix))

        return matrix

    # ------------------------------------------------------------------
    # 2. No-lookahead validation
    # ------------------------------------------------------------------

    def validate_no_lookahead_matrix(self, matrix: pd.DataFrame) -> bool:
        """Verify every row satisfies point-in-time constraints.

        A row is valid IFF:
        - ``feature_available_date <= prediction_date``
        - ``target_available_date > prediction_date``

        Rows with empty target_available_date (last row with no future
        target) are excluded from the target check.

        Returns True if all applicable rows pass.
        """
        if matrix.empty:
            return True

        violations = 0

        for idx, row in matrix.iterrows():
            feat_date = str(row.get("feature_available_date", ""))
            pred_date = str(row.get("prediction_date", ""))
            tgt_date = str(row.get("target_available_date", ""))

            # Feature availability check
            if feat_date and pred_date and feat_date > pred_date:
                logger.warning(
                    "Lookahead violation row %s: feature_available_date (%s) > prediction_date (%s)",
                    idx, feat_date, pred_date,
                )
                violations += 1

            # Target availability check (skip rows without a target)
            if tgt_date and pred_date and tgt_date <= pred_date:
                logger.warning(
                    "Lookahead violation row %s: target_available_date (%s) <= prediction_date (%s)",
                    idx, tgt_date, pred_date,
                )
                violations += 1

        if violations:
            logger.error("No-lookahead validation FAILED: %d violations", violations)
            return False

        logger.info("No-lookahead validation PASSED (%d rows checked)", len(matrix))
        return True

    # ------------------------------------------------------------------
    # 3. Train primary model
    # ------------------------------------------------------------------

    def train_primary_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple:
        """Train Ridge or ElasticNet based on ``config.primary_model``.

        Parameters
        ----------
        features : pd.DataFrame
            Numeric feature matrix (rows = observations, cols = features).
        target : pd.Series
            Target values aligned with *features*.

        Returns
        -------
        tuple[model, dict]
            Fitted model and dict mapping feature names to coefficients.
        """
        # Drop rows where target is NaN
        valid = target.notna()
        X = features.loc[valid].copy()
        y = target.loc[valid].copy()

        if len(y) < 2:
            logger.warning("Insufficient data to train model (%d rows)", len(y))
            return None, {}

        # Fill remaining NaN features with 0 (conservative)
        X = X.fillna(0.0)

        model_type = self.config.primary_model.lower()
        if model_type == "ridge":
            model = Ridge(alpha=1.0)
        else:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

        model.fit(X, y)

        coefficients = dict(zip(X.columns, model.coef_))
        logger.info(
            "Trained %s model on %d observations, %d features",
            model_type, len(y), X.shape[1],
        )
        return model, coefficients

    # ------------------------------------------------------------------
    # 4. Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> dict:
        """Expanding-window walk-forward validation.

        Minimum training window = ``config.min_train_years`` periods.
        Each fold trains on all prior data and predicts the next period.

        Returns
        -------
        dict
            ``{"predictions": [...], "actuals": [...], "folds": [...]}``.
            Each fold entry contains train_size, test_index, predicted,
            actual.
        """
        min_train = self.config.min_train_years
        valid = target.notna()
        X = features.loc[valid].copy().fillna(0.0)
        y = target.loc[valid].copy()

        n = len(y)

        if n < min_train + 1:
            logger.warning(
                "Insufficient data for walk-forward validation: %d rows available, "
                "need at least %d (min_train=%d + 1 test). "
                "Model audit will show no out-of-sample performance.",
                n, min_train + 1, min_train,
            )
            return {"predictions": [], "actuals": [], "folds": []}

        predictions: list[float] = []
        actuals: list[float] = []
        folds: list[dict] = []

        for i in range(min_train, n):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[[i]]
            y_test = y.iloc[i]

            model_type = self.config.primary_model.lower()
            if model_type == "ridge":
                model = Ridge(alpha=1.0)
            else:
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

            model.fit(X_train, y_train)
            pred = float(model.predict(X_test)[0])

            predictions.append(pred)
            actuals.append(float(y_test))
            folds.append({
                "train_size": i,
                "test_index": int(y.index[i]),
                "predicted": pred,
                "actual": float(y_test),
            })

        logger.info("Walk-forward validation: %d folds", len(folds))
        return {"predictions": predictions, "actuals": actuals, "folds": folds}

    # ------------------------------------------------------------------
    # 5. Baselines
    # ------------------------------------------------------------------

    def compute_baselines(self, target: pd.Series) -> dict:
        """Compute 4 baseline prediction strategies.

        Returns dict mapping baseline name to list of predictions
        aligned with the target (same length, NaN where not computable).
        """
        n = len(target)
        baselines: dict[str, list] = {
            "last_period": [],
            "trailing_4q_avg": [],
            "three_year_avg": [],
            "linear_trend": [],
        }

        for i in range(n):
            # last_period: use previous value
            if i >= 1 and pd.notna(target.iloc[i - 1]):
                baselines["last_period"].append(float(target.iloc[i - 1]))
            else:
                baselines["last_period"].append(np.nan)

            # trailing_4q_avg: average of last 4 values
            if i >= 4:
                window = target.iloc[i - 4: i]
                valid_window = window.dropna()
                if len(valid_window) >= 2:
                    baselines["trailing_4q_avg"].append(float(valid_window.mean()))
                else:
                    baselines["trailing_4q_avg"].append(np.nan)
            else:
                baselines["trailing_4q_avg"].append(np.nan)

            # three_year_avg: average of last 3 values (annual periods)
            if i >= 3:
                window = target.iloc[i - 3: i]
                valid_window = window.dropna()
                if len(valid_window) >= 2:
                    baselines["three_year_avg"].append(float(valid_window.mean()))
                else:
                    baselines["three_year_avg"].append(np.nan)
            else:
                baselines["three_year_avg"].append(np.nan)

            # linear_trend: fit line to all prior points, extrapolate
            if i >= 2:
                prior = target.iloc[:i]
                valid_idx = prior.dropna().index
                if len(valid_idx) >= 2:
                    x = np.arange(len(valid_idx), dtype=float)
                    y = prior.loc[valid_idx].values.astype(float)
                    coeffs = np.polyfit(x, y, 1)
                    pred = float(np.polyval(coeffs, len(valid_idx)))
                    baselines["linear_trend"].append(pred)
                else:
                    baselines["linear_trend"].append(np.nan)
            else:
                baselines["linear_trend"].append(np.nan)

        return baselines

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: list[float] | np.ndarray,
        actuals: list[float] | np.ndarray,
    ) -> dict:
        """Compute MAE, RMSE, and directional accuracy.

        Parameters
        ----------
        predictions, actuals : array-like
            Must be same length.  NaN entries are excluded pairwise.

        Returns
        -------
        dict
            ``{"mae": float, "rmse": float, "directional_accuracy": float,
               "n_evaluated": int}``.
        """
        preds = np.asarray(predictions, dtype=float)
        acts = np.asarray(actuals, dtype=float)

        valid = ~(np.isnan(preds) | np.isnan(acts))
        preds = preds[valid]
        acts = acts[valid]

        n = len(preds)
        if n == 0:
            return {"mae": np.nan, "rmse": np.nan, "directional_accuracy": np.nan, "n_evaluated": 0}

        errors = preds - acts
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # Directional accuracy: did prediction and actual move in the
        # same direction relative to the previous actual?
        if n >= 2:
            pred_dir = np.sign(np.diff(preds))
            act_dir = np.sign(np.diff(acts))
            dir_acc = float(np.mean(pred_dir == act_dir))
        else:
            dir_acc = np.nan

        return {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "directional_accuracy": round(dir_acc, 4) if not np.isnan(dir_acc) else np.nan,
            "n_evaluated": n,
        }

    # ------------------------------------------------------------------
    # 6b. ML quality assessment
    # ------------------------------------------------------------------

    def get_coefficient_summary(self, model) -> dict:
        """Return {feature_name: coefficient} for audit and quality assessment.

        Works with any sklearn linear model that exposes ``coef_`` and
        the feature names stored during ``train_primary_model``.

        Parameters
        ----------
        model
            A fitted sklearn linear model with ``coef_`` attribute.
            If *model* is None or lacks ``coef_``, returns an empty dict.

        Returns
        -------
        dict
            Mapping of feature name → coefficient value.
        """
        if model is None or not hasattr(model, "coef_"):
            return {}
        # coef_ may be stored alongside feature_names_in_ (sklearn ≥1.0)
        if hasattr(model, "feature_names_in_"):
            return dict(zip(model.feature_names_in_, model.coef_))
        # Fallback: return indexed keys
        return {f"feature_{i}": float(c) for i, c in enumerate(model.coef_)}

    def assess_ml_quality(
        self,
        model_results: dict,
        baseline_results: dict,
    ) -> "ComponentStatus":
        """Assess ML model quality and return a ComponentStatus.

        Checks (in order):
        1. **Degenerate model** — if max(|coef|) < 0.001, the model
           learned nothing useful → ``diagnostic_only``.
        2. **Underperforms baselines** — if the model's walk-forward MAE
           is worse than *every* baseline MAE → ``diagnostic_only``.
        3. Otherwise → ``usable``.

        Parameters
        ----------
        model_results : dict
            Must contain:
            - ``"coefficients"`` : dict  {feature: coef}
            - ``"walk_forward"``  : dict  output of ``walk_forward_validate``
            - ``"training_obs"``  : int   number of training observations
        baseline_results : dict
            Mapping baseline_name → evaluation dict (output of
            ``evaluate``).  Each value must have ``"mae"`` key.

        Returns
        -------
        ComponentStatus
        """
        from src.config import ComponentStatus, ComponentStatusEnum

        coefficients: dict = model_results.get("coefficients", {})
        walk_forward: dict = model_results.get("walk_forward", {})
        training_obs: int = model_results.get("training_obs", 0)

        # --- Check 1: degenerate model (all coefficients near-zero) ---
        if coefficients:
            max_abs_coef = max(abs(v) for v in coefficients.values())
            if max_abs_coef < 0.001:
                return ComponentStatus(
                    component_name="ml_signal",
                    status=ComponentStatusEnum.DIAGNOSTIC_ONLY,
                    reason="degenerate model",
                    details={
                        "max_abs_coefficient": max_abs_coef,
                        "training_observations": training_obs,
                    },
                )
        elif not coefficients:
            # No coefficients at all — model wasn't trained
            return ComponentStatus(
                component_name="ml_signal",
                status=ComponentStatusEnum.DIAGNOSTIC_ONLY,
                reason="model not trained or no coefficients available",
                details={"training_observations": training_obs},
            )

        # --- Check 2: model MAE worse than ALL baselines ---
        wf_preds = walk_forward.get("predictions", [])
        wf_actuals = walk_forward.get("actuals", [])

        if wf_preds and wf_actuals:
            model_eval = self.evaluate(wf_preds, wf_actuals)
            model_mae = model_eval.get("mae", np.nan)

            if not np.isnan(model_mae) and baseline_results:
                worse_than_all = True
                for bl_name, bl_eval in baseline_results.items():
                    bl_mae = bl_eval.get("mae", np.nan)
                    if np.isnan(bl_mae):
                        continue
                    if model_mae <= bl_mae:
                        worse_than_all = False
                        break
                if worse_than_all:
                    return ComponentStatus(
                        component_name="ml_signal",
                        status=ComponentStatusEnum.DIAGNOSTIC_ONLY,
                        reason="underperforms baselines",
                        details={
                            "model_mae": model_mae,
                            "baseline_maes": {
                                k: v.get("mae") for k, v in baseline_results.items()
                            },
                            "training_observations": training_obs,
                        },
                    )

        # --- Passed all checks → usable ---
        return ComponentStatus(
            component_name="ml_signal",
            status=ComponentStatusEnum.USABLE,
            reason="model trained and competitive with baselines",
            details={"training_observations": training_obs},
        )

    # ------------------------------------------------------------------
    # 7. Model audit
    # ------------------------------------------------------------------

    def generate_model_audit(
        self,
        model,
        baselines: dict,
        features: pd.DataFrame,
        walk_forward_results: Optional[dict] = None,
        target: Optional[pd.Series] = None,
        tree_model=None,
        tree_importances: Optional[pd.DataFrame] = None,
        tree_walk_forward: Optional[dict] = None,
        shap_importance: Optional[pd.DataFrame] = None,
    ) -> str:
        """Generate an honest model audit and save to outputs/model_audit.md.

        Includes:
        - Model type and hyper-parameters
        - Training observations count and feature count
        - Feature list with coefficient values
        - Walk-forward MAE vs each baseline
        - Honest assessment of predictive value
        - Annual-only caveat when training observations < 12
        """
        lines: list[str] = []
        lines.append("# Model Audit — ML Driver Model")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Target:** {self.config.ml_target}")
        lines.append(f"**Primary model:** {self.config.primary_model}")
        lines.append(f"**Min training window:** {self.config.min_train_years} periods")
        lines.append("")

        # --- Training summary ---
        n_features = features.shape[1] if features is not None and not features.empty else 0
        n_obs = len(target) if target is not None else 0

        lines.append("## Training Summary")
        lines.append("")
        lines.append(f"- **Training observations:** {n_obs}")
        lines.append(f"- **Feature count:** {n_features}")
        lines.append("")

        # --- Model coefficients ---
        lines.append("## Model Coefficients")
        lines.append("")
        if model is not None and hasattr(model, "coef_"):
            coef_dict = self.get_coefficient_summary(model)
            # If get_coefficient_summary returned indexed keys, use features columns
            if coef_dict and list(coef_dict.keys())[0].startswith("feature_"):
                coef_dict = dict(zip(features.columns, model.coef_))
            lines.append("| Feature | Coefficient |")
            lines.append("|---------|------------|")
            for feat, coef in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"| {feat} | {coef:.6f} |")
            if hasattr(model, "intercept_"):
                lines.append(f"| (intercept) | {model.intercept_:.6f} |")
        else:
            lines.append("Model not trained or coefficients unavailable.")
        lines.append("")

        # --- Walk-forward results ---
        lines.append("## Walk-Forward Validation Results")
        lines.append("")
        wf_eval: Optional[dict] = None
        if walk_forward_results and walk_forward_results.get("predictions"):
            wf_eval = self.evaluate(
                walk_forward_results["predictions"],
                walk_forward_results["actuals"],
            )
            lines.append(f"- **MAE:** {wf_eval['mae']}")
            lines.append(f"- **RMSE:** {wf_eval['rmse']}")
            lines.append(f"- **Directional accuracy:** {wf_eval['directional_accuracy']}")
            lines.append(f"- **Folds evaluated:** {wf_eval['n_evaluated']}")
        else:
            lines.append("Walk-forward validation not performed or insufficient data.")
        lines.append("")

        # --- Baseline comparison (walk-forward MAE vs each baseline) ---
        lines.append("## Baseline Comparison")
        lines.append("")
        if baselines and target is not None and len(target) > 0:
            lines.append("| Baseline | MAE | RMSE | Dir. Accuracy |")
            lines.append("|----------|-----|------|---------------|")
            actuals_list = target.tolist()
            for bl_name, bl_preds in baselines.items():
                bl_eval = self.evaluate(bl_preds, actuals_list)
                mae_str = f"{bl_eval['mae']:.6f}" if not np.isnan(bl_eval['mae']) else "N/A"
                rmse_str = f"{bl_eval['rmse']:.6f}" if not np.isnan(bl_eval['rmse']) else "N/A"
                da_str = f"{bl_eval['directional_accuracy']:.4f}" if not np.isnan(bl_eval.get('directional_accuracy', np.nan)) else "N/A"
                lines.append(f"| {bl_name} | {mae_str} | {rmse_str} | {da_str} |")

            # Add model row for direct comparison
            if wf_eval and not np.isnan(wf_eval.get("mae", np.nan)):
                model_mae_str = f"{wf_eval['mae']:.6f}"
                model_rmse_str = f"{wf_eval['rmse']:.6f}"
                model_da_str = f"{wf_eval['directional_accuracy']:.4f}" if not np.isnan(wf_eval.get('directional_accuracy', np.nan)) else "N/A"
                lines.append(f"| **ML model (walk-forward)** | **{model_mae_str}** | **{model_rmse_str}** | **{model_da_str}** |")
        else:
            lines.append("Baselines not computed or target unavailable.")
        lines.append("")

        # --- Tree model feature importances ---
        if tree_model is not None and tree_importances is not None and not tree_importances.empty:
            tree_type = type(tree_model).__name__
            lines.append(f"## Tree Model Feature Importances ({tree_type})")
            lines.append("")
            lines.append("| Feature | Importance |")
            lines.append("|---------|-----------|")
            for _, row in tree_importances.iterrows():
                lines.append(f"| {row['feature']} | {row['importance']:.6f} |")
            lines.append("")

        # --- Honest assessment of predictive value ---
        lines.append("## Predictive Value Assessment")
        lines.append("")

        model_mae_val = wf_eval["mae"] if wf_eval and not np.isnan(wf_eval.get("mae", np.nan)) else None
        best_baseline_mae = np.nan
        best_bl_name = ""
        if baselines and target is not None:
            for bl_name, bl_preds in baselines.items():
                bl_eval = self.evaluate(bl_preds, target.tolist())
                if not np.isnan(bl_eval["mae"]):
                    if np.isnan(best_baseline_mae) or bl_eval["mae"] < best_baseline_mae:
                        best_baseline_mae = bl_eval["mae"]
                        best_bl_name = bl_name

        if model_mae_val is not None and not np.isnan(best_baseline_mae):
            if model_mae_val > best_baseline_mae:
                lines.append(
                    f"**The ML model does NOT outperform the best baseline** ({best_bl_name}, "
                    f"MAE={best_baseline_mae:.6f} vs model MAE={model_mae_val:.6f}). "
                    "The model's value lies in feature organisation, scenario "
                    "discipline, and auditability rather than raw predictive power."
                )
            else:
                lines.append(
                    f"The ML model outperforms the best baseline ({best_bl_name}, "
                    f"MAE={best_baseline_mae:.6f}) with model MAE={model_mae_val:.6f}."
                )
        elif model_mae_val is None:
            lines.append(
                "Walk-forward validation was not performed or produced no results. "
                "Predictive value cannot be assessed."
            )
        lines.append("")

        # --- Limitations and Honest Disclosure ---
        lines.append("## Limitations and Honest Disclosure")
        lines.append("")

        lines.append(
            f"- **Observations:** {n_obs} periods available for training/validation."
        )
        lines.append(f"- **Features:** {n_features} features used.")

        if n_obs < 10:
            lines.append(
                "- **WARNING:** Very small sample size. Statistical conclusions "
                "are unreliable. Model results should be treated as exploratory."
            )

        # Annual-only caveat (Req 18.3): fewer than 12 training observations
        if n_obs < 12:
            lines.append(
                "- **Annual-only caveat:** ML is exploratory and underpowered "
                "due to limited annual observations. Walk-forward "
                "validation has very few folds. Results should not be over-interpreted."
            )

        lines.append("")
        lines.append("## Value of the ML Framework")
        lines.append("")
        lines.append(
            "Even when the model does not outperform simple baselines, the ML "
            "framework provides:\n"
            "1. **Feature organisation** — forces explicit enumeration of drivers.\n"
            "2. **Scenario discipline** — coefficients inform scenario assumptions.\n"
            "3. **Auditability** — every input is traceable to a filing or data source.\n"
            "4. **Point-in-time controls** — no-lookahead validation is built in."
        )
        lines.append("")

        audit_text = "\n".join(lines)

        # Append tree model audit section if tree model was trained
        if tree_model is not None:
            tree_audit = self.generate_tree_model_audit(
                tree_model=tree_model,
                tree_walk_forward=tree_walk_forward,
                baselines=baselines,
                features=features,
                target=target,
                shap_importance=shap_importance,
                builtin_importance=tree_importances,
                primary_walk_forward=walk_forward_results,
            )
            audit_text += "\n" + tree_audit

        # Save
        out_dir = Path(self.config.outputs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "model_audit.md"
        out_path.write_text(audit_text, encoding="utf-8")
        logger.info("Saved model audit to %s", out_path)

        return audit_text

    # ------------------------------------------------------------------
    # 8. Tree model (optional — RF / GBT)
    # ------------------------------------------------------------------

    def train_tree_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_type: str = "random_forest",
    ) -> tuple:
        """Train a RandomForestRegressor or GradientBoostingRegressor.

        Parameters
        ----------
        features : pd.DataFrame
            Numeric feature matrix.
        target : pd.Series
            Target values aligned with *features*.
        model_type : str
            ``"random_forest"`` or ``"gradient_boosting"``.

        Returns
        -------
        tuple[model, dict]
            Fitted tree model and dict mapping feature names to
            importance scores (from ``feature_importances_``).
        """
        valid = target.notna()
        X = features.loc[valid].copy().fillna(0.0)
        y = target.loc[valid].copy()

        if len(y) < 2:
            logger.warning("Insufficient data to train tree model (%d rows)", len(y))
            return None, {}

        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=42,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=None, random_state=42,
            )

        model.fit(X, y)

        importances = dict(zip(X.columns, model.feature_importances_))
        logger.info(
            "Trained %s tree model on %d observations, %d features",
            model_type, len(y), X.shape[1],
        )
        return model, importances

    def compute_feature_importances(
        self,
        model,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract ``feature_importances_`` from a fitted tree model.

        This serves as a lightweight alternative to SHAP values.

        Parameters
        ----------
        model
            A fitted sklearn tree-based model with ``feature_importances_``.
        features : pd.DataFrame
            The feature matrix used for training (column names needed).

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame (``feature``, ``importance``) sorted by
            descending importance.
        """
        if model is None or not hasattr(model, "feature_importances_"):
            logger.warning("Model has no feature_importances_ attribute")
            return pd.DataFrame(columns=["feature", "importance"])

        imp_df = pd.DataFrame({
            "feature": features.columns.tolist(),
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return imp_df

    # ------------------------------------------------------------------
    # 8b. SHAP feature importance (optional)
    # ------------------------------------------------------------------

    def compute_shap_values(
        self,
        model,
        features: pd.DataFrame,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Compute SHAP values for a fitted tree model.

        Uses ``shap.TreeExplainer`` for tree-based models.  If the
        ``shap`` library is not installed, logs a warning and returns
        (None, None).

        Parameters
        ----------
        model
            A fitted sklearn tree-based model (RandomForest or
            GradientBoosting).
        features : pd.DataFrame
            Numeric feature matrix used for training.

        Returns
        -------
        tuple[pd.DataFrame | None, pd.DataFrame | None]
            (shap_values_df, shap_importance_df).
            ``shap_values_df`` has the same shape as *features* with
            SHAP values per sample per feature.
            ``shap_importance_df`` has columns ``feature`` and
            ``mean_abs_shap``, sorted descending by importance.
            Both are None if SHAP is unavailable or model is None.
        """
        if model is None:
            logger.warning("No model provided for SHAP computation")
            return None, None

        try:
            import shap  # noqa: F811
        except ImportError:
            logger.warning(
                "shap library not installed — skipping SHAP analysis. "
                "Install with: pip install shap"
            )
            return None, None

        # Prepare features: drop NaN rows, fill remaining NaN
        valid = features.notna().all(axis=1)
        X = features.loc[valid].copy().fillna(0.0)

        if X.empty:
            logger.warning("No valid rows for SHAP computation")
            return None, None

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # shap_values is a numpy array of shape (n_samples, n_features)
            shap_df = pd.DataFrame(
                shap_values,
                columns=X.columns,
                index=X.index,
            )

            # Mean absolute SHAP value per feature → importance ranking
            mean_abs = shap_df.abs().mean().sort_values(ascending=False)
            importance_df = pd.DataFrame({
                "feature": mean_abs.index,
                "mean_abs_shap": mean_abs.values,
            }).reset_index(drop=True)

            logger.info(
                "SHAP values computed for %d samples, %d features",
                len(shap_df), len(importance_df),
            )
            return shap_df, importance_df

        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # 8c. Walk-forward validation for tree models
    # ------------------------------------------------------------------

    def walk_forward_validate_tree(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_type: str = "gradient_boosting",
    ) -> dict:
        """Expanding-window walk-forward validation for tree models.

        Same protocol as ``walk_forward_validate`` but uses
        RandomForest or GradientBoosting instead of Ridge/ElasticNet.

        Parameters
        ----------
        features : pd.DataFrame
            Numeric feature matrix.
        target : pd.Series
            Target values aligned with *features*.
        model_type : str
            ``"random_forest"`` or ``"gradient_boosting"``.

        Returns
        -------
        dict
            ``{"predictions": [...], "actuals": [...], "folds": [...]}``.
        """
        min_train = self.config.min_train_years
        valid = target.notna()
        X = features.loc[valid].copy().fillna(0.0)
        y = target.loc[valid].copy()

        n = len(y)

        if n < min_train + 1:
            logger.warning(
                "Insufficient data for tree walk-forward validation: %d rows, "
                "need at least %d",
                n, min_train + 1,
            )
            return {"predictions": [], "actuals": [], "folds": []}

        predictions: list[float] = []
        actuals: list[float] = []
        folds: list[dict] = []

        for i in range(min_train, n):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[[i]]
            y_test = y.iloc[i]

            if model_type == "gradient_boosting":
                tree = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, random_state=42,
                )
            else:
                tree = RandomForestRegressor(
                    n_estimators=100, max_depth=None, random_state=42,
                )

            tree.fit(X_train, y_train)
            pred = float(tree.predict(X_test)[0])

            predictions.append(pred)
            actuals.append(float(y_test))
            folds.append({
                "train_size": i,
                "test_index": int(y.index[i]),
                "predicted": pred,
                "actual": float(y_test),
            })

        logger.info("Tree walk-forward validation (%s): %d folds", model_type, len(folds))
        return {"predictions": predictions, "actuals": actuals, "folds": folds}

    # ------------------------------------------------------------------
    # 8d. Tree model audit
    # ------------------------------------------------------------------

    def generate_tree_model_audit(
        self,
        tree_model,
        tree_walk_forward: Optional[dict],
        baselines: dict,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        shap_importance: Optional[pd.DataFrame] = None,
        builtin_importance: Optional[pd.DataFrame] = None,
        primary_walk_forward: Optional[dict] = None,
    ) -> str:
        """Generate a supplementary audit section for the tree model.

        Includes: model type, hyperparameters, walk-forward MAE vs
        baselines, feature importance (SHAP or built-in), and comparison
        with the primary Ridge/ElasticNet model.

        Parameters
        ----------
        tree_model
            Fitted tree model.
        tree_walk_forward : dict | None
            Walk-forward results from ``walk_forward_validate_tree``.
        baselines : dict
            Baseline predictions from ``compute_baselines``.
        features : pd.DataFrame
            Feature matrix.
        target : pd.Series | None
            Target values.
        shap_importance : pd.DataFrame | None
            SHAP-based importance (``feature``, ``mean_abs_shap``).
        builtin_importance : pd.DataFrame | None
            Built-in importance (``feature``, ``importance``).
        primary_walk_forward : dict | None
            Walk-forward results from the primary model for comparison.

        Returns
        -------
        str
            Markdown text for the tree model audit section.
        """
        lines: list[str] = []
        lines.append("")
        lines.append("## Tree Model Audit (Supplementary)")
        lines.append("")

        if tree_model is None:
            lines.append("Tree model was not trained.")
            return "\n".join(lines)

        tree_type = type(tree_model).__name__
        lines.append(f"**Model type:** {tree_type}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Hyperparameters
        lines.append("### Hyperparameters")
        lines.append("")
        params = tree_model.get_params()
        key_params = ["n_estimators", "max_depth", "learning_rate", "min_samples_split",
                       "min_samples_leaf", "random_state"]
        for p in key_params:
            if p in params:
                lines.append(f"- **{p}:** {params[p]}")
        lines.append("")

        # Walk-forward results
        lines.append("### Walk-Forward Validation")
        lines.append("")
        tree_eval: Optional[dict] = None
        if tree_walk_forward and tree_walk_forward.get("predictions"):
            tree_eval = self.evaluate(
                tree_walk_forward["predictions"],
                tree_walk_forward["actuals"],
            )
            lines.append(f"- **MAE:** {tree_eval['mae']}")
            lines.append(f"- **RMSE:** {tree_eval['rmse']}")
            lines.append(f"- **Directional accuracy:** {tree_eval['directional_accuracy']}")
            lines.append(f"- **Folds evaluated:** {tree_eval['n_evaluated']}")
        else:
            lines.append("Walk-forward validation not performed or insufficient data.")
        lines.append("")

        # Baseline comparison
        lines.append("### Baseline Comparison")
        lines.append("")
        if baselines and target is not None and len(target) > 0:
            lines.append("| Model/Baseline | MAE | RMSE |")
            lines.append("|----------------|-----|------|")
            actuals_list = target.tolist()
            for bl_name, bl_preds in baselines.items():
                bl_eval = self.evaluate(bl_preds, actuals_list)
                mae_str = f"{bl_eval['mae']:.6f}" if not np.isnan(bl_eval['mae']) else "N/A"
                rmse_str = f"{bl_eval['rmse']:.6f}" if not np.isnan(bl_eval['rmse']) else "N/A"
                lines.append(f"| {bl_name} | {mae_str} | {rmse_str} |")
            if tree_eval and not np.isnan(tree_eval.get("mae", np.nan)):
                lines.append(
                    f"| **{tree_type} (walk-forward)** | "
                    f"**{tree_eval['mae']:.6f}** | **{tree_eval['rmse']:.6f}** |"
                )
        else:
            lines.append("Baselines not computed or target unavailable.")
        lines.append("")

        # Comparison with primary model
        lines.append("### Comparison with Primary Model")
        lines.append("")
        primary_eval: Optional[dict] = None
        if primary_walk_forward and primary_walk_forward.get("predictions"):
            primary_eval = self.evaluate(
                primary_walk_forward["predictions"],
                primary_walk_forward["actuals"],
            )
        if primary_eval and tree_eval:
            p_mae = primary_eval.get("mae", np.nan)
            t_mae = tree_eval.get("mae", np.nan)
            if not np.isnan(p_mae) and not np.isnan(t_mae):
                lines.append(f"- **Primary model MAE:** {p_mae:.6f}")
                lines.append(f"- **Tree model MAE:** {t_mae:.6f}")
                if t_mae < p_mae:
                    lines.append(
                        f"- Tree model outperforms primary by "
                        f"{((p_mae - t_mae) / p_mae * 100):.1f}%."
                    )
                elif t_mae > p_mae:
                    lines.append(
                        f"- Primary model outperforms tree model by "
                        f"{((t_mae - p_mae) / t_mae * 100):.1f}%."
                    )
                else:
                    lines.append("- Both models have identical MAE.")
            else:
                lines.append("Insufficient data for model comparison.")
        else:
            lines.append("Primary model walk-forward results not available for comparison.")
        lines.append("")

        # Feature importance
        lines.append("### Feature Importance")
        lines.append("")
        if shap_importance is not None and not shap_importance.empty:
            lines.append("**Method:** SHAP (TreeExplainer)")
            lines.append("")
            lines.append("| Feature | Mean Abs SHAP |")
            lines.append("|---------|---------------|")
            for _, row in shap_importance.head(15).iterrows():
                lines.append(f"| {row['feature']} | {row['mean_abs_shap']:.6f} |")
        elif builtin_importance is not None and not builtin_importance.empty:
            lines.append("**Method:** Built-in feature importance (Gini/variance reduction)")
            lines.append("")
            lines.append("*Note: SHAP library not available. Using built-in importances.*")
            lines.append("")
            lines.append("| Feature | Importance |")
            lines.append("|---------|-----------|")
            for _, row in builtin_importance.head(15).iterrows():
                lines.append(f"| {row['feature']} | {row['importance']:.6f} |")
        else:
            lines.append("Feature importance not available.")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pivot_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Pivot long-format metrics into wide format keyed by fiscal_period."""
        if metrics.empty:
            return pd.DataFrame()

        # Keep only rows with non-null metric_value
        df = metrics.dropna(subset=["metric_value"]).copy()

        # Pivot: one row per fiscal_period, one column per metric_name
        pivoted = df.pivot_table(
            index="fiscal_period",
            columns="metric_name",
            values="metric_value",
            aggfunc="first",
        ).reset_index()

        # Attach metadata (filing_date, source_available_date, source_accession)
        # from the first row of each period
        meta = (
            df.groupby("fiscal_period")[
                ["filing_date", "source_available_date", "source_accession"]
            ]
            .first()
            .reset_index()
        )
        pivoted = pivoted.merge(meta, on="fiscal_period", how="left")

        return pivoted

    def _pivot_nlp(
        self, nlp: pd.DataFrame, metrics: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Pivot NLP features into wide format, aligned to fiscal_period.

        NLP data is keyed by filing_date; we map each NLP row to the
        closest metrics fiscal_period by matching on filing_date.
        """
        if nlp.empty:
            return None

        # Build filing_date → fiscal_period mapping from metrics
        if metrics.empty:
            return None

        date_to_period = (
            metrics.groupby("filing_date")["fiscal_period"]
            .first()
            .to_dict()
        )

        nlp_copy = nlp.copy()
        nlp_copy["fiscal_period"] = nlp_copy["filing_date"].map(date_to_period)
        nlp_copy = nlp_copy.dropna(subset=["fiscal_period"])

        if nlp_copy.empty:
            return None

        # Create a unique feature name: section_featuretype_featurename
        nlp_copy["feature_col"] = (
            "nlp_" + nlp_copy["section"].astype(str) + "_"
            + nlp_copy["feature_name"].astype(str)
        )

        pivoted = nlp_copy.pivot_table(
            index="fiscal_period",
            columns="feature_col",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Attach metadata
        meta = (
            nlp_copy.groupby("fiscal_period")[
                ["filing_date", "source_available_date", "source_accession"]
            ]
            .first()
            .reset_index()
        )
        pivoted = pivoted.merge(meta, on="fiscal_period", how="left")

        return pivoted

    @staticmethod
    def _resolve_target_column(
        feature_df: pd.DataFrame, target_name: str,
    ) -> Optional[str]:
        """Find the column in feature_df that best matches the target name."""
        # Direct match
        if target_name in feature_df.columns:
            return target_name

        # Case-insensitive search
        lower_map = {c.lower(): c for c in feature_df.columns}
        if target_name.lower() in lower_map:
            return lower_map[target_name.lower()]

        # Partial match (e.g. "revenue_growth_yoy" → "revenue_growth_YoY")
        for col in feature_df.columns:
            if target_name.lower().replace("_", "") in col.lower().replace("_", ""):
                return col

        return None

    # ------------------------------------------------------------------
    # 9. Semiconductor peer-panel ML (optional — 17.5)
    # ------------------------------------------------------------------

    # Minimum years of peer history required for inclusion
    PEER_MIN_HISTORY_YEARS: int = 10

    def build_peer_panel_dataset(
        self,
        nvda_features: pd.DataFrame,
        nvda_target: pd.Series,
        peer_data: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
        """Construct a panel dataset combining NVDA and peer company data.

        Filters peers by history length (≥ ``PEER_MIN_HISTORY_YEARS``
        rows), identifies common feature columns, and concatenates into
        a single pooled DataFrame.

        Parameters
        ----------
        nvda_features : pd.DataFrame
            NVDA feature matrix (rows = periods, cols = features).
        nvda_target : pd.Series
            NVDA target values aligned with *nvda_features*.
        peer_data : pd.DataFrame, optional
            Peer feature/target data with columns matching NVDA features
            plus ``ticker`` and ``target_value`` columns.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, dict]
            (pooled_features, pooled_target, peer_info) where
            ``peer_info`` maps ticker → {"rows": int, "included": bool,
            "reason": str}.
        """
        peer_info: dict[str, dict] = {}

        if peer_data is None or peer_data.empty:
            logger.warning(
                "Peer-panel ML requires 10-year peer historicals. "
                "No peer data provided — returning NVDA-only dataset."
            )
            return nvda_features.copy(), nvda_target.copy(), peer_info

        if "target_value" not in peer_data.columns:
            logger.warning("Peer data missing 'target_value' column.")
            return nvda_features.copy(), nvda_target.copy(), peer_info

        # Identify common feature columns
        meta_cols = {
            "ticker", "target_value", "fiscal_period", "filing_date",
            "source_available_date", "source_accession",
            "feature_period", "feature_available_date", "prediction_date",
            "target_period", "target_available_date", "target_name",
            "source_accessions",
        }
        nvda_feature_cols = set(nvda_features.columns)
        peer_feature_cols = set(peer_data.columns) - meta_cols
        common_features = sorted(nvda_feature_cols & peer_feature_cols)

        if not common_features:
            logger.warning(
                "No common features between NVDA and peer data — "
                "returning NVDA-only dataset."
            )
            return nvda_features.copy(), nvda_target.copy(), peer_info

        # Filter peers by history length
        included_peer_rows: list[pd.DataFrame] = []
        if "ticker" in peer_data.columns:
            for ticker, group in peer_data.groupby("ticker"):
                n_rows = len(group)
                has_target = group["target_value"].notna().sum()
                all_nan = group[common_features].isna().all().all()

                if all_nan:
                    peer_info[str(ticker)] = {
                        "rows": n_rows,
                        "included": False,
                        "reason": "all feature data is NaN",
                    }
                    logger.info(
                        "Skipping peer %s: all feature data is NaN", ticker
                    )
                elif n_rows < self.PEER_MIN_HISTORY_YEARS:
                    peer_info[str(ticker)] = {
                        "rows": n_rows,
                        "included": False,
                        "reason": f"insufficient history ({n_rows} rows, need {self.PEER_MIN_HISTORY_YEARS})",
                    }
                    logger.info(
                        "Skipping peer %s: insufficient history (%d rows, "
                        "need %d)",
                        ticker, n_rows, self.PEER_MIN_HISTORY_YEARS,
                    )
                elif has_target == 0:
                    peer_info[str(ticker)] = {
                        "rows": n_rows,
                        "included": False,
                        "reason": "no valid target values",
                    }
                    logger.info(
                        "Skipping peer %s: no valid target values", ticker
                    )
                else:
                    peer_info[str(ticker)] = {
                        "rows": n_rows,
                        "included": True,
                        "reason": "included",
                    }
                    included_peer_rows.append(
                        group[common_features + ["target_value"]].copy()
                    )
        else:
            # No ticker column — treat all peer data as one block
            n_rows = len(peer_data)
            if n_rows >= self.PEER_MIN_HISTORY_YEARS:
                included_peer_rows.append(
                    peer_data[common_features + ["target_value"]].copy()
                )
                peer_info["unknown"] = {
                    "rows": n_rows,
                    "included": True,
                    "reason": "included (no ticker column)",
                }
            else:
                peer_info["unknown"] = {
                    "rows": n_rows,
                    "included": False,
                    "reason": f"insufficient history ({n_rows} rows)",
                }

        # Build pooled dataset
        nvda_panel = nvda_features[common_features].copy()
        nvda_panel["target_value"] = nvda_target.values

        if included_peer_rows:
            peer_panel = pd.concat(included_peer_rows, ignore_index=True)
            pooled = pd.concat([nvda_panel, peer_panel], ignore_index=True)
        else:
            pooled = nvda_panel.copy()

        pooled_target = pooled.pop("target_value")

        logger.info(
            "Panel dataset: %d total rows (%d NVDA + %d peer), "
            "%d features, %d peers included, %d peers skipped",
            len(pooled),
            len(nvda_panel),
            sum(len(r) for r in included_peer_rows),
            len(common_features),
            sum(1 for v in peer_info.values() if v["included"]),
            sum(1 for v in peer_info.values() if not v["included"]),
        )

        return pooled, pooled_target, peer_info

    def train_peer_panel_model(
        self,
        nvda_features: pd.DataFrame,
        nvda_target: pd.Series,
        peer_data: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Train a panel model pooling NVDA + semiconductor peer features.

        This method enables cross-sectional learning from peer companies
        (AMD, AVGO, INTC, QCOM, MRVL) to improve NVDA revenue growth
        forecasting.

        **Limitation:** Requires 10-year peer historicals in the same
        feature format as NVDA.  If ``peer_data`` is None or empty, the
        method logs a warning and returns (None, {}).

        Parameters
        ----------
        nvda_features : pd.DataFrame
            NVDA feature matrix (rows = periods, cols = features).
        nvda_target : pd.Series
            NVDA target values aligned with *nvda_features*.
        peer_data : pd.DataFrame, optional
            Peer feature/target data with columns matching NVDA features
            plus a ``ticker`` column and a ``target_value`` column.
            Each row represents one peer-period observation.

        Returns
        -------
        tuple[model | None, dict]
            Fitted panel model and coefficient dict.  Returns (None, {})
            if peer data is unavailable.
        """
        if peer_data is None or peer_data.empty:
            logger.warning(
                "Peer-panel ML requires 10-year peer historicals in the same "
                "feature format as NVDA. No peer data provided — skipping "
                "peer-panel model. To enable, supply a DataFrame with columns "
                "matching NVDA features plus 'ticker' and 'target_value'."
            )
            return None, {}

        # Validate peer data has required columns
        if "target_value" not in peer_data.columns:
            logger.warning("Peer data missing 'target_value' column — skipping peer-panel model.")
            return None, {}

        # Identify common feature columns
        meta_cols = {"ticker", "target_value", "fiscal_period", "filing_date",
                     "source_available_date", "source_accession",
                     "feature_period", "feature_available_date", "prediction_date",
                     "target_period", "target_available_date", "target_name",
                     "source_accessions"}
        nvda_feature_cols = set(nvda_features.columns)
        peer_feature_cols = set(peer_data.columns) - meta_cols
        common_features = sorted(nvda_feature_cols & peer_feature_cols)

        if not common_features:
            logger.warning("No common features between NVDA and peer data — skipping peer-panel model.")
            return None, {}

        # Build pooled dataset: NVDA + peers
        nvda_panel = nvda_features[common_features].copy()
        nvda_panel["target_value"] = nvda_target.values

        peer_panel = peer_data[common_features + ["target_value"]].copy()

        pooled = pd.concat([nvda_panel, peer_panel], ignore_index=True)
        pooled_target = pooled.pop("target_value")

        # Drop rows with NaN target
        valid = pooled_target.notna()
        X = pooled.loc[valid].fillna(0.0)
        y = pooled_target.loc[valid]

        if len(y) < 5:
            logger.warning("Pooled panel has only %d valid rows — insufficient for training.", len(y))
            return None, {}

        # Train ElasticNet on pooled data
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        model.fit(X, y)

        coefficients = dict(zip(X.columns, model.coef_))
        logger.info(
            "Trained peer-panel ElasticNet on %d pooled observations (%d features), "
            "including %d peer rows.",
            len(y), X.shape[1], len(peer_panel),
        )
        return model, coefficients

    def walk_forward_validate_panel(
        self,
        pooled_features: pd.DataFrame,
        pooled_target: pd.Series,
    ) -> dict:
        """Walk-forward validation on the pooled panel dataset.

        Uses an expanding window respecting time ordering.  The panel
        rows must already be sorted chronologically (NVDA and peer rows
        interleaved by period).

        Parameters
        ----------
        pooled_features : pd.DataFrame
            Pooled feature matrix from ``build_peer_panel_dataset``.
        pooled_target : pd.Series
            Pooled target values aligned with *pooled_features*.

        Returns
        -------
        dict
            ``{"predictions": [...], "actuals": [...], "folds": [...]}``.
        """
        min_train = max(self.config.min_train_years, 5)
        valid = pooled_target.notna()
        X = pooled_features.loc[valid].copy().fillna(0.0)
        y = pooled_target.loc[valid].copy()

        n = len(y)

        if n < min_train + 1:
            logger.warning(
                "Insufficient panel data for walk-forward validation: "
                "%d rows, need at least %d",
                n, min_train + 1,
            )
            return {"predictions": [], "actuals": [], "folds": []}

        predictions: list[float] = []
        actuals: list[float] = []
        folds: list[dict] = []

        for i in range(min_train, n):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[[i]]
            y_test = y.iloc[i]

            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
            model.fit(X_train, y_train)
            pred = float(model.predict(X_test)[0])

            predictions.append(pred)
            actuals.append(float(y_test))
            folds.append({
                "train_size": i,
                "test_index": int(y.index[i]),
                "predicted": pred,
                "actual": float(y_test),
            })

        logger.info("Panel walk-forward validation: %d folds", len(folds))
        return {"predictions": predictions, "actuals": actuals, "folds": folds}

    def generate_peer_panel_audit(
        self,
        panel_model,
        panel_coefficients: dict,
        panel_walk_forward: dict,
        peer_info: dict,
        nvda_obs: int,
        baselines: Optional[dict] = None,
        target: Optional[pd.Series] = None,
    ) -> str:
        """Generate an audit report section for the peer-panel model.

        Parameters
        ----------
        panel_model
            Fitted peer-panel model (or None).
        panel_coefficients : dict
            Feature → coefficient mapping.
        panel_walk_forward : dict
            Walk-forward results from ``walk_forward_validate_panel``.
        peer_info : dict
            Peer inclusion/exclusion info from ``build_peer_panel_dataset``.
        nvda_obs : int
            Number of NVDA-only observations.
        baselines : dict, optional
            Baseline predictions for comparison.
        target : pd.Series, optional
            Target values for baseline evaluation.

        Returns
        -------
        str
            Markdown text for the peer-panel audit section.
        """
        lines: list[str] = []
        lines.append("")
        lines.append("## Peer-Panel ML Model Audit (Supplementary — Task 17.5)")
        lines.append("")
        lines.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("")

        if panel_model is None:
            lines.append(
                "Peer-panel model was **not trained**. This is expected when "
                "10-year peer historicals are unavailable in the required "
                "feature format."
            )
            lines.append("")
            if peer_info:
                lines.append("### Peer Data Availability")
                lines.append("")
                lines.append("| Peer | Rows | Included | Reason |")
                lines.append("|------|------|----------|--------|")
                for ticker, info in sorted(peer_info.items()):
                    inc = "✓" if info["included"] else "✗"
                    lines.append(
                        f"| {ticker} | {info['rows']} | {inc} | "
                        f"{info['reason']} |"
                    )
                lines.append("")
            return "\n".join(lines)

        # Model was trained
        total_peer_rows = sum(
            v["rows"] for v in peer_info.values() if v["included"]
        )
        lines.append(f"**Model type:** ElasticNet (panel)")
        lines.append(
            f"**Pooled observations:** {nvda_obs + total_peer_rows} "
            f"({nvda_obs} NVDA + {total_peer_rows} peer)"
        )
        lines.append(f"**Features:** {len(panel_coefficients)}")
        lines.append("")

        # Peer inclusion table
        lines.append("### Peer Data Availability")
        lines.append("")
        lines.append("| Peer | Rows | Included | Reason |")
        lines.append("|------|------|----------|--------|")
        for ticker, info in sorted(peer_info.items()):
            inc = "✓" if info["included"] else "✗"
            lines.append(
                f"| {ticker} | {info['rows']} | {inc} | "
                f"{info['reason']} |"
            )
        lines.append("")

        # Coefficients
        lines.append("### Panel Model Coefficients")
        lines.append("")
        lines.append("| Feature | Coefficient |")
        lines.append("|---------|------------|")
        for feat, coef in sorted(
            panel_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            lines.append(f"| {feat} | {coef:.6f} |")
        if hasattr(panel_model, "intercept_"):
            lines.append(
                f"| (intercept) | {panel_model.intercept_:.6f} |"
            )
        lines.append("")

        # Walk-forward results
        lines.append("### Panel Walk-Forward Validation")
        lines.append("")
        if panel_walk_forward and panel_walk_forward.get("predictions"):
            wf_eval = self.evaluate(
                panel_walk_forward["predictions"],
                panel_walk_forward["actuals"],
            )
            lines.append(f"- **MAE:** {wf_eval['mae']}")
            lines.append(f"- **RMSE:** {wf_eval['rmse']}")
            lines.append(
                f"- **Directional accuracy:** "
                f"{wf_eval['directional_accuracy']}"
            )
            lines.append(f"- **Folds evaluated:** {wf_eval['n_evaluated']}")
        else:
            lines.append(
                "Walk-forward validation not performed or insufficient data."
            )
        lines.append("")

        # Limitations
        lines.append("### Limitations")
        lines.append("")
        lines.append(
            "- Cross-sectional pooling assumes peer revenue growth "
            "dynamics are informative for NVDA forecasting."
        )
        lines.append(
            "- Peer data may not span the full 10-year window; "
            "peers with insufficient history are excluded."
        )
        lines.append(
            "- Panel model is supplementary — the primary "
            "Ridge/ElasticNet model remains the main signal."
        )
        skipped = [
            t for t, v in peer_info.items() if not v["included"]
        ]
        if skipped:
            lines.append(
                f"- **Skipped peers:** {', '.join(skipped)}"
            )
        lines.append("")

        return "\n".join(lines)
