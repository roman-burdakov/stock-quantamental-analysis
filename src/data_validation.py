"""
NVDA Quantamental Engine — Data Validation Gate.

Hard gate between XBRL parsing and downstream pipeline stages.
Cross-checks parsed financial data against published reference values,
assigns severity and blocker status, and determines whether the pipeline
may issue a formal recommendation.

Requirements: 14.1–14.5, 14.9–14.12
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.config import (
    DataQualityIssue,
    DataQualityStatus,
    EngineConfig,
    ValidatedMetric,
)

logger = logging.getLogger(__name__)

# Metrics whose failure blocks a formal recommendation (valuation-critical).
_VALUATION_CRITICAL_METRICS = frozenset({
    "revenue",
    "operating_cash_flow",
    "capex",
    "diluted_shares",
    "cash_and_securities",
    "total_debt",
})


class DataValidationGate:
    """Hard gate between XBRL parsing and downstream pipeline stages.

    Validates parsed financial data against published reference values,
    computes diff_pct and severity for each metric, and determines the
    overall :class:`DataQualityStatus`.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.core_metrics: list[str] = list(config.core_validation_metrics)
        self.tolerance: float = config.validation_tolerance_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_parsed_data(
        self,
        parsed_df: pd.DataFrame,
        published_values: dict,
    ) -> tuple[DataQualityStatus, list[ValidatedMetric]]:
        """Run validation for all core metrics across all fiscal years.

        For each core metric and fiscal year present in *published_values*,
        find the corresponding parsed value in *parsed_df*, compute the
        percentage difference, and assign severity / blocker status.

        Parameters
        ----------
        parsed_df:
            DataFrame output from :class:`XBRLParser.parse_companyfacts`.
            Expected columns include ``metric_name``, ``fiscal_year``,
            ``fiscal_period``, and ``value``.
        published_values:
            Dict matching the ``tests/fixtures/nvda_published_values.json``
            schema.  Top-level key ``"values"`` maps fiscal-year labels
            (e.g. ``"FY2025"``) to ``{"metrics": {metric_name: {"value": ...}}}``.

        Returns
        -------
        tuple[DataQualityStatus, list[ValidatedMetric]]
            Overall pipeline status and per-metric validation results.
        """
        validated: list[ValidatedMetric] = []
        values_by_fy = published_values.get("values", {})

        for fy_label, fy_data in values_by_fy.items():
            fy_num = self._parse_fy_number(fy_label)
            if fy_num is None:
                continue

            metrics_dict = fy_data.get("metrics", {})

            for metric_name in self.core_metrics:
                published_entry = self._resolve_published_entry(
                    metric_name, metrics_dict,
                )
                published_value = (
                    published_entry.get("value") if published_entry else None
                )

                # Look up parsed value (annual / FY period)
                parsed_value = self._lookup_parsed_value(
                    parsed_df, metric_name, fy_num,
                )

                # Determine severity and blocker
                severity = self._assign_severity(metric_name)
                is_blocker = severity == "critical"

                # Compute diff_pct and status
                diff_pct: float | None = None
                status: str

                if parsed_value is None and published_value is None:
                    status = "missing"
                elif parsed_value is None:
                    status = "missing"
                elif published_value is None:
                    # Req 14.11: missing published value for valuation-critical
                    # base-year metric → DATA_BLOCKED unless manually sourced.
                    status = "missing"
                elif published_value == 0:
                    # Avoid division by zero; treat as fail if parsed != 0
                    diff_pct = 0.0 if parsed_value == 0 else 100.0
                    status = "pass" if diff_pct <= self.tolerance else "fail"
                else:
                    diff_pct = (
                        abs(parsed_value - published_value)
                        / abs(published_value)
                        * 100
                    )
                    status = "pass" if diff_pct <= self.tolerance else "fail"

                validated.append(
                    ValidatedMetric(
                        metric_name=metric_name,
                        fiscal_year=fy_num,
                        parsed_value=parsed_value,
                        published_value=published_value,
                        diff_pct=diff_pct,
                        tolerance=self.tolerance,
                        status=status,
                        severity=severity,
                        blocker=is_blocker,
                    )
                )

        # --- Req 14.11: missing published value for valuation-critical ---
        # base-year metric → DATA_BLOCKED unless manually sourced.
        # We check the latest FY in published_values for valuation-critical
        # metrics that have no published reference at all.
        self._check_missing_valuation_critical(validated, values_by_fy)

        # --- Req 14.12: validation_coverage_pct check ---
        status = self._compute_overall_status(validated, values_by_fy)

        return status, validated

    def check_fiscal_year_selection(
        self, parsed_df: pd.DataFrame,
    ) -> list[DataQualityIssue]:
        """Verify no quarterly/YTD values were selected as annual.

        Flags any fact where ``fiscal_period == "FY"`` but
        ``fiscal_period_type`` is not ``"annual"`` or ``"instant"``.

        Parameters
        ----------
        parsed_df:
            DataFrame from :class:`XBRLParser.parse_companyfacts`.

        Returns
        -------
        list[DataQualityIssue]
            Issues found (empty list if all clean).
        """
        issues: list[DataQualityIssue] = []

        if parsed_df.empty:
            return issues

        # Only inspect rows tagged as full-year (FY)
        fy_rows = parsed_df[parsed_df["fiscal_period"] == "FY"]

        for _, row in fy_rows.iterrows():
            period_type = row.get("fiscal_period_type", "")
            if period_type in ("annual", "instant", ""):
                continue  # acceptable

            metric = row.get("metric_name", "unknown")
            fy = row.get("fiscal_year", "?")
            duration = row.get("duration_days")

            issues.append(
                DataQualityIssue(
                    category="fiscal_year_mismatch",
                    detail=(
                        f"Metric '{metric}' for FY{fy} has "
                        f"fiscal_period_type='{period_type}' "
                        f"(duration={duration} days) but is tagged as FY. "
                        f"Expected annual or instant."
                    ),
                    filing_or_source=str(row.get("accession_number", "")),
                    severity="critical",
                )
            )

        if issues:
            logger.warning(
                "Fiscal-year selection issues found: %d facts with "
                "non-annual period types tagged as FY",
                len(issues),
            )

        return issues

    def generate_validation_summary(
        self,
        status: DataQualityStatus,
        metrics: list[ValidatedMetric],
    ) -> dict:
        """Machine-readable summary for ``audit_status.json``.

        Parameters
        ----------
        status:
            Overall :class:`DataQualityStatus`.
        metrics:
            List of :class:`ValidatedMetric` from validation.

        Returns
        -------
        dict
            Summary with keys: overall_status, metrics_validated,
            metrics_passed, metrics_failed, metrics_missing,
            validation_coverage_pct, blocking_issues, details.
        """
        total = len(metrics)
        passed = sum(1 for m in metrics if m.status == "pass")
        failed = sum(1 for m in metrics if m.status == "fail")
        missing = sum(1 for m in metrics if m.status == "missing")

        # Coverage = (pass + warn) / total — warn maps to pass here
        coverage_pct = (passed / total * 100) if total > 0 else 0.0

        blocking_issues: list[str] = []
        for m in metrics:
            if m.severity == "critical" and m.blocker and m.status == "fail":
                blocking_issues.append(
                    f"{m.metric_name} FY{m.fiscal_year}: "
                    f"parsed={m.parsed_value}, published={m.published_value}, "
                    f"diff={m.diff_pct:.2f}%"
                    if m.diff_pct is not None
                    else f"{m.metric_name} FY{m.fiscal_year}: missing"
                )
            elif m.severity == "critical" and m.blocker and m.status == "missing":
                blocking_issues.append(
                    f"{m.metric_name} FY{m.fiscal_year}: missing value"
                )

        details: list[dict] = []
        for m in metrics:
            details.append({
                "metric_name": m.metric_name,
                "fiscal_year": m.fiscal_year,
                "parsed_value": m.parsed_value,
                "published_value": m.published_value,
                "diff_pct": round(m.diff_pct, 4) if m.diff_pct is not None else None,
                "tolerance": m.tolerance,
                "status": m.status,
                "severity": m.severity,
                "blocker": m.blocker,
            })

        return {
            "overall_status": status.value,
            "metrics_validated": total,
            "metrics_passed": passed,
            "metrics_failed": failed,
            "metrics_missing": missing,
            "validation_coverage_pct": round(coverage_pct, 2),
            "blocking_issues": blocking_issues,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "details": details,
        }

    def should_block_recommendation(self, status: DataQualityStatus) -> bool:
        """Return ``True`` if *status* is :attr:`DataQualityStatus.DATA_BLOCKED`.

        Parameters
        ----------
        status:
            The overall data quality status from validation.
        """
        return status is DataQualityStatus.DATA_BLOCKED

    def get_diagnostic_label(self, status: DataQualityStatus) -> str:
        """Return the appropriate label for valuation outputs.

        Returns ``"Diagnostic only — do not use for recommendation"`` when
        *status* is :attr:`DataQualityStatus.DATA_BLOCKED`, otherwise ``""``.
        """
        if status is DataQualityStatus.DATA_BLOCKED:
            return "Diagnostic only — do not use for recommendation"
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fy_number(fy_label: str) -> int | None:
        """Extract the fiscal year number from a label like ``"FY2025"``."""
        try:
            return int(fy_label.replace("FY", ""))
        except (ValueError, AttributeError):
            return None

    def _assign_severity(self, metric_name: str) -> str:
        """Assign severity for a core metric.

        All core annual metrics get ``"critical"`` severity (Req 14.4).
        """
        if metric_name in self.core_metrics:
            return "critical"
        return "major"

    def _resolve_published_entry(
        self,
        metric_name: str,
        metrics_dict: dict,
    ) -> dict | None:
        """Resolve a core metric name to its published_values entry.

        Handles mapping differences between our internal metric names
        and the keys used in the published_values fixture.
        """
        # Direct match
        if metric_name in metrics_dict:
            return metrics_dict[metric_name]

        # Mapping from our core_validation_metrics names to published keys
        alias_map: dict[str, list[str]] = {
            "FCF": ["free_cash_flow", "FCF"],
            "free_cash_flow": ["free_cash_flow", "FCF"],
            "cash_and_securities": [
                "cash_and_short_term_investments",
                "cash_and_securities",
                "cash_and_equivalents",
            ],
            "diluted_EPS": ["diluted_eps", "diluted_EPS"],
            "diluted_eps": ["diluted_eps", "diluted_EPS"],
            "r_and_d": ["r_and_d", "R&D", "research_and_development"],
            "R&D": ["r_and_d", "R&D", "research_and_development"],
        }

        candidates = alias_map.get(metric_name, [])
        for alias in candidates:
            if alias in metrics_dict:
                return metrics_dict[alias]

        return None

    def _lookup_parsed_value(
        self,
        parsed_df: pd.DataFrame,
        metric_name: str,
        fy_num: int,
    ) -> float | None:
        """Find the parsed value for a metric + fiscal year.

        Looks for annual (FY) period rows first.  Also tries common
        metric-name aliases used in the parsed DataFrame.

        For per-share metrics (diluted_eps, diluted_shares), prefers
        the ``raw_value`` column when available so that validation
        compares on the same basis as the published reference (which
        is typically on the pre-split basis for historical years).
        """
        if parsed_df.empty:
            return None

        # Aliases: our core_validation_metrics names may differ from
        # the metric_name column in the parsed DataFrame.
        name_candidates = [metric_name]
        alias_map: dict[str, list[str]] = {
            "FCF": ["FCF", "free_cash_flow"],
            "free_cash_flow": ["free_cash_flow", "FCF"],
            "cash_and_securities": [
                "cash_and_securities",
                "cash_and_short_term_investments",
            ],
            "diluted_EPS": ["diluted_EPS", "diluted_eps"],
            "diluted_eps": ["diluted_eps", "diluted_EPS"],
            "r_and_d": ["r_and_d", "R&D"],
            "R&D": ["R&D", "r_and_d"],
        }
        if metric_name in alias_map:
            name_candidates = alias_map[metric_name]

        # Per-share metrics: use raw_value for validation against
        # published references (which are typically on pre-split basis).
        _PER_SHARE_METRICS = {"diluted_eps", "diluted_EPS", "diluted_shares"}
        use_raw = metric_name in _PER_SHARE_METRICS

        for name in name_candidates:
            mask = (
                (parsed_df["metric_name"] == name)
                & (parsed_df["fiscal_year"] == fy_num)
                & (parsed_df["fiscal_period"] == "FY")
            )
            matched = parsed_df.loc[mask]
            if not matched.empty:
                row = matched.iloc[0]
                # For per-share metrics, prefer raw_value (pre-split basis)
                # so we compare apples-to-apples with published references.
                if use_raw and "raw_value" in matched.columns:
                    raw = row.get("raw_value")
                    if raw is not None and pd.notna(raw):
                        return float(raw)
                val = row["value"]
                if val is not None and pd.notna(val):
                    return float(val)

        return None

    def _check_missing_valuation_critical(
        self,
        validated: list[ValidatedMetric],
        values_by_fy: dict,
    ) -> None:
        """Req 14.11: missing published value for valuation-critical metric.

        If the latest fiscal year has no published reference for a
        valuation-critical metric, mark it as missing + critical + blocker.
        """
        if not values_by_fy:
            return

        # Determine the latest FY
        fy_numbers = []
        for label in values_by_fy:
            n = self._parse_fy_number(label)
            if n is not None:
                fy_numbers.append(n)
        if not fy_numbers:
            return

        latest_fy = max(fy_numbers)

        # Check which valuation-critical metrics already have a validated
        # entry for the latest FY
        existing = {
            (m.metric_name, m.fiscal_year)
            for m in validated
        }

        for metric_name in _VALUATION_CRITICAL_METRICS:
            if metric_name not in self.core_metrics:
                continue
            if (metric_name, latest_fy) in existing:
                # Already validated (may be pass, fail, or missing from parsed)
                continue
            # No published reference at all → add as missing + blocker
            validated.append(
                ValidatedMetric(
                    metric_name=metric_name,
                    fiscal_year=latest_fy,
                    parsed_value=None,
                    published_value=None,
                    diff_pct=None,
                    tolerance=self.tolerance,
                    status="missing",
                    severity="critical",
                    blocker=True,
                )
            )

    def _compute_overall_status(
        self,
        validated: list[ValidatedMetric],
        values_by_fy: dict,
    ) -> DataQualityStatus:
        """Determine the overall :class:`DataQualityStatus`.

        Rules:
        - ``DATA_BLOCKED`` if any valuation-critical metric has
          status in ("fail", "missing") — regardless of whether the
          published reference exists.  Valuation-critical metrics are
          defined in ``_VALUATION_CRITICAL_METRICS``.
        - ``DATA_BLOCKED`` if validation_coverage_pct for the latest 3 FYs
          falls below 90% (Req 14.12).
        - ``PASS_WITH_WARNINGS`` if any non-critical warnings exist
          (status="fail" with severity != "critical" for non-valuation-
          critical metrics).
        - ``PASS`` otherwise.
        """
        # Check for valuation-critical blockers — any fail or missing
        # on a valuation-critical metric blocks formal rating.
        for m in validated:
            if m.metric_name in _VALUATION_CRITICAL_METRICS and m.status in ("fail", "missing"):
                logger.warning(
                    "DATA_BLOCKED: valuation-critical metric %s FY%d status=%s (diff_pct=%s)",
                    m.metric_name, m.fiscal_year, m.status, m.diff_pct,
                )
                return DataQualityStatus.DATA_BLOCKED

        # Also check any explicitly-marked critical blocker
        for m in validated:
            if m.severity == "critical" and m.blocker and m.status in ("fail", "missing"):
                logger.warning(
                    "DATA_BLOCKED: %s FY%d status=%s (diff_pct=%s)",
                    m.metric_name, m.fiscal_year, m.status, m.diff_pct,
                )
                return DataQualityStatus.DATA_BLOCKED

        # Req 14.12: coverage check for latest 3 FYs
        if self._coverage_below_threshold(validated, values_by_fy):
            logger.warning("DATA_BLOCKED: validation_coverage_pct below 90%% for latest 3 FYs")
            return DataQualityStatus.DATA_BLOCKED

        # Check for non-critical warnings
        has_warnings = any(
            m.status == "fail" and m.severity != "critical"
            for m in validated
        )
        if has_warnings:
            return DataQualityStatus.PASS_WITH_WARNINGS

        # Check for any missing (non-blocker) metrics
        has_non_blocker_missing = any(
            m.status == "missing" and not m.blocker
            for m in validated
        )
        if has_non_blocker_missing:
            return DataQualityStatus.PASS_WITH_WARNINGS

        return DataQualityStatus.PASS

    def _coverage_below_threshold(
        self,
        validated: list[ValidatedMetric],
        values_by_fy: dict,
        threshold: float = 90.0,
    ) -> bool:
        """Check if validation coverage for the latest 3 FYs is below threshold.

        Req 14.12: validation_coverage_pct = (pass + warn) / total_core_metric_year_pairs.
        When below 90% for latest 3 FYs → DATA_BLOCKED unless missing
        references are non-valuation-critical.
        """
        fy_numbers = []
        for label in values_by_fy:
            n = self._parse_fy_number(label)
            if n is not None:
                fy_numbers.append(n)

        if not fy_numbers:
            return False

        fy_numbers.sort(reverse=True)
        latest_3 = fy_numbers[:3]

        # Count total expected and passed for latest 3 FYs
        total_pairs = 0
        passed_pairs = 0

        for m in validated:
            if m.fiscal_year not in latest_3:
                continue
            total_pairs += 1
            if m.status in ("pass",):
                passed_pairs += 1

        if total_pairs == 0:
            return True  # No data at all → blocked

        coverage_pct = passed_pairs / total_pairs * 100

        if coverage_pct < threshold:
            # Check if ALL missing/failed are non-valuation-critical
            all_non_critical = all(
                m.metric_name not in _VALUATION_CRITICAL_METRICS
                for m in validated
                if m.fiscal_year in latest_3 and m.status in ("fail", "missing")
            )
            if all_non_critical:
                return False  # Non-valuation-critical misses don't block
            return True

        return False
