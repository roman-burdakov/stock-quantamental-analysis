"""
Tests for audit consistency checks and audit_status.json generation.

Validates:
- limitations.md vs data_quality_report.md contradiction detection
- report rating vs DataQualityStatus contradiction detection
- model_audit claims vs walk-forward results contradiction detection
- Consistent files pass audit
- audit_status.json schema and content
- generate_limitations() includes validation failures

Reqs: 14.8, 14.10, 22.1–22.5
"""

from __future__ import annotations

import json

import pytest

from src.audit_utils import AuditModule
from src.config import (
    AuditStatus,
    DataQualityIssue,
    DataQualityStatus,
    EngineConfig,
    ValidatedMetric,
    get_default_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path) -> EngineConfig:
    cfg = get_default_config()
    cfg.outputs_dir = tmp_path
    return cfg


def _make_failure(
    metric: str = "revenue",
    fy: int = 2025,
    parsed: float = 26_974_000_000.0,
    published: float = 130_497_000_000.0,
) -> ValidatedMetric:
    diff_pct = abs(parsed - published) / abs(published) * 100 if published else None
    return ValidatedMetric(
        metric_name=metric,
        fiscal_year=fy,
        parsed_value=parsed,
        published_value=published,
        diff_pct=diff_pct,
        tolerance=1.0,
        status="fail",
        severity="critical",
        blocker=True,
    )


# ---------------------------------------------------------------------------
# Tests: generate_limitations with validation_failures (Task 12.1)
# ---------------------------------------------------------------------------

class TestLimitationsWithValidationFailures:
    """Req 14.8, 22.1: limitations.md MUST include validation failures."""

    def test_includes_validation_failures_table(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        failures = [_make_failure()]
        content = mod.generate_limitations([], [], validation_failures=failures)
        assert "Validation Failures" in content
        assert "revenue" in content
        assert "FY2025" in content

    def test_includes_parsed_and_published_values(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        failures = [_make_failure()]
        content = mod.generate_limitations([], [], validation_failures=failures)
        assert "26,974,000,000.00" in content
        assert "130,497,000,000.00" in content

    def test_includes_diff_pct(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        failures = [_make_failure()]
        content = mod.generate_limitations([], [], validation_failures=failures)
        # diff_pct should be present as a percentage
        assert "%" in content

    def test_no_issues_message_absent_when_failures_present(self, tmp_path):
        """Req 22.1: SHALL NOT output 'No data quality issues recorded'
        when validation failures are present."""
        mod = AuditModule(_make_config(tmp_path))
        failures = [_make_failure()]
        content = mod.generate_limitations([], [], validation_failures=failures)
        assert "No data quality issues recorded" not in content

    def test_no_issues_message_present_when_no_failures(self, tmp_path):
        """When there are truly no issues, the message is acceptable."""
        mod = AuditModule(_make_config(tmp_path))
        content = mod.generate_limitations([], [])
        assert "No blocking data quality issues" in content

    def test_multiple_failures_listed(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        failures = [
            _make_failure("revenue", 2025, 26_974e6, 130_497e6),
            _make_failure("gross_profit", 2025, 10_000e6, 97_862e6),
        ]
        content = mod.generate_limitations([], [], validation_failures=failures)
        assert "revenue" in content
        assert "gross_profit" in content

    def test_gaps_and_failures_both_shown(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        gaps = [
            DataQualityIssue(
                category="missing_tag",
                detail="SBC not available for FY2016",
                filing_or_source="10-K FY2016",
                severity="warning",
            )
        ]
        failures = [_make_failure()]
        content = mod.generate_limitations(gaps, [], validation_failures=failures)
        assert "Validation Failures" in content
        assert "missing_tag" in content
        assert "No data quality issues recorded" not in content

    def test_passing_metrics_not_in_failures_section(self, tmp_path):
        """Only status='fail' metrics appear in the Validation Failures section."""
        mod = AuditModule(_make_config(tmp_path))
        passing = ValidatedMetric(
            metric_name="net_income",
            fiscal_year=2025,
            parsed_value=72_880e6,
            published_value=72_880e6,
            diff_pct=0.0,
            tolerance=1.0,
            status="pass",
            severity="critical",
            blocker=True,
        )
        content = mod.generate_limitations([], [], validation_failures=[passing])
        # No failures section since the only metric passed
        assert "Validation Failures" not in content


# ---------------------------------------------------------------------------
# Tests: check_audit_consistency (Task 12.2)
# ---------------------------------------------------------------------------

class TestCheckAuditConsistency:
    """Req 22.1–22.3: Cross-file contradiction detection."""

    def _consistent_limitations(self) -> str:
        return (
            "# Limitations\n\n"
            "## Validation Failures\n\n"
            "| Metric | Fiscal Year | Parsed Value | Published Value | Diff % |\n"
            "|--------|-------------|-------------|-----------------|--------|\n"
            "| revenue | FY2025 | 26,974,000,000 | 130,497,000,000 | 79.33% |\n\n"
            "## Data Gaps and Quality Issues\n\n"
            "No additional data gaps beyond the validation failures listed above.\n"
        )

    def _no_issues_limitations(self) -> str:
        return (
            "# Limitations\n\n"
            "## Data Gaps and Quality Issues\n\n"
            "No data quality issues recorded.\n"
        )

    def _dqr_with_failures(self) -> str:
        return (
            "# Data Quality Report\n\n"
            "## Validation Results\n\n"
            "| Metric | Status | Severity |\n"
            "|--------|--------|----------|\n"
            "| revenue | fail | critical |\n"
        )

    def _dqr_clean(self) -> str:
        return (
            "# Data Quality Report\n\n"
            "## Validation Results\n\n"
            "All metrics passed validation.\n"
        )

    def _report_with_buy(self) -> str:
        return (
            "# NVDA Quantamental Report\n\n"
            "**Rating: Buy**\n\n"
            "Target price: $200\n"
        )

    def _report_not_rated(self) -> str:
        return (
            "# NVDA Quantamental Report\n\n"
            "**Not Rated — Data Validation Required**\n\n"
        )

    def _model_audit_claims_outperformance(self) -> str:
        return (
            "# Model Audit\n\n"
            "The model outperforms baselines with lower MAE.\n"
        )

    def _model_audit_honest(self) -> str:
        return (
            "# Model Audit\n\n"
            "The model is exploratory. MAE is comparable to baselines.\n"
        )

    def _ml_results_underperforms(self) -> dict:
        return {
            "model_mae": 0.15,
            "baseline_maes": {
                "last_period": 0.10,
                "trailing_4q_avg": 0.08,
                "three_year_avg": 0.09,
                "linear_trend": 0.11,
            },
        }

    def _ml_results_outperforms(self) -> dict:
        return {
            "model_mae": 0.05,
            "baseline_maes": {
                "last_period": 0.10,
                "trailing_4q_avg": 0.08,
            },
        }

    # --- Check 1: limitations vs DQR ---

    def test_fail_limitations_no_issues_but_dqr_has_failures(self, tmp_path):
        """Req 22.1: Fail when limitations says 'no issues' but DQR has failures."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._no_issues_limitations(),
            data_quality_report_md=self._dqr_with_failures(),
            report_md=self._report_not_rated(),
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            model_audit_md=self._model_audit_honest(),
            ml_results={},
        )
        assert result.overall_status == "fail"
        assert result.checks_failed >= 1
        assert any("limitations" in d.lower() for d in result.failure_details)
        assert result.component_statuses["limitations_consistency"] == "fail"

    def test_pass_limitations_consistent_with_dqr(self, tmp_path):
        """When limitations acknowledges failures and DQR has failures, no contradiction."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._consistent_limitations(),
            data_quality_report_md=self._dqr_with_failures(),
            report_md=self._report_not_rated(),
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            model_audit_md=self._model_audit_honest(),
            ml_results={},
        )
        assert result.component_statuses["limitations_consistency"] == "pass"

    # --- Check 2: report rating vs DATA_BLOCKED ---

    def test_fail_report_buy_but_data_blocked(self, tmp_path):
        """Req 22.2: Fail when report shows Buy but DATA_BLOCKED."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._consistent_limitations(),
            data_quality_report_md=self._dqr_with_failures(),
            report_md=self._report_with_buy(),
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            model_audit_md=self._model_audit_honest(),
            ml_results={},
        )
        assert result.overall_status == "fail"
        assert result.component_statuses["rating_consistency"] == "fail"
        assert any("buy/hold/sell" in d.lower() or "data_blocked" in d.lower()
                    for d in result.failure_details)

    def test_pass_report_not_rated_when_data_blocked(self, tmp_path):
        """Not Rated + DATA_BLOCKED is consistent."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._consistent_limitations(),
            data_quality_report_md=self._dqr_with_failures(),
            report_md=self._report_not_rated(),
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            model_audit_md=self._model_audit_honest(),
            ml_results={},
        )
        assert result.component_statuses["rating_consistency"] == "pass"

    def test_pass_report_buy_when_data_pass(self, tmp_path):
        """Buy + PASS is consistent."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._no_issues_limitations(),
            data_quality_report_md=self._dqr_clean(),
            report_md=self._report_with_buy(),
            data_quality_status=DataQualityStatus.PASS,
            model_audit_md=self._model_audit_honest(),
            ml_results={},
        )
        assert result.component_statuses["rating_consistency"] == "pass"

    # --- Check 3: model_audit claims vs walk-forward results ---

    def test_fail_model_audit_claims_outperformance_but_underperforms(self, tmp_path):
        """Req 22.3: Fail when model_audit claims outperformance but results disagree."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._no_issues_limitations(),
            data_quality_report_md=self._dqr_clean(),
            report_md=self._report_with_buy(),
            data_quality_status=DataQualityStatus.PASS,
            model_audit_md=self._model_audit_claims_outperformance(),
            ml_results=self._ml_results_underperforms(),
        )
        assert result.overall_status == "fail"
        assert result.component_statuses["model_audit_consistency"] == "fail"
        assert any("model" in d.lower() for d in result.failure_details)

    def test_pass_model_audit_honest_about_underperformance(self, tmp_path):
        """Honest model_audit + underperforming results is consistent."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._no_issues_limitations(),
            data_quality_report_md=self._dqr_clean(),
            report_md=self._report_with_buy(),
            data_quality_status=DataQualityStatus.PASS,
            model_audit_md=self._model_audit_honest(),
            ml_results=self._ml_results_underperforms(),
        )
        assert result.component_statuses["model_audit_consistency"] == "pass"

    def test_pass_model_audit_claims_outperformance_and_does(self, tmp_path):
        """Claims outperformance + actually outperforms is consistent."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._no_issues_limitations(),
            data_quality_report_md=self._dqr_clean(),
            report_md=self._report_with_buy(),
            data_quality_status=DataQualityStatus.PASS,
            model_audit_md=self._model_audit_claims_outperformance(),
            ml_results=self._ml_results_outperforms(),
        )
        assert result.component_statuses["model_audit_consistency"] == "pass"

    # --- All consistent ---

    def test_pass_all_consistent(self, tmp_path):
        """All files consistent → overall pass."""
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._no_issues_limitations(),
            data_quality_report_md=self._dqr_clean(),
            report_md=self._report_with_buy(),
            data_quality_status=DataQualityStatus.PASS,
            model_audit_md=self._model_audit_honest(),
            ml_results=self._ml_results_outperforms(),
        )
        assert result.overall_status == "pass"
        assert result.checks_failed == 0
        assert result.checks_run == 5
        assert result.checks_passed == 5
        assert result.failure_details == []
        assert result.blocking_issues == []

    def test_audit_status_has_timestamp(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        result = mod.check_audit_consistency(
            limitations_md=self._no_issues_limitations(),
            data_quality_report_md=self._dqr_clean(),
            report_md=self._report_with_buy(),
            data_quality_status=DataQualityStatus.PASS,
            model_audit_md=self._model_audit_honest(),
            ml_results={},
        )
        assert result.timestamp
        assert "T" in result.timestamp  # ISO format


# ---------------------------------------------------------------------------
# Tests: generate_audit_status_json (Task 12.3)
# ---------------------------------------------------------------------------

class TestGenerateAuditStatusJson:
    """Req 14.10, 22.4: audit_status.json generation."""

    def _make_passing_audit_status(self) -> AuditStatus:
        return AuditStatus(
            overall_status="pass",
            data_quality_status="pass",
            recommendation_eligibility="formal_rating",
            component_statuses={
                "limitations_consistency": "pass",
                "rating_consistency": "pass",
                "model_audit_consistency": "pass",
            },
            checks_run=3,
            checks_passed=3,
            checks_failed=0,
            blocking_issues=[],
            failure_details=[],
            timestamp="2026-04-25T12:00:00Z",
        )

    def _make_failing_audit_status(self) -> AuditStatus:
        return AuditStatus(
            overall_status="fail",
            data_quality_status="data_blocked",
            recommendation_eligibility="diagnostic_not_rated",
            component_statuses={
                "limitations_consistency": "fail",
                "rating_consistency": "pass",
                "model_audit_consistency": "pass",
            },
            checks_run=3,
            checks_passed=2,
            checks_failed=1,
            blocking_issues=["Contradiction: limitations.md claims no issues"],
            failure_details=["Contradiction: limitations.md claims no issues"],
            timestamp="2026-04-25T12:00:00Z",
        )

    def test_writes_json_file(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        status = self._make_passing_audit_status()
        mod.generate_audit_status_json(status, "formal_rating_pass")
        assert (tmp_path / "audit_status.json").exists()

    def test_returns_valid_json(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        status = self._make_passing_audit_status()
        json_str = mod.generate_audit_status_json(status, "formal_rating_pass")
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_schema_fields_present(self, tmp_path):
        """Verify all required fields are in the JSON output."""
        mod = AuditModule(_make_config(tmp_path))
        status = self._make_passing_audit_status()
        json_str = mod.generate_audit_status_json(status, "formal_rating_pass")
        parsed = json.loads(json_str)

        required_fields = [
            "overall_status",
            "package_status",
            "data_quality_status",
            "recommendation_eligibility",
            "component_statuses",
            "checks_run",
            "checks_passed",
            "checks_failed",
            "blocking_issues",
            "failure_details",
            "timestamp",
        ]
        for field in required_fields:
            assert field in parsed, f"Missing field: {field}"

    def test_passing_content(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        status = self._make_passing_audit_status()
        json_str = mod.generate_audit_status_json(status, "formal_rating_pass")
        parsed = json.loads(json_str)

        assert parsed["overall_status"] == "pass"
        assert parsed["package_status"] == "formal_rating_pass"
        assert parsed["data_quality_status"] == "pass"
        assert parsed["recommendation_eligibility"] == "formal_rating"
        assert parsed["checks_run"] == 3
        assert parsed["checks_passed"] == 3
        assert parsed["checks_failed"] == 0
        assert parsed["blocking_issues"] == []
        assert parsed["failure_details"] == []

    def test_failing_content(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        status = self._make_failing_audit_status()
        json_str = mod.generate_audit_status_json(status, "failed")
        parsed = json.loads(json_str)

        assert parsed["overall_status"] == "fail"
        assert parsed["package_status"] == "failed"
        assert parsed["data_quality_status"] == "data_blocked"
        assert parsed["checks_failed"] == 1
        assert len(parsed["blocking_issues"]) == 1
        assert len(parsed["failure_details"]) == 1

    def test_diagnostic_not_rated_pass(self, tmp_path):
        mod = AuditModule(_make_config(tmp_path))
        status = AuditStatus(
            overall_status="pass",
            data_quality_status="data_blocked",
            recommendation_eligibility="diagnostic_not_rated",
            component_statuses={
                "limitations_consistency": "pass",
                "rating_consistency": "pass",
                "model_audit_consistency": "pass",
            },
            checks_run=3,
            checks_passed=3,
            checks_failed=0,
            blocking_issues=[],
            failure_details=[],
            timestamp="2026-04-25T12:00:00Z",
        )
        json_str = mod.generate_audit_status_json(status, "diagnostic_not_rated_pass")
        parsed = json.loads(json_str)
        assert parsed["package_status"] == "diagnostic_not_rated_pass"
        assert parsed["overall_status"] == "pass"

    def test_file_content_matches_return(self, tmp_path):
        """The file on disk should match the returned string."""
        mod = AuditModule(_make_config(tmp_path))
        status = self._make_passing_audit_status()
        json_str = mod.generate_audit_status_json(status, "formal_rating_pass")
        file_content = (tmp_path / "audit_status.json").read_text()
        assert json.loads(file_content) == json.loads(json_str)
