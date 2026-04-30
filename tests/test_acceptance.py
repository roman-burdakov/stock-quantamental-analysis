"""
End-to-end acceptance tests for the NVDA Quantamental Engine.

Covers Milestone 15 tasks:
  15.1 — FY2025 revenue misparse regression
  15.2 — DATA_BLOCKED prevents formal rating
  15.3 — limitations.md consistency
  15.4 — End-to-end pipeline integration
  15.5 — Clean data produces formal rating

All tests run offline using fixtures in tests/fixtures/.
Reqs: 13.1, 14.5, 14.6, 14.7, 14.8, 14.10, 15.7, 17.1, 17.6, 22.1
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit_utils import AuditModule
from src.config import (
    AuditStatus,
    ComponentStatus,
    ComponentStatusEnum,
    DataQualityIssue,
    DataQualityStatus,
    EngineConfig,
    RecommendationStatus,
    ReportMode,
    ValidatedMetric,
    get_default_config,
)
from src.data_validation import DataValidationGate
from src.valuation import ValuationModule
from src.xbrl_parser import XBRLParser

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures")

_PARSED_COLUMNS = [
    "ticker", "fiscal_period", "fiscal_year", "filing_date",
    "source_available_date", "accession_number", "metric_name",
    "value", "unit", "form_type",
    "fiscal_period_type", "period_start", "period_end",
    "duration_days", "frame", "xbrl_concept",
    "selection_rank", "selection_reason",
    "raw_value", "adjusted_value", "adjustment_factor",
    "adjustment_basis", "validation_basis",
]


def _load_published_values() -> dict:
    with open(FIXTURES / "nvda_published_values.json") as f:
        return json.load(f)


def _make_row(
    metric_name: str,
    fiscal_year: int,
    value: float | None,
    *,
    fiscal_period: str = "FY",
    form_type: str = "10-K",
    fiscal_period_type: str = "annual",
    duration_days: int = 363,
    period_start: str = "2024-01-29",
    period_end: str = "2025-01-26",
) -> dict:
    return {
        "ticker": "NVDA",
        "fiscal_period": fiscal_period,
        "fiscal_year": fiscal_year,
        "filing_date": "2025-02-26",
        "source_available_date": "2025-02-26",
        "accession_number": "0001045810-25-000023",
        "metric_name": metric_name,
        "value": value,
        "unit": "USD",
        "form_type": form_type,
        "fiscal_period_type": fiscal_period_type,
        "period_start": period_start,
        "period_end": period_end,
        "duration_days": duration_days,
        "frame": None,
        "xbrl_concept": "",
        "selection_rank": 1,
        "selection_reason": "10-K annual duration",
        "raw_value": value,
        "adjusted_value": value,
        "adjustment_factor": 1.0,
        "adjustment_basis": "as_reported",
        "validation_basis": "as_reported",
    }


def _build_matching_df(published_values: dict) -> pd.DataFrame:
    """Build a parsed DataFrame with values exactly matching published values."""
    rows = []
    for fy_label, fy_data in published_values.get("values", {}).items():
        fy_num = int(fy_label.replace("FY", ""))
        fy_end = fy_data.get("fiscal_year_end", "2025-01-26")
        fy_start = fy_data.get("fiscal_year_start", "2024-01-29")
        for metric_name, metric_info in fy_data.get("metrics", {}).items():
            rows.append(_make_row(
                metric_name=metric_name,
                fiscal_year=fy_num,
                value=metric_info["value"],
                period_start=fy_start,
                period_end=fy_end,
            ))
    return pd.DataFrame(rows, columns=_PARSED_COLUMNS)


def _build_df_from_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_PARSED_COLUMNS)


def _make_config(**overrides) -> EngineConfig:
    return EngineConfig(**overrides)



# ===================================================================
# Task 15.1 — Regression test: FY2025 revenue misparse
# ===================================================================


class TestFY2025RevenueMisparseRegression:
    """Load companyfacts fixture, run parser, verify FY2025 revenue ~$130.5B.

    Verify that the $27B value is classified as quarterly/YTD and rejected.
    Reqs: 15.7
    """

    @pytest.fixture()
    def regression_fixture(self) -> dict:
        path = FIXTURES / "regression" / "fy2025_revenue_candidates.json"
        with open(path) as f:
            return json.load(f)

    @pytest.fixture()
    def companyfacts_path(self) -> Path:
        return Path("data/raw/companyfacts_CIK0001045810.json")

    # --- Test with real companyfacts if available ---

    def test_parser_selects_correct_fy2025_revenue(self, companyfacts_path: Path):
        """Run XBRLParser on real companyfacts and verify FY2025 revenue ~$130.5B."""
        if not companyfacts_path.exists():
            pytest.skip("Real companyfacts file not available")

        with open(companyfacts_path) as f:
            facts_json = json.load(f)

        parser = XBRLParser()
        df = parser.parse_companyfacts(facts_json)

        # Find FY2025 annual revenue
        mask = (
            (df["metric_name"] == "revenue")
            & (df["fiscal_year"] == 2025)
            & (df["fiscal_period"] == "FY")
        )
        fy2025_rev = df.loc[mask]

        assert not fy2025_rev.empty, "No FY2025 annual revenue found"
        parsed_value = float(fy2025_rev.iloc[0]["value"])

        # Must be ~$130.5B (within 1%)
        expected = 130497000000
        diff_pct = abs(parsed_value - expected) / expected * 100
        assert diff_pct <= 1.0, (
            f"FY2025 revenue parsed as ${parsed_value / 1e9:.3f}B, "
            f"expected ~${expected / 1e9:.3f}B (diff={diff_pct:.2f}%)"
        )

    def test_27b_value_not_selected_as_fy2025(self, companyfacts_path: Path):
        """Verify the $26.974B misparse value is NOT selected as FY2025 revenue."""
        if not companyfacts_path.exists():
            pytest.skip("Real companyfacts file not available")

        with open(companyfacts_path) as f:
            facts_json = json.load(f)

        parser = XBRLParser()
        df = parser.parse_companyfacts(facts_json)

        mask = (
            (df["metric_name"] == "revenue")
            & (df["fiscal_year"] == 2025)
            & (df["fiscal_period"] == "FY")
        )
        fy2025_rev = df.loc[mask]

        assert not fy2025_rev.empty
        parsed_value = float(fy2025_rev.iloc[0]["value"])

        # Must NOT be the misparse value (~$27B)
        misparse_value = 26974000000
        assert abs(parsed_value - misparse_value) / misparse_value > 0.10, (
            f"FY2025 revenue parsed as ${parsed_value / 1e9:.3f}B — "
            "this is the misparse value that should have been rejected"
        )

    def test_quarterly_candidates_classified_correctly(self, regression_fixture: dict):
        """Verify quarterly candidates in the fixture are classified as quarterly."""
        candidates = regression_fixture["candidates"]
        quarterly = [c for c in candidates if c["period_type"] == "quarterly"]

        assert len(quarterly) > 0, "No quarterly candidates in fixture"
        for q in quarterly:
            assert q["duration_days"] < 100, (
                f"Quarterly candidate has duration {q['duration_days']} days"
            )

    def test_ytd_candidates_classified_correctly(self, regression_fixture: dict):
        """Verify YTD candidates in the fixture are classified as YTD."""
        candidates = regression_fixture["candidates"]
        ytd = [c for c in candidates if c["period_type"] == "ytd"]

        assert len(ytd) > 0, "No YTD candidates in fixture"
        for y in ytd:
            assert 100 <= y["duration_days"] < 340, (
                f"YTD candidate has duration {y['duration_days']} days"
            )

    def test_correct_annual_value_in_fixture(self, regression_fixture: dict):
        """Verify the fixture's correct_annual_value matches published $130.497B."""
        correct = regression_fixture["correct_annual_value"]
        assert correct["val"] == 130497000000
        assert correct["period_type"] == "annual"
        assert correct["duration_days"] >= 350

    def test_misparse_value_is_comparative_period(self, regression_fixture: dict):
        """Verify the misparse value is from a comparative period (FY2023)."""
        misparse = regression_fixture["misparse_value"]
        assert misparse["val"] == 26974000000
        # The misparse is FY2023 revenue reported in the FY2025 10-K
        assert misparse["start"] == "2022-01-31"
        assert misparse["end"] == "2023-01-29"

    def test_validation_gate_passes_with_correct_parse(self, companyfacts_path: Path):
        """Run parser + validation gate: correct parse should not block on revenue."""
        if not companyfacts_path.exists():
            pytest.skip("Real companyfacts file not available")

        with open(companyfacts_path) as f:
            facts_json = json.load(f)

        parser = XBRLParser()
        df = parser.parse_companyfacts(facts_json)
        published = _load_published_values()

        gate = DataValidationGate(EngineConfig())
        status, validated = gate.validate_parsed_data(df, published)

        # Revenue specifically should pass
        rev_2025 = [
            m for m in validated
            if m.metric_name == "revenue" and m.fiscal_year == 2025
        ]
        assert len(rev_2025) >= 1
        assert rev_2025[0].status == "pass", (
            f"Revenue FY2025 validation status={rev_2025[0].status}, "
            f"parsed={rev_2025[0].parsed_value}, "
            f"published={rev_2025[0].published_value}, "
            f"diff={rev_2025[0].diff_pct}%"
        )


# ===================================================================
# Task 15.2 — Regression test: DATA_BLOCKED prevents formal rating
# ===================================================================


class TestDataBlockedPreventsRating:
    """Inject bad revenue value, run validation gate, verify:
    - DataQualityStatus = DATA_BLOCKED
    - recommendation = "Not Rated"
    - valuation labeled "Diagnostic only"

    Reqs: 14.5, 14.6, 14.7
    """

    @pytest.fixture()
    def bad_revenue_df(self) -> pd.DataFrame:
        """Build a parsed DataFrame with bad FY2025 revenue ($27B misparse)."""
        published = _load_published_values()
        df = _build_matching_df(published)
        # Inject the misparse value for FY2025 revenue
        mask = (df["metric_name"] == "revenue") & (df["fiscal_year"] == 2025)
        df.loc[mask, "value"] = 26974000000
        return df

    @pytest.fixture()
    def blocked_status_and_metrics(self, bad_revenue_df):
        """Run validation gate on bad data, return (status, validated)."""
        published = _load_published_values()
        gate = DataValidationGate(EngineConfig())
        return gate.validate_parsed_data(bad_revenue_df, published)

    def test_data_quality_status_is_blocked(self, blocked_status_and_metrics):
        """DataQualityStatus must be DATA_BLOCKED."""
        status, _ = blocked_status_and_metrics
        assert status is DataQualityStatus.DATA_BLOCKED

    def test_recommendation_is_not_rated(self, blocked_status_and_metrics):
        """When DATA_BLOCKED, recommendation must be 'Not Rated'."""
        status, validated = blocked_status_and_metrics

        # Build component statuses for eligibility gate
        val_module = ValuationModule()
        val_input_status = val_module.check_valuation_inputs(status, validated)

        audit = AuditModule()
        component_statuses = [
            ComponentStatus("data_validation", ComponentStatusEnum.BLOCKED,
                            "DATA_BLOCKED"),
            ComponentStatus("market_price", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("dcf_assumptions", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("split_basis", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("lookahead", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("audit_consistency", ComponentStatusEnum.USABLE, "ok"),
        ]
        rec_status = audit.run_pre_report_audit(
            status, component_statuses, val_input_status,
        )

        assert rec_status.eligibility_status == ReportMode.DIAGNOSTIC_NOT_RATED

        # Generate recommendation
        scenarios = val_module.build_scenarios(
            base_revenue=130497000000,
            wacc=0.10,
            net_cash=34747000000,
            shares=24804000000,
        )
        reverse_dcf = pd.DataFrame({"margin=0.30": [100.0]}, index=["cagr=0.20"])
        reverse_dcf.attrs["highlight"] = pd.DataFrame(
            {"margin=0.30": [False]}, index=["cagr=0.20"],
        )

        rec = val_module.generate_recommendation(
            scenarios=scenarios,
            current_price=120.0,
            reverse_dcf=reverse_dcf,
            recommendation_status=rec_status,
        )

        assert rec.rating == "Not Rated"
        assert rec.eligibility_status == "diagnostic_not_rated"
        assert rec.not_rated_reason is not None

    def test_valuation_labeled_diagnostic_only(self, blocked_status_and_metrics):
        """Valuation outputs must be labeled 'Diagnostic only'."""
        status, _ = blocked_status_and_metrics
        gate = DataValidationGate(EngineConfig())

        label = gate.get_diagnostic_label(status)
        assert "Diagnostic only" in label

    def test_valuation_inputs_blocked(self, blocked_status_and_metrics):
        """ValuationModule.check_valuation_inputs should return BLOCKED."""
        status, validated = blocked_status_and_metrics
        val_module = ValuationModule()
        val_status = val_module.check_valuation_inputs(status, validated)

        assert val_status.status == ComponentStatusEnum.BLOCKED

    def test_blocking_issues_populated(self, blocked_status_and_metrics):
        """RecommendationStatus should list blocking issues."""
        status, validated = blocked_status_and_metrics

        val_module = ValuationModule()
        val_input_status = val_module.check_valuation_inputs(status, validated)

        audit = AuditModule()
        component_statuses = [
            ComponentStatus("data_validation", ComponentStatusEnum.BLOCKED,
                            "DATA_BLOCKED"),
            ComponentStatus("market_price", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("dcf_assumptions", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("split_basis", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("lookahead", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("audit_consistency", ComponentStatusEnum.USABLE, "ok"),
        ]
        rec_status = audit.run_pre_report_audit(
            status, component_statuses, val_input_status,
        )

        assert len(rec_status.blocking_issues) > 0
        # Should mention DATA_BLOCKED
        combined = " ".join(rec_status.blocking_issues).lower()
        assert "data_blocked" in combined or "blocked" in combined


# ===================================================================
# Task 15.3 — Regression test: limitations.md consistency
# ===================================================================


class TestLimitationsConsistency:
    """Run pipeline with validation failures, verify:
    - limitations.md lists failures (not "no issues")
    - audit consistency check passes

    Reqs: 14.8, 22.1
    """

    @pytest.fixture()
    def validation_failures(self) -> list[ValidatedMetric]:
        """Create a list of ValidatedMetric with failures."""
        return [
            ValidatedMetric(
                metric_name="revenue",
                fiscal_year=2025,
                parsed_value=26974000000,
                published_value=130497000000,
                diff_pct=79.33,
                tolerance=1.0,
                status="fail",
                severity="critical",
                blocker=True,
            ),
            ValidatedMetric(
                metric_name="net_income",
                fiscal_year=2025,
                parsed_value=72880000000,
                published_value=72880000000,
                diff_pct=0.0,
                tolerance=1.0,
                status="pass",
                severity="critical",
                blocker=True,
            ),
        ]

    @pytest.fixture()
    def audit_with_tmpdir(self, tmp_path):
        """Create an AuditModule writing to a temp directory."""
        config = get_default_config()
        config.outputs_dir = tmp_path
        return AuditModule(config)

    def test_limitations_lists_validation_failures(
        self, audit_with_tmpdir, validation_failures,
    ):
        """limitations.md must include validation failures when they exist."""
        audit = audit_with_tmpdir
        limitations_md = audit.generate_limitations(
            gaps=[],
            assumptions=["WACC = 10%"],
            validation_failures=validation_failures,
        )

        # Must mention the failed metric
        assert "revenue" in limitations_md.lower()
        assert "fail" in limitations_md.lower() or "validation" in limitations_md.lower()

    def test_limitations_does_not_say_no_issues_when_failures_exist(
        self, audit_with_tmpdir, validation_failures,
    ):
        """limitations.md SHALL NOT say 'no issues' when validation failures exist."""
        audit = audit_with_tmpdir
        limitations_md = audit.generate_limitations(
            gaps=[],
            assumptions=[],
            validation_failures=validation_failures,
        )

        lim_lower = limitations_md.lower()
        no_issues_phrases = [
            "no data quality issues recorded",
            "no data quality issues found",
            "no validation failures",
        ]
        for phrase in no_issues_phrases:
            assert phrase not in lim_lower, (
                f"limitations.md contains '{phrase}' despite validation failures"
            )

    def test_limitations_includes_diff_pct(
        self, audit_with_tmpdir, validation_failures,
    ):
        """limitations.md should include diff_pct for failed metrics."""
        audit = audit_with_tmpdir
        limitations_md = audit.generate_limitations(
            gaps=[],
            assumptions=[],
            validation_failures=validation_failures,
        )

        # Should contain the diff percentage for the failed revenue
        assert "79" in limitations_md  # ~79.33%

    def test_audit_consistency_passes_when_limitations_lists_failures(
        self, audit_with_tmpdir, validation_failures,
    ):
        """Audit consistency check should PASS when limitations.md correctly
        lists validation failures and the report is consistent.
        """
        audit = audit_with_tmpdir

        # Generate limitations with failures
        limitations_md = audit.generate_limitations(
            gaps=[],
            assumptions=[],
            validation_failures=validation_failures,
        )

        # Generate a data quality report that also mentions failures
        dqr_md = (
            "# Data Quality Report\n\n"
            "**Status: DATA_BLOCKED**\n\n"
            "| Metric | Status |\n|---|---|\n"
            "| revenue | fail |\n"
            "| net_income | pass |\n"
        )

        # Report that says "Not Rated" (consistent with DATA_BLOCKED)
        report_md = (
            "# NVDA Quantamental Report\n\n"
            "**Rating: Not Rated — Data Validation Required**\n\n"
            "Valuation: Diagnostic only\n"
        )

        model_audit_md = "# Model Audit\n\nML model is diagnostic only.\n"

        audit_status = audit.check_audit_consistency(
            limitations_md=limitations_md,
            data_quality_report_md=dqr_md,
            report_md=report_md,
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            model_audit_md=model_audit_md,
            ml_results={},
        )

        assert audit_status.overall_status == "pass", (
            f"Audit consistency failed: {audit_status.failure_details}"
        )

    def test_audit_consistency_fails_when_limitations_says_no_issues(
        self, audit_with_tmpdir,
    ):
        """Audit consistency should FAIL if limitations says 'no issues'
        but data_quality_report has failures.
        """
        audit = audit_with_tmpdir

        # Broken limitations that claims no issues
        limitations_md = (
            "# Limitations\n\n"
            "## Data Gaps and Quality Issues\n\n"
            "No data quality issues recorded.\n"
        )

        # DQR that has failures
        dqr_md = (
            "# Data Quality Report\n\n"
            "**Status: DATA_BLOCKED**\n\n"
            "revenue FY2025: fail (critical)\n"
        )

        report_md = "# Report\n\nRating: Not Rated\n"
        model_audit_md = "# Model Audit\n"

        audit_status = audit.check_audit_consistency(
            limitations_md=limitations_md,
            data_quality_report_md=dqr_md,
            report_md=report_md,
            data_quality_status=DataQualityStatus.DATA_BLOCKED,
            model_audit_md=model_audit_md,
            ml_results={},
        )

        assert audit_status.overall_status == "fail"
        assert any(
            "contradiction" in d.lower() or "limitations" in d.lower()
            for d in audit_status.failure_details
        )

    def test_limitations_with_data_gaps_and_failures(
        self, audit_with_tmpdir, validation_failures,
    ):
        """limitations.md should include both data gaps and validation failures."""
        audit = audit_with_tmpdir
        gaps = [
            DataQualityIssue(
                category="missing_tag",
                detail="capex tag not found for FY2023",
                filing_or_source="0001045810-23-000017",
                severity="warning",
            ),
        ]

        limitations_md = audit.generate_limitations(
            gaps=gaps,
            assumptions=["WACC = 10%"],
            validation_failures=validation_failures,
        )

        # Should contain both the gap and the validation failure
        assert "capex" in limitations_md.lower()
        assert "revenue" in limitations_md.lower()


# ===================================================================
# Task 15.4 — End-to-end pipeline test
# ===================================================================


class TestEndToEndPipeline:
    """Run full pipeline stages with cached/fixture data, verify:
    - All required outputs exist
    - audit_status.json is valid
    - Report mode matches data quality status

    Reqs: 13.1, 14.10
    """

    @pytest.fixture()
    def pipeline_outputs(self, tmp_path):
        """Run the core pipeline stages and return the outputs directory."""
        config = get_default_config()
        config.outputs_dir = tmp_path

        published = _load_published_values()
        df = _build_matching_df(published)

        # Stage 1: Validation gate
        gate = DataValidationGate(config)
        dq_status, validated = gate.validate_parsed_data(df, published)

        # Stage 2: Valuation input check
        val_module = ValuationModule(config)
        val_input_status = val_module.check_valuation_inputs(dq_status, validated)

        # Stage 3: Pre-report eligibility audit
        audit = AuditModule(config)
        component_statuses = [
            ComponentStatus("data_validation", ComponentStatusEnum.USABLE,
                            "All metrics pass"),
            ComponentStatus("market_price", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("dcf_assumptions", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("split_basis", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("lookahead", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("audit_consistency", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY,
                            "Limited training data"),
            ComponentStatus("nlp_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY,
                            "Poor extraction quality"),
        ]
        rec_status = audit.run_pre_report_audit(
            dq_status, component_statuses, val_input_status,
        )

        # Stage 4: Generate limitations
        limitations_md = audit.generate_limitations(
            gaps=[],
            assumptions=["WACC = 10%", "Terminal growth = 3%"],
            validation_failures=validated,
        )

        # Stage 5: Generate validation summary
        val_summary = gate.generate_validation_summary(dq_status, validated)

        # Stage 6: Generate a minimal report
        report_md = "# NVDA Quantamental Report\n\n"
        if rec_status.eligibility_status == ReportMode.FORMAL_RATING:
            report_md += "**Rating: Hold**\n\n"
        else:
            report_md += "**Rating: Not Rated**\n\n"
        report_md += "Valuation analysis follows.\n"

        # Stage 7: Audit consistency check
        dqr_md = "# Data Quality Report\n\nAll metrics pass validation.\n"
        model_audit_md = "# Model Audit\n\nML model is diagnostic only.\n"

        audit_result = audit.check_audit_consistency(
            limitations_md=limitations_md,
            data_quality_report_md=dqr_md,
            report_md=report_md,
            data_quality_status=dq_status,
            model_audit_md=model_audit_md,
            ml_results={},
        )

        # Stage 8: Determine package status
        if audit_result.overall_status == "fail":
            package_status = "failed"
        elif rec_status.eligibility_status == ReportMode.FORMAL_RATING:
            package_status = "formal_rating_pass"
        else:
            package_status = "diagnostic_not_rated_pass"

        # Stage 9: Write audit_status.json
        audit.generate_audit_status_json(audit_result, package_status)

        # Write other outputs for completeness
        (tmp_path / "limitations.md").write_text(limitations_md)
        (tmp_path / "nvda_quantamental_report.md").write_text(report_md)
        (tmp_path / "data_quality_report.md").write_text(dqr_md)
        (tmp_path / "model_audit.md").write_text(model_audit_md)

        return {
            "outputs_dir": tmp_path,
            "dq_status": dq_status,
            "rec_status": rec_status,
            "audit_result": audit_result,
            "package_status": package_status,
            "validated": validated,
        }

    def test_audit_status_json_exists(self, pipeline_outputs):
        """audit_status.json must be written to outputs."""
        path = pipeline_outputs["outputs_dir"] / "audit_status.json"
        assert path.exists(), "audit_status.json not found"

    def test_audit_status_json_is_valid(self, pipeline_outputs):
        """audit_status.json must be valid JSON with required fields."""
        path = pipeline_outputs["outputs_dir"] / "audit_status.json"
        with open(path) as f:
            data = json.load(f)

        required_keys = [
            "overall_status", "package_status", "data_quality_status",
            "recommendation_eligibility", "component_statuses",
            "checks_run", "checks_passed", "checks_failed",
            "blocking_issues", "failure_details", "timestamp",
        ]
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in audit_status.json"

    def test_audit_status_json_package_status_valid(self, pipeline_outputs):
        """package_status must be one of the three valid values."""
        path = pipeline_outputs["outputs_dir"] / "audit_status.json"
        with open(path) as f:
            data = json.load(f)

        valid_statuses = {
            "formal_rating_pass",
            "diagnostic_not_rated_pass",
            "failed",
        }
        assert data["package_status"] in valid_statuses

    def test_report_mode_matches_data_quality(self, pipeline_outputs):
        """Report mode should be consistent with data quality status."""
        dq_status = pipeline_outputs["dq_status"]
        rec_status = pipeline_outputs["rec_status"]

        if dq_status is DataQualityStatus.PASS:
            # With clean data and all gates passing, should be formal_rating
            assert rec_status.eligibility_status == ReportMode.FORMAL_RATING
        elif dq_status is DataQualityStatus.DATA_BLOCKED:
            assert rec_status.eligibility_status == ReportMode.DIAGNOSTIC_NOT_RATED

    def test_all_required_output_files_exist(self, pipeline_outputs):
        """All key output files should be present."""
        outputs_dir = pipeline_outputs["outputs_dir"]
        required_files = [
            "audit_status.json",
            "limitations.md",
            "nvda_quantamental_report.md",
        ]
        for fname in required_files:
            assert (outputs_dir / fname).exists(), f"Missing output: {fname}"

    def test_audit_consistency_passes_for_clean_pipeline(self, pipeline_outputs):
        """Audit consistency should pass when all files are consistent."""
        audit_result = pipeline_outputs["audit_result"]
        assert audit_result.overall_status == "pass", (
            f"Audit failed: {audit_result.failure_details}"
        )

    def test_checks_run_count(self, pipeline_outputs):
        """At least 3 consistency checks should be run."""
        audit_result = pipeline_outputs["audit_result"]
        assert audit_result.checks_run >= 3

    def test_package_status_is_formal_rating_pass(self, pipeline_outputs):
        """With clean data, package status should be formal_rating_pass."""
        assert pipeline_outputs["package_status"] == "formal_rating_pass"


# ===================================================================
# Task 15.5 — Clean data produces formal rating
# ===================================================================


class TestCleanDataProducesFormalRating:
    """Use fixtures with correct published values, verify:
    - DataQualityStatus = PASS
    - recommendation is Buy/Hold/Sell (not "Not Rated")
    - scorecard has no blocked components

    Reqs: 17.1, 17.6
    """

    @pytest.fixture()
    def clean_validation(self):
        """Run validation with clean (matching) data."""
        published = _load_published_values()
        df = _build_matching_df(published)
        gate = DataValidationGate(EngineConfig())
        return gate.validate_parsed_data(df, published)

    @pytest.fixture()
    def formal_recommendation(self, clean_validation):
        """Build a full recommendation through the eligibility gate."""
        dq_status, validated = clean_validation
        config = get_default_config()

        # Valuation input check
        val_module = ValuationModule(config)
        val_input_status = val_module.check_valuation_inputs(dq_status, validated)

        # Pre-report eligibility audit
        audit = AuditModule(config)
        component_statuses = [
            ComponentStatus("data_validation", ComponentStatusEnum.USABLE,
                            "All metrics pass"),
            ComponentStatus("market_price", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("dcf_assumptions", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("split_basis", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("lookahead", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("audit_consistency", ComponentStatusEnum.USABLE, "ok"),
            ComponentStatus("ml_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY,
                            "Limited training data"),
            ComponentStatus("nlp_signal", ComponentStatusEnum.DIAGNOSTIC_ONLY,
                            "Poor extraction quality"),
        ]
        rec_status = audit.run_pre_report_audit(
            dq_status, component_statuses, val_input_status,
        )

        # Build scenarios and recommendation
        base_revenue = 130497000000
        net_cash = 43210000000 - 8463000000  # cash - debt from FY2025
        shares = 24804000000

        val_module.set_base_revenue(base_revenue)
        scenarios = val_module.build_scenarios(
            base_revenue=base_revenue,
            wacc=config.wacc,
            net_cash=net_cash,
            shares=shares,
        )

        reverse_dcf = val_module.compute_reverse_dcf_grid(
            price=120.0,
            shares=shares,
            cash=net_cash,
            wacc=config.wacc,
            cagr_range=config.reverse_dcf_cagr_range,
            margin_range=config.reverse_dcf_margin_range,
            current_price=120.0,
        )

        rec = val_module.generate_recommendation(
            scenarios=scenarios,
            current_price=120.0,
            reverse_dcf=reverse_dcf,
            ml_signal=0.5,
            narrative_signal=0.3,
            recommendation_status=rec_status,
        )

        return {
            "dq_status": dq_status,
            "rec_status": rec_status,
            "recommendation": rec,
            "val_input_status": val_input_status,
        }

    def test_data_quality_status_is_pass(self, clean_validation):
        """DataQualityStatus must be PASS with clean data."""
        status, _ = clean_validation
        assert status is DataQualityStatus.PASS

    def test_eligibility_is_formal_rating(self, formal_recommendation):
        """Eligibility status must be FORMAL_RATING."""
        rec_status = formal_recommendation["rec_status"]
        assert rec_status.eligibility_status == ReportMode.FORMAL_RATING

    def test_recommendation_is_buy_hold_or_sell(self, formal_recommendation):
        """Recommendation must be Buy, Hold, or Sell (not 'Not Rated')."""
        rec = formal_recommendation["recommendation"]
        assert rec.rating in ("Buy", "Hold", "Sell"), (
            f"Expected formal rating, got '{rec.rating}'"
        )

    def test_recommendation_not_not_rated(self, formal_recommendation):
        """Recommendation must NOT be 'Not Rated'."""
        rec = formal_recommendation["recommendation"]
        assert rec.rating != "Not Rated"

    def test_eligibility_status_field_is_formal(self, formal_recommendation):
        """Recommendation.eligibility_status should be 'formal_rating'."""
        rec = formal_recommendation["recommendation"]
        assert rec.eligibility_status == "formal_rating"

    def test_not_rated_reason_is_none(self, formal_recommendation):
        """not_rated_reason should be None for formal rating."""
        rec = formal_recommendation["recommendation"]
        assert rec.not_rated_reason is None

    def test_scorecard_has_no_blocked_components(self, formal_recommendation):
        """Scorecard should have no 'blocked' components."""
        rec = formal_recommendation["recommendation"]
        for component_name, component_data in rec.scorecard.items():
            status = component_data.get("status", "")
            assert status != ComponentStatusEnum.BLOCKED.value, (
                f"Scorecard component '{component_name}' is blocked"
            )

    def test_scorecard_no_none_statuses(self, formal_recommendation):
        """Scorecard components must never have None status (Req 17.4)."""
        rec = formal_recommendation["recommendation"]
        for component_name, component_data in rec.scorecard.items():
            assert component_data.get("status") is not None, (
                f"Scorecard component '{component_name}' has None status"
            )

    def test_valuation_inputs_usable(self, formal_recommendation):
        """Valuation inputs should be usable with clean data."""
        val_status = formal_recommendation["val_input_status"]
        assert val_status.status == ComponentStatusEnum.USABLE

    def test_no_blocking_issues(self, formal_recommendation):
        """RecommendationStatus should have no blocking issues."""
        rec_status = formal_recommendation["rec_status"]
        assert len(rec_status.blocking_issues) == 0

    def test_all_validated_metrics_pass(self, clean_validation):
        """Every validated metric with both parsed and published values should pass."""
        _, validated = clean_validation
        for m in validated:
            if m.parsed_value is not None and m.published_value is not None:
                assert m.status == "pass", (
                    f"{m.metric_name} FY{m.fiscal_year}: "
                    f"status={m.status}, diff={m.diff_pct}%"
                )

    def test_diagnostic_only_components_dont_block(self, formal_recommendation):
        """ML and NLP being diagnostic_only should NOT block formal rating."""
        rec_status = formal_recommendation["rec_status"]
        # Verify ML and NLP are diagnostic_only in component statuses
        ml_statuses = [
            cs for cs in rec_status.component_statuses
            if cs.component_name == "ml_signal"
        ]
        nlp_statuses = [
            cs for cs in rec_status.component_statuses
            if cs.component_name == "nlp_signal"
        ]

        if ml_statuses:
            assert ml_statuses[0].status == ComponentStatusEnum.DIAGNOSTIC_ONLY
        if nlp_statuses:
            assert nlp_statuses[0].status == ComponentStatusEnum.DIAGNOSTIC_ONLY

        # Despite diagnostic_only ML/NLP, eligibility should still be formal
        assert rec_status.eligibility_status == ReportMode.FORMAL_RATING
