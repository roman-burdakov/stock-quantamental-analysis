"""
Task 8.5 — Cross-file consistency test for the v2 ML/NLP layer.

This is the pytest counterpart to ``scripts/check_v2_consistency.sh``.
It guards against terminology drift between the spec docs and the
implemented code (the failure mode the Andrew Lo–style audits flagged
repeatedly during spec iteration).

Two kinds of checks:

1. **Banned terms** — symbols and phrases that should NEVER appear in
   implemented v2 code, report templates, or audit_status.json. These
   represent the rejected naming from earlier spec drafts.

2. **Required terms** — symbols that MUST be present where the v2 design
   defines them. These prevent silent regressions where the implementation
   drifts back to old naming.

Run with: ``pytest tests/test_v2_spec_consistency.py``
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

# --- v2 source files. May not exist yet during early implementation. ---
V2_SRC_FILES = [
    REPO_ROOT / "src" / "ml_decision.py",
    REPO_ROOT / "src" / "analyst_estimates.py",
    REPO_ROOT / "src" / "peer_freshness.py",
    REPO_ROOT / "src" / "quarterly_panel.py",
    REPO_ROOT / "src" / "market_features.py",
    REPO_ROOT / "src" / "ml_v2_provenance.py",
]

# --- Existing files modified by v2. v2 code paths inside must be clean. ---
V2_MOD_FILES = [
    REPO_ROOT / "src" / "config.py",
    REPO_ROOT / "src" / "valuation.py",
    REPO_ROOT / "src" / "charts.py",
    REPO_ROOT / "src" / "audit_utils.py",
    REPO_ROOT / "src" / "report_utils.py",
    REPO_ROOT / "scripts" / "run_pipeline.py",
]

TEMPLATE_DIR = REPO_ROOT / "src" / "templates"
AUDIT_JSON = REPO_ROOT / "outputs" / "audit_status.json"

# --- Banned terms (apply to source code AND templates AND audit JSON) ---
BANNED_TERMS = [
    "binomial_p_vs_naive",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_target_price_ci",
    "decision_with_analyst",
    "analyst_consensus_baseline",
    "analyst_baseline",
]

# --- Banned phrases in rendered output (templates) ---
# These are reader-facing terms that should be replaced with
# "empirical residual band" wherever they used to apply to bootstrap output.
BANNED_TEMPLATE_PHRASES = [
    "90% interval",
    "90% confidence interval",
    "confidence interval",
    "confidence-band",
]


def _find_in_file(path: Path, needle: str) -> list[tuple[int, str]]:
    """Return list of (line_number, line) where needle appears in path."""
    if not path.exists() or not path.is_file():
        return []
    matches: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            matches.append((i, line.strip()))
    return matches


def _existing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_file()]


# ---------------------------------------------------------------------------
# Tests: banned terms in v2 code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("banned", BANNED_TERMS)
def test_banned_terms_not_in_v2_src(banned: str) -> None:
    """No banned term appears in any implemented v2 source file."""
    offenders: list[str] = []
    for path in _existing(V2_SRC_FILES + V2_MOD_FILES):
        for line_no, line in _find_in_file(path, banned):
            offenders.append(f"{path}:{line_no}: {line}")
    assert not offenders, (
        f"Banned term '{banned}' found in v2 implementation:\n"
        + "\n".join(offenders)
        + "\n\nReplace per the v2 spec (see .kiro/specs/nvda-quantamental-engine-v2/)."
    )


@pytest.mark.parametrize("banned", BANNED_TEMPLATE_PHRASES)
def test_banned_phrases_not_in_templates(banned: str) -> None:
    """Reader-facing templates must use 'empirical residual band' wording."""
    if not TEMPLATE_DIR.exists():
        pytest.skip("Template directory not found")
    template_files = [
        p
        for p in TEMPLATE_DIR.rglob("*")
        if p.is_file() and p.suffix in {".j2", ".md", ".html"}
    ]
    offenders: list[str] = []
    for path in template_files:
        for line_no, line in _find_in_file(path, banned):
            offenders.append(f"{path}:{line_no}: {line}")
    assert not offenders, (
        f"Banned template phrase '{banned}' found:\n"
        + "\n".join(offenders)
        + "\n\nUse 'empirical residual band, 5th–95th percentile' instead."
    )


@pytest.mark.parametrize("banned", BANNED_TERMS)
def test_banned_keys_not_in_audit_json(banned: str) -> None:
    """audit_status.json must not contain banned key names."""
    if not AUDIT_JSON.exists():
        pytest.skip("audit_status.json not yet written")
    text = AUDIT_JSON.read_text(encoding="utf-8")
    pattern = f'"{banned}"'
    assert pattern not in text, (
        f"Banned key '{banned}' found in {AUDIT_JSON}. "
        f"audit_status.json schema must use the corrected naming."
    )


# ---------------------------------------------------------------------------
# Tests: forbidden methods on analyst classes
# ---------------------------------------------------------------------------

def test_analyst_estimates_has_no_to_features() -> None:
    """AnalystEstimateFetcher must NOT define a to_features() method.

    Analyst data is a decision-time overlay (Req 2), never a learned ML
    feature. Defining to_features would invite reintroduction of the
    original mathematically-degenerate analyst-augmented model.
    """
    path = REPO_ROOT / "src" / "analyst_estimates.py"
    if not path.exists():
        pytest.skip("analyst_estimates.py not yet implemented")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"def\s+to_features\s*\(", re.MULTILINE)
    matches = pattern.findall(text)
    assert not matches, (
        "src/analyst_estimates.py must NOT define a to_features() method. "
        "Analyst data is a decision-time overlay only (Req 2). "
        "Use AnalystEstimateFetcher.render_overlay_summary() for the "
        "report comparator narrative instead."
    )


# ---------------------------------------------------------------------------
# Tests: required terms must appear when their parent module is implemented
# ---------------------------------------------------------------------------

def test_mlconfig_in_engine_config() -> None:
    """MLConfig must be importable and an EngineConfig field (Milestone 0)."""
    from src.config import EngineConfig, MLConfig

    cfg = EngineConfig()
    assert hasattr(cfg, "mlconfig")
    assert isinstance(cfg.mlconfig, MLConfig)


def test_mlconfig_has_no_decision_with_analyst_group() -> None:
    """feature_groups must not contain the rejected 'decision_with_analyst' key."""
    from src.config import MLConfig

    mc = MLConfig()
    assert "decision_with_analyst" not in mc.feature_groups, (
        "feature_groups must not include 'decision_with_analyst'. "
        "Analyst overlay fields are stored separately in "
        "MLConfig.analyst_overlay_fields and never enter the ML matrix."
    )


def test_mlconfig_analyst_overlay_fields_exist() -> None:
    """analyst_overlay_fields must exist as a separate (non-feature-group) attr."""
    from src.config import MLConfig

    mc = MLConfig()
    assert hasattr(mc, "analyst_overlay_fields")
    assert isinstance(mc.analyst_overlay_fields, list)
    assert len(mc.analyst_overlay_fields) > 0


def test_mlconfig_target_definitions_match_spec() -> None:
    """Primary / secondary / tertiary target names match the v2 spec."""
    from src.config import MLConfig

    mc = MLConfig()
    assert mc.target_definitions.get("primary") == "rev_growth_quarterly_YoY"
    assert mc.target_definitions.get("secondary") == "rev_growth_annual_FY"
    assert (
        mc.target_definitions.get("tertiary")
        == "excess_return_12m_vs_spx_direction"
    )


def test_mlconfig_no_gradient_boosting_in_models() -> None:
    """GradientBoosting must be excluded entirely (Req 6.5)."""
    from src.config import MLConfig

    mc = MLConfig()
    all_models = [mc.primary_model, *mc.sensitivity_models]
    forbidden = {
        "gradientboosting",
        "gradient_boosting",
        "gbm",
        "xgboost",
        "lightgbm",
    }
    for m in all_models:
        assert m.lower().replace("-", "_") not in forbidden, (
            f"Model '{m}' is excluded by Req 6.5 (overfitting risk at N≈30)."
        )


def test_mlconfig_adjustment_weights_match_spec() -> None:
    """Adjustment weights are the single source of truth (Req 8.2)."""
    from src.config import MLConfig

    mc = MLConfig()
    assert mc.adjustment_weights["diagnostic"] == 0.0
    assert mc.adjustment_weights["contributing"] == 0.20
    assert mc.adjustment_weights["high_confidence"] == 0.35


def test_mlconfig_walk_forward_embargo_horizon_aware() -> None:
    """Embargo must equal horizon for each target (Req 6.4)."""
    from src.config import MLConfig

    mc = MLConfig()
    for target, params in mc.walk_forward_params.items():
        assert (
            params["embargo_quarters"] == params["target_horizon_quarters"]
        ), (
            f"Target {target}: embargo {params['embargo_quarters']} must "
            f"equal target horizon {params['target_horizon_quarters']} per Req 6.4."
        )


def test_mlconfig_full_feature_columns_excludes_analyst() -> None:
    """full_feature_columns() must not contain any analyst-overlay field."""
    from src.config import MLConfig

    mc = MLConfig()
    full = set(mc.full_feature_columns())
    overlay = set(mc.analyst_overlay_fields)
    intersection = full & overlay
    assert not intersection, (
        f"full_feature_columns() must not contain analyst overlay fields, "
        f"but found: {intersection}"
    )


def test_mlconfig_canonical_dict_is_deterministic() -> None:
    """Two MLConfig() instances produce identical canonical_dict output."""
    from src.config import MLConfig

    a = MLConfig().canonical_dict()
    b = MLConfig().canonical_dict()
    assert a == b


# ---------------------------------------------------------------------------
# Tests: provenance manifest schema (Req 12.5)
# ---------------------------------------------------------------------------

def test_provenance_manifest_schema() -> None:
    """build_provenance_manifest returns all Req 12.5 fields."""
    from src.config import EngineConfig
    from src.ml_v2_provenance import build_provenance_manifest

    cfg = EngineConfig()
    m = build_provenance_manifest(cfg)
    required_keys = {
        "git_commit_sha",
        "git_dirty",
        "git_dirty_files",
        "mlconfig_hash",
        "mlconfig_canonical_json",
        "feature_columns_used",
        "target_definitions",
        "target_definitions_version",
        "panel_csv_path",
        "panel_csv_hash",
        "raw_data_manifest",
        "mlconfig_committed_pre_walk_forward",
        "pipeline_version",
        "python_version",
        "platform",
        "key_package_versions",
        "random_seed",
        "timestamp",
    }
    missing = required_keys - set(m.keys())
    assert not missing, f"Provenance manifest missing fields: {missing}"


def test_provenance_pre_registration_gate_dirty_tree() -> None:
    """When the working tree is dirty, the pre-registration gate is False.

    This is the contract the audit relies on (Req 12.6): tier promotions
    above diagnostic require ``mlconfig_committed_pre_walk_forward=True``.
    """
    from src.config import EngineConfig
    from src.ml_v2_provenance import build_provenance_manifest

    cfg = EngineConfig()
    m = build_provenance_manifest(cfg)
    if m["git_dirty"]:
        assert m["mlconfig_committed_pre_walk_forward"] is False, (
            "When git_dirty=True, the pre-registration gate MUST be False. "
            "This is the audit contract guarding against post-hoc tuning."
        )


# ---------------------------------------------------------------------------
# Tests: shell script parity
# ---------------------------------------------------------------------------

def test_shell_consistency_check_runs() -> None:
    """The shell script must be runnable and exit cleanly on current code."""
    script = REPO_ROOT / "scripts" / "check_v2_consistency.sh"
    if not script.exists():
        pytest.skip("check_v2_consistency.sh not present")
    result = subprocess.run(
        [str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"check_v2_consistency.sh exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
