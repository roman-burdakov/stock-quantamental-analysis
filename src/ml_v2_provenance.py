"""
v2 ML Pre-Registration Provenance Manifest (Req 12.5, Task 0.2)

This module writes ``outputs/ml_config_provenance.json`` at the start of
the ml_v2 pipeline stage. The manifest records the complete state of the
codebase, configuration, input data, and runtime environment at the
moment walk-forward validation begins.

The audit-consistency check (Req 12.6) verifies that
``mlconfig_committed_pre_walk_forward`` is True for any v2 walk-forward
result that promotes a target above diagnostic tier; otherwise the tier
is forcibly downgraded to diagnostic with reason
``pre_registration_violation``.

Why this exists:
- Pre-registration is the standard discipline against post-hoc tuning.
- A bare git SHA is not enough: feature engineering code, target
  construction logic, panel CSV contents, and Python package versions
  can all change in ways that affect statistical claims.
- The manifest hashes everything that matters and freezes the state.

Usage:

    from src.ml_v2_provenance import write_provenance_manifest
    from src.config import EngineConfig

    cfg = EngineConfig()
    manifest = write_provenance_manifest(cfg)
    if not manifest["mlconfig_committed_pre_walk_forward"]:
        logger.warning(
            "MLConfig was modified in working tree or not committed "
            "before walk-forward; tier promotions will be downgraded."
        )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import EngineConfig, MLConfig

logger = logging.getLogger(__name__)


PIPELINE_VERSION = "2.0.0-mvp-essential"


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stripped stdout, or empty string on error."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("git command failed: %s (%s)", args, e)
        return ""


def _git_head_sha(repo_root: Path) -> str:
    """Return the current HEAD commit SHA (full)."""
    return _run_git(["rev-parse", "HEAD"], repo_root)


def _git_dirty_status(repo_root: Path) -> tuple[bool, list[str]]:
    """Return (dirty: bool, list of dirty file paths).

    For pre-registration purposes, the working tree is "dirty" if either:
    - Any tracked file has uncommitted modifications or staged changes, OR
    - There are any UNTRACKED files matching ``src/*.py`` or
      ``scripts/*.py`` (new source code that has not been committed yet
      could affect ML behaviour).

    Generated data files (``data/processed/*.csv`` etc.) and gitignored
    paths are intentionally NOT counted, because pre-registration is a
    code-state question, not a data-state question.

    The git porcelain v1 format is ``XY filename`` where XY is a
    2-character status code. We must NOT strip the output: the leading
    space in codes like " M" or "??" is part of the format.
    """
    try:
        # untracked-files=normal so we can see new src/scripts files,
        # but we filter the untracked entries ourselves below.
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("git status failed: %s", e)
        return (False, [])

    output = result.stdout  # do NOT strip — leading whitespace is meaningful
    if not output.strip():
        return (False, [])

    dirty_files: list[str] = []
    for line in output.splitlines():
        if len(line) <= 3:
            continue
        status_code = line[:2]
        path = line[3:]
        # Tracked-file modifications: any non-"??" status code is dirty.
        if status_code != "??":
            dirty_files.append(path)
            continue
        # Untracked files: only count Python source under src/ or scripts/.
        # Trailing slash (directory) is treated as one entry covering all
        # untracked Python files inside it.
        if path.endswith("/"):
            # Untracked directory — descend lazily by checking if it contains
            # any *.py file. For simplicity, treat any untracked directory
            # under src/ or scripts/ as dirty.
            if path.startswith("src/") or path.startswith("scripts/"):
                dirty_files.append(path)
        else:
            if (
                (path.startswith("src/") and path.endswith(".py"))
                or (path.startswith("scripts/") and path.endswith((".py", ".sh")))
            ):
                dirty_files.append(path)

    return (len(dirty_files) > 0, dirty_files)


def _last_commit_for_file(repo_root: Path, relative_path: str) -> str:
    """Return the SHA of the most recent commit that modified the file."""
    return _run_git(
        ["log", "-n", "1", "--pretty=format:%H", "--", relative_path],
        repo_root,
    )


def _mlconfig_committed_pre_walk_forward(
    repo_root: Path,
    head_sha: str,
    git_dirty: bool,
) -> bool:
    """Return True iff src/config.py (which contains MLConfig) is fully
    committed and was last modified in a commit at or before HEAD.

    This is the operationalisation of Req 12.6: tier promotions above
    diagnostic require this to be True.
    """
    if git_dirty:
        return False
    # If config.py was last modified in HEAD or an ancestor commit, we are
    # OK. The simplest check: the file has no uncommitted changes (already
    # implied by git_dirty=False) and the last commit touching it is
    # reachable from HEAD.
    last_commit = _last_commit_for_file(repo_root, "src/config.py")
    if not last_commit:
        return False
    # Verify last_commit is an ancestor of HEAD (or equal to HEAD).
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", last_commit, head_sha],
        cwd=str(repo_root),
        capture_output=True,
        check=False,
    )
    # Exit status 0 = ancestor, 1 = not ancestor, other = error.
    return result.returncode == 0


def _sha256_of_text(text: str) -> str:
    """Return hex SHA256 of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_of_file(path: Path) -> Optional[str]:
    """Return hex SHA256 of file contents, or None if file does not exist."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _mlconfig_canonical_json(mlconfig: MLConfig) -> str:
    """Return canonical-JSON-serialised MLConfig for stable hashing.

    Uses the dataclass's `canonical_dict()` method which returns a sorted,
    deterministic representation. JSON is dumped with sort_keys=True and
    no ASCII-escape so byte content is reproducible across runs.
    """
    return json.dumps(
        mlconfig.canonical_dict(),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _hash_mlconfig(mlconfig: MLConfig) -> str:
    return _sha256_of_text(_mlconfig_canonical_json(mlconfig))


def _build_raw_data_manifest(cfg: EngineConfig) -> dict[str, Any]:
    """List input files with size and mtime for reproducibility.

    The list is intentionally narrow: only files that the v2 ml stage
    consumes. Adding files outside this list does not invalidate the
    manifest.
    """
    candidates: list[Path] = [
        cfg.raw_dir / f"companyfacts_CIK{cfg.cik}.json",
        cfg.raw_dir / f"submissions_CIK{cfg.cik}.json",
        cfg.raw_dir / "market_prices.csv",
        cfg.raw_dir / "peer_financials.csv",
        cfg.processed_dir / "nvda_metrics.csv",
        cfg.processed_dir / "nvda_nlp_features.csv",
    ]
    entries: list[dict[str, Any]] = []
    for p in candidates:
        if p.exists() and p.is_file():
            stat = p.stat()
            entries.append({
                "path": str(p),
                "size_bytes": stat.st_size,
                "mtime_iso": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            })
    # Sort for deterministic ordering
    entries.sort(key=lambda e: e["path"])
    serialized = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "files": entries,
        "manifest_hash": _sha256_of_text(serialized),
    }


def _key_package_versions() -> dict[str, str]:
    """Return versions of packages whose behaviour affects ML outputs."""
    versions: dict[str, str] = {}
    packages = [
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "yfinance",
        "matplotlib",
    ]
    try:
        from importlib.metadata import PackageNotFoundError, version

        for pkg in packages:
            try:
                versions[pkg] = version(pkg)
            except PackageNotFoundError:
                versions[pkg] = "not_installed"
    except ImportError:
        versions["error"] = "importlib.metadata unavailable"
    return versions


def build_provenance_manifest(
    cfg: EngineConfig,
    panel_csv_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Build the pre-registration provenance manifest dict (Req 12.5).

    Parameters
    ----------
    cfg : EngineConfig
        The active engine configuration. cfg.mlconfig is hashed.
    panel_csv_path : Path, optional
        Path to ``data/processed/ml_quarterly_panel.csv``. If the file
        does not yet exist (e.g., first-run pre-registration before the
        panel is built), the hash is None and the audit will record this.
    repo_root : Path, optional
        Override the git repo root. Defaults to the current working
        directory's git toplevel.

    Returns
    -------
    dict
        The manifest dict, ready to be written to JSON.
    """
    if repo_root is None:
        repo_root = Path(_run_git(["rev-parse", "--show-toplevel"], Path.cwd()))
        if not repo_root or not repo_root.exists():
            repo_root = Path.cwd()

    head_sha = _git_head_sha(repo_root)
    git_dirty, dirty_files = _git_dirty_status(repo_root)
    pre_reg_ok = _mlconfig_committed_pre_walk_forward(
        repo_root, head_sha, git_dirty
    )

    if panel_csv_path is None:
        panel_csv_path = cfg.processed_dir / "ml_quarterly_panel.csv"

    manifest: dict[str, Any] = {
        # Git
        "git_commit_sha": head_sha,
        "git_dirty": git_dirty,
        "git_dirty_files": dirty_files,
        # MLConfig
        "mlconfig_hash": _hash_mlconfig(cfg.mlconfig),
        "mlconfig_canonical_json": _mlconfig_canonical_json(cfg.mlconfig),
        "feature_columns_used": {
            "fundamentals": list(cfg.mlconfig.feature_groups.get("fundamentals", [])),
            "market": list(cfg.mlconfig.feature_groups.get("market", [])),
            "nlp": list(cfg.mlconfig.feature_groups.get("nlp", [])),
            "full_no_analyst": cfg.mlconfig.full_feature_columns(),
        },
        "target_definitions": dict(cfg.mlconfig.target_definitions),
        "target_definitions_version": cfg.mlconfig.target_definitions_version,
        # Panel
        "panel_csv_path": str(panel_csv_path),
        "panel_csv_hash": _sha256_of_file(panel_csv_path),
        # Raw data
        "raw_data_manifest": _build_raw_data_manifest(cfg),
        # Pre-registration verdict (the audit-checked field)
        "mlconfig_committed_pre_walk_forward": pre_reg_ok,
        # Pipeline / runtime
        "pipeline_version": PIPELINE_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "key_package_versions": _key_package_versions(),
        "random_seed": cfg.mlconfig.random_seed,
        # Timing
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    return manifest


def write_provenance_manifest(
    cfg: EngineConfig,
    output_path: Optional[Path] = None,
    panel_csv_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Build and write the manifest to ``outputs/ml_config_provenance.json``.

    Returns the manifest dict so callers can immediately inspect
    ``mlconfig_committed_pre_walk_forward`` and warn if False.
    """
    manifest = build_provenance_manifest(
        cfg, panel_csv_path=panel_csv_path, repo_root=repo_root
    )
    if output_path is None:
        output_path = cfg.outputs_dir / "ml_config_provenance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(
        "Pre-registration manifest written to %s (committed=%s, dirty=%s)",
        output_path,
        manifest["mlconfig_committed_pre_walk_forward"],
        manifest["git_dirty"],
    )
    return manifest
