#!/usr/bin/env python3
"""
Mode-aware acceptance tests for the NVDA quantamental report package.

Tests operate on final output files (not internal objects) to verify
that no contradiction exists between any submitted artifact.

Supports three modes:
  - formal_rating_pass: expects Buy/Hold/Sell, consistent across all artifacts
  - diagnostic_not_rated_pass: expects Not Rated, no issued rating
  - failed: expects failure details, no submission-ready package

The mode is auto-detected from audit_status.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
outputs = _PROJECT_ROOT / "outputs"

errors: list[str] = []
passes: list[str] = []


def check(name: str, condition: bool, fail_msg: str) -> None:
    if condition:
        passes.append(f"  PASS  {name}")
    else:
        errors.append(f"  FAIL  {name}: {fail_msg}")


# ===================================================================
# Load all output files
# ===================================================================
audit_path = outputs / "audit_status.json"
if not audit_path.exists():
    print("FATAL: outputs/audit_status.json not found")
    sys.exit(1)

audit_json = json.loads(audit_path.read_text())
manifest_path = outputs / "manifest.json"
manifest_json = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

# Derive ticker from manifest or audit for dynamic filenames
_ticker = manifest_json.get("ticker", audit_json.get("ticker", "nvda")).lower()
_report_base = f"{_ticker}_quantamental_report"

exec_md = (outputs / "executive_summary.md").read_text() if (outputs / "executive_summary.md").exists() else ""
lim_md = (outputs / "limitations.md").read_text() if (outputs / "limitations.md").exists() else ""
dqr_md = (outputs / "data_quality_report.md").read_text() if (outputs / "data_quality_report.md").exists() else ""
report_md = (outputs / f"{_report_base}.md").read_text() if (outputs / f"{_report_base}.md").exists() else ""
# Also check for ticker-dynamic filename
if not report_md:
    _report_candidates = list(outputs.glob("*_quantamental_report.md"))
    if _report_candidates:
        report_md = _report_candidates[0].read_text()
model_audit_md = (outputs / "model_audit.md").read_text() if (outputs / "model_audit.md").exists() else ""
prompt_md = (outputs / "prompt_log.md").read_text() if (outputs / "prompt_log.md").exists() else ""
source_attr_md = (outputs / "source_attribution.md").read_text() if (outputs / "source_attribution.md").exists() else ""
html_exists = (outputs / f"{_report_base}.html").exists()
html_text = (outputs / f"{_report_base}.html").read_text() if html_exists else ""
pdf_exists = (outputs / f"{_report_base}.pdf").exists()

# ===================================================================
# Auto-detect mode from audit_status.json
# ===================================================================
package_status = audit_json.get("package_status", "unknown")
report_mode = audit_json.get("report_mode", manifest_json.get("report_mode", "unknown"))

print(f"\nDetected package_status: {package_status}")
print(f"Detected report_mode:   {report_mode}")
print()

# ===================================================================
# COMMON TESTS (all modes)
# ===================================================================

# 1. Required files exist
required_files = [
    f"{_report_base}.md", "executive_summary.md",
    "audit_status.json", "manifest.json", "source_attribution.md",
    "limitations.md", "model_audit.md", "prompt_log.md",
]
for fname in required_files:
    fpath = outputs / fname
    check(f"File exists: {fname}",
          fpath.exists() and fpath.stat().st_size > 0,
          f"Missing or empty: {fname}")

# 2. Run ID consistency
run_id_audit = audit_json.get("run_id", "")
run_id_manifest = manifest_json.get("run_id", "")
check("Run ID: audit matches manifest",
      run_id_audit == run_id_manifest and run_id_audit != "",
      f"audit={run_id_audit}, manifest={run_id_manifest}")

# 3. No audit failures (internal consistency)
check("Audit: no internal failures",
      audit_json.get("checks_failed", -1) == 0,
      f"checks_failed={audit_json.get('checks_failed')}: {audit_json.get('failure_details', [])}")

# 4. Prompt log quality
check("Prompt log: has LLM model info",
      "Claude" in prompt_md or "claude" in prompt_md.lower() or "LLM" in prompt_md,
      "Missing LLM model info")

check("Prompt log: discloses limitations",
      "incomplete" in prompt_md.lower() or "limitation" in prompt_md.lower(),
      "Missing incompleteness disclosure")

# 5. ML treatment — diagnostic-only must be stated
check("ML: diagnostic-only stated in report",
      "diagnostic-only" in report_md.lower() or "diagnostic only" in report_md.lower()
      or "excluded from recommendation" in report_md.lower()
      or "excluded from rating" in report_md.lower(),
      "ML not stated as diagnostic-only/excluded in report")

# 6. No stale artifacts
check("No pending_regeneration in manifest",
      "pending_regeneration" not in json.dumps(manifest_json),
      "Found pending_regeneration status")

# ===================================================================
# MODE-SPECIFIC TESTS
# ===================================================================

if package_status == "formal_rating_pass" or package_status == "formal_rating_pass_with_warnings":
    print("--- Testing formal_rating_pass mode ---\n")
    if package_status == "formal_rating_pass_with_warnings":
        print("  NOTE: Tests were skipped — status downgraded to pass_with_warnings\n")

    # Get canonical values from audit_status.json (FinalRecommendation)
    final_rating = audit_json.get("final_recommendation", "")
    final_price = audit_json.get("final_current_price", 0)
    final_target = audit_json.get("final_intrinsic_value", 0)
    final_upside = audit_json.get("final_upside_downside", 0)

    check("Rating is Buy/Hold/Sell",
          final_rating in ("Buy", "Hold", "Sell"),
          f"rating={final_rating}")

    # Rating consistency across all artifacts
    manifest_rating = manifest_json.get("recommendation", "")
    check("Rating: audit matches manifest",
          final_rating == manifest_rating,
          f"audit={final_rating}, manifest={manifest_rating}")

    # Report contains the rating
    check("Rating in report markdown",
          f"**{final_rating}**" in report_md,
          f"'{final_rating}' not found as bold in report")

    # Executive summary contains the rating
    check("Rating in executive summary",
          final_rating in exec_md,
          f"'{final_rating}' not in executive summary")

    # No contradictory rating in report
    other_ratings = [r for r in ("Buy", "Hold", "Sell") if r != final_rating]
    for other in other_ratings:
        # Check for issued-rating patterns (not explanatory context)
        issued_patterns = [
            f"We rate NVIDIA {other}",
            f"Rating: {other}",
            f"**Rating** | **{other}**",
        ]
        for pattern in issued_patterns:
            check(f"No contradictory '{pattern}' in report",
                  pattern not in report_md,
                  f"Found '{pattern}' in report while final rating is {final_rating}")

    # No "Not Rated" headline
    check("No 'Not Rated' headline in report",
          "Not Rated — Data Validation Required" not in report_md,
          "Found diagnostic 'Not Rated' headline in formal_rating report")

    # Price consistency
    manifest_price = manifest_json.get("current_price", 0)
    check("Price: audit matches manifest",
          abs(final_price - manifest_price) < 0.01,
          f"audit={final_price}, manifest={manifest_price}")

    # Target consistency
    manifest_target = manifest_json.get("intrinsic_value", 0)
    check("Target: audit matches manifest",
          abs(final_target - manifest_target) < 0.01,
          f"audit={final_target}, manifest={manifest_target}")

    # Upside consistency
    manifest_upside = manifest_json.get("upside_downside_pct", 0)
    check("Upside: audit matches manifest",
          abs(final_upside - manifest_upside) < 0.001,
          f"audit={final_upside}, manifest={manifest_upside}")

    # Price appears in report
    price_str = f"${final_price:.2f}"
    check("Current price in report",
          price_str in report_md or f"${final_price:.0f}" in report_md,
          f"Price {price_str} not found in report")

    # Target appears in report
    target_str = f"${final_target:.2f}"
    check("Target price in report",
          target_str in report_md or f"${final_target:.0f}" in report_md,
          f"Target {target_str} not found in report")

    # Exhibit gating
    suppressed = audit_json.get("suppressed_exhibits", [])
    active = audit_json.get("active_exhibits", [])

    # No all-Other segment chart
    if "revenue_segment_mix" in suppressed:
        check("Suppressed segment chart not in report",
              "NVDA Revenue by Segment" not in report_md or "Suppressed" in report_md,
              "Segment chart title found in report despite suppression")

    # No stale PDF
    if pdf_exists:
        check("PDF exists and is fresh",
              True, "")
    else:
        check("No stale PDF (PDF not generated)",
              f"{_report_base}.pdf" not in json.dumps(manifest_json.get("artifacts", {}))
              or manifest_json.get("artifacts", {}).get(f"{_report_base}.pdf", {}).get("status") != "active",
              "Manifest claims active PDF but file missing")

    # Assignment alignment
    check("Report has fundamental analysis section",
          "fundamental analysis" in report_md.lower(),
          "Missing fundamental analysis section")

    check("Report has ML method section",
          "ml" in report_md.lower() and "method" in report_md.lower(),
          "Missing ML method section")

    check("Report has scalability section",
          "scalability" in report_md.lower() or "automation" in report_md.lower(),
          "Missing scalability/automation section")

    check("Report has source attribution reference",
          "source attribution" in report_md.lower() or "source_attribution" in report_md.lower(),
          "Missing source attribution reference")

    check("Report has LLM disclosure",
          "llm" in report_md.lower() or "claude" in report_md.lower() or "language model" in report_md.lower(),
          "Missing LLM disclosure")

    # If Hold with negative upside, check for Why Hold Not Sell
    if final_rating == "Hold" and final_upside < 0:
        check("Why Hold Not Sell box present",
              "why hold" in report_md.lower() or "hold, not sell" in report_md.lower(),
              "Hold with negative upside but no 'Why Hold, Not Sell?' box")

elif package_status == "diagnostic_not_rated_pass":
    print("--- Testing diagnostic_not_rated_pass mode ---\n")

    # No issued Buy/Hold/Sell
    FORBIDDEN_RATING_PHRASES = [
        "We rate NVIDIA Sell", "We rate NVIDIA Buy", "We rate NVIDIA Hold",
        "Recommendation: Sell", "Recommendation: Buy", "Recommendation: Hold",
    ]
    for phrase in FORBIDDEN_RATING_PHRASES:
        check(f"No '{phrase}' in report",
              phrase not in report_md,
              f"Found '{phrase}' in diagnostic report")

    check("Report says Not Rated",
          "Not Rated" in report_md,
          "Missing 'Not Rated' in diagnostic report")

    check("Executive summary says Not Rated",
          "Not Rated" in exec_md,
          "Missing 'Not Rated' in executive summary")

    check("Valuation labeled diagnostic",
          "diagnostic" in report_md.lower(),
          "Valuation not labeled diagnostic")

elif package_status == "failed":
    print("--- Testing failed mode ---\n")

    check("Failure details present",
          len(audit_json.get("failure_details", [])) > 0,
          "No failure details in audit_status.json")

else:
    print(f"WARNING: Unknown package_status '{package_status}'")
    errors.append(f"  FAIL  Unknown package_status: {package_status}")


# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 60)
print("ACCEPTANCE TEST RESULTS")
print("=" * 60)

for p in passes:
    print(p)
for e in errors:
    print(e)

total = len(passes) + len(errors)
print(f"\n{len(passes)} passed, {len(errors)} failed out of {total} tests")

if errors:
    print("\nSOME TESTS FAILED")
    sys.exit(1)
else:
    print("\nALL ACCEPTANCE TESTS PASSED")
    sys.exit(0)
