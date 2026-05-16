#!/usr/bin/env bash
#
# scripts/check_v2_consistency.sh — Task 8.5
#
# Cross-file consistency guardrail for the v2 ML/NLP layer (Lo audit).
# Greps the implemented v2 source code, report templates, and audit
# JSON for stale terms that should NEVER appear after the v2 reconciliation.
#
# Banned terms (banned in implementation code and report templates):
#   - binomial_p_vs_naive   (use mcnemar_p_vs_naive)
#   - bootstrap_ci_low / bootstrap_ci_high  (use residual_band_low / residual_band_high)
#   - bootstrap_target_price_ci  (use bootstrap_residual_band)
#   - decision_with_analyst  (analyst is overlay only; not a feature group)
#   - to_features  (forbidden on AnalystEstimateFetcher)
#   - 90% interval / 90% confidence interval / confidence interval (when
#     applied to bootstrap output)
#   - confidence-band  (use empirical residual band)
#   - analyst_consensus_baseline / analyst_baseline (analyst is not a baseline)
#
# Allowed in spec markdown (negative assertions, comparative prose,
# historical commentary). NOT allowed in implemented Python code or
# rendered report templates / JSON outputs.
#
# Exit code:
#   0 = clean (no banned terms in implemented code)
#   1 = at least one banned term found in implemented code/templates/audit
#
# Usage:
#   ./scripts/check_v2_consistency.sh
#   ./scripts/check_v2_consistency.sh --verbose     # show grep details
#

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

VERBOSE=0
if [[ "${1:-}" == "--verbose" ]]; then
    VERBOSE=1
fi

# Implemented v2 source files (Python). Files do not have to exist yet;
# missing files are skipped silently (early in implementation).
V2_SRC_FILES=(
    "src/ml_decision.py"
    "src/analyst_estimates.py"
    "src/peer_freshness.py"
    "src/quarterly_panel.py"
    "src/market_features.py"
    "src/ml_v2_provenance.py"
)

# v2 modifications to existing modules. Stale terms in these files are
# also banned (the v2 code paths inside these files must be clean).
V2_MOD_FILES=(
    "src/config.py"
    "src/valuation.py"
    "src/charts.py"
    "src/audit_utils.py"
    "src/report_utils.py"
    "scripts/run_pipeline.py"
)

# Report template directory: any rendered output term must be clean.
TEMPLATE_DIR="src/templates"

# Banned terms (regex-friendly; word-boundary handled via grep -w where useful).
BANNED_TERMS=(
    "binomial_p_vs_naive"
    "bootstrap_ci_low"
    "bootstrap_ci_high"
    "bootstrap_target_price_ci"
    "decision_with_analyst"
    "analyst_consensus_baseline"
    "analyst_baseline"
)

# Banned phrases in templates / report output (more permissive matching).
BANNED_PHRASES_IN_TEMPLATES=(
    "90% interval"
    "90% confidence interval"
    "confidence interval"
    "confidence-band"
)

FAIL=0
FAIL_DETAILS=()

check_file() {
    local file="$1"
    local kind="$2"   # "src" or "template"
    [[ -f "${file}" ]] || return 0

    for term in "${BANNED_TERMS[@]}"; do
        # grep -F = fixed string; -n = line number
        if grep -nF "${term}" "${file}" >/dev/null 2>&1; then
            FAIL=1
            local matches
            matches="$(grep -nF "${term}" "${file}" | head -3)"
            FAIL_DETAILS+=("${file}:${term}")
            if [[ "${VERBOSE}" -eq 1 ]]; then
                echo "FAIL ${file}: banned term '${term}'"
                echo "${matches}" | sed 's/^/    /'
            fi
        fi
    done

    if [[ "${kind}" == "template" ]]; then
        for phrase in "${BANNED_PHRASES_IN_TEMPLATES[@]}"; do
            if grep -nF "${phrase}" "${file}" >/dev/null 2>&1; then
                FAIL=1
                local matches
                matches="$(grep -nF "${phrase}" "${file}" | head -3)"
                FAIL_DETAILS+=("${file}:'${phrase}'")
                if [[ "${VERBOSE}" -eq 1 ]]; then
                    echo "FAIL ${file}: banned template phrase '${phrase}'"
                    echo "${matches}" | sed 's/^/    /'
                fi
            fi
        done
    fi

    # AnalystEstimateFetcher must NOT define to_features().
    if [[ "${file}" == *"analyst_estimates.py" ]]; then
        if grep -nE "def[[:space:]]+to_features" "${file}" >/dev/null 2>&1; then
            FAIL=1
            FAIL_DETAILS+=("${file}:to_features method (forbidden on analyst classes)")
            if [[ "${VERBOSE}" -eq 1 ]]; then
                echo "FAIL ${file}: forbidden 'def to_features' method on analyst class"
            fi
        fi
    fi
}

# 1. v2 source files
for f in "${V2_SRC_FILES[@]}"; do
    check_file "${f}" "src"
done

# 2. v2 modifications to existing files
for f in "${V2_MOD_FILES[@]}"; do
    check_file "${f}" "src"
done

# 3. Report templates (Jinja2 / markdown)
if [[ -d "${TEMPLATE_DIR}" ]]; then
    while IFS= read -r -d '' tmpl; do
        check_file "${tmpl}" "template"
    done < <(find "${TEMPLATE_DIR}" -type f \( -name "*.j2" -o -name "*.md" -o -name "*.html" \) -print0)
fi

# 4. audit_status.json — banned keys
AUDIT_JSON="outputs/audit_status.json"
if [[ -f "${AUDIT_JSON}" ]]; then
    for term in "${BANNED_TERMS[@]}"; do
        if grep -nF "\"${term}\"" "${AUDIT_JSON}" >/dev/null 2>&1; then
            FAIL=1
            FAIL_DETAILS+=("${AUDIT_JSON}:key '${term}'")
            if [[ "${VERBOSE}" -eq 1 ]]; then
                echo "FAIL ${AUDIT_JSON}: banned audit key '${term}'"
            fi
        fi
    done
fi

if [[ "${FAIL}" -eq 0 ]]; then
    echo "OK: v2 cross-file consistency check passed (no banned terms in implemented code/templates/audit)."
    exit 0
else
    echo "FAIL: v2 consistency check found banned terms:"
    for d in "${FAIL_DETAILS[@]}"; do
        echo "  - ${d}"
    done
    echo
    echo "Run with --verbose to see line numbers and surrounding context."
    exit 1
fi
