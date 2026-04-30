# Regression Baseline: Original Pipeline Failure

## Failure Summary

**Date captured:** Snapshot of pipeline outputs from initial run (report date 2026-04-29)

**Root cause:** The XBRL parser confused a quarterly/YTD value with the annual total for FY2025 revenue.

| Metric | Parsed (Broken) | Published (Correct) | Diff % |
|--------|-----------------|---------------------|--------|
| FY2025 Revenue | $26.974B | $130.497B | −79.3% |

The parser selected a quarterly or YTD fact (~$27B) instead of the full fiscal-year annual revenue (~$130.5B) because it lacked fiscal-period-type classification and annual-duration filtering.

## Cascading Impact

1. **Valuation:** DCF used ~$27B as base revenue → target price of $47.56 (vs $213.17 market price)
2. **Recommendation:** Sell with −77.7% downside — catastrophically wrong
3. **Limitations file:** Stated "No data quality issues recorded" despite 15/24 validation failures
4. **Data quality report:** Showed massive failures (FY2025 revenue 79.33% off, FY2024 revenue 55.82% off, etc.)
5. **Contradiction:** limitations.md and data_quality_report.md directly contradicted each other

## Additional Validation Failures (from data_quality_report.md)

| Metric | FY | Parsed | Published | Diff % | Status |
|--------|-----|--------|-----------|--------|--------|
| revenue | 2025 | $26.974B | $130.497B | 79.33% | ❌ fail |
| revenue | 2024 | $26.914B | $60.922B | 55.82% | ❌ fail |
| revenue | 2023 | $16.675B | $26.974B | 38.18% | ❌ fail |
| net_income | 2025 | $4.368B | $72.880B | 94.01% | ❌ fail |
| net_income | 2024 | $9.752B | $29.760B | 67.23% | ❌ fail |
| operating_cash_flow | 2025 | $5.641B | $64.089B | 91.20% | ❌ fail |
| operating_cash_flow | 2024 | $9.108B | $28.090B | 67.58% | ❌ fail |
| diluted_shares | 2024 | 2.535B | 24.940B | 89.84% | ❌ fail |
| diluted_shares | 2023 | 2.510B | 24.960B | 89.94% | ❌ fail |
| r_and_d | 2025 | $7.339B | $12.893B | 43.08% | ❌ fail |
| r_and_d | 2024 | $5.268B | $8.675B | 39.27% | ❌ fail |
| r_and_d | 2023 | $3.924B | $7.339B | 46.53% | ❌ fail |
| diluted_eps | 2025 | $0 | $3 | 94.22% | ❌ fail |
| diluted_eps | 2024 | $4 | $12 | 67.73% | ❌ fail |

**Summary:** 3/24 passed, 15 failed, 6 missing

## Files in This Directory

| File | Source | Description |
|------|--------|-------------|
| `broken_report_v1.md` | `outputs/nvda_quantamental_report.md` | Full report with wrong Sell rating |
| `broken_dqr_v1.md` | `outputs/data_quality_report.md` | Data quality report showing validation failures |
| `broken_limitations_v1.md` | `outputs/limitations.md` | Limitations file that incorrectly says "No data quality issues" |
| `broken_metrics_v1.csv` | `data/processed/nvda_metrics.csv` | Metrics CSV with wrong parsed values |

## Purpose

These files serve as regression baselines to ensure:
1. The fiscal-year selection fix correctly parses FY2025 revenue as ~$130.5B
2. The data validation gate blocks formal recommendations when validation fails
3. The limitations.md generator never says "No data quality issues" when failures exist
4. The audit consistency checker catches contradictions between output files
