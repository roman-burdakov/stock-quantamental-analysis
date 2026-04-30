# Data Quality Report

**Ticker:** NVDA
**Report Date:** 2026-04-29

## Missing XBRL Tags

- **total_debt**: no matching XBRL concept found

## Fallback Tags Used

| Metric | Primary Concept | Fallback Used |
|--------|----------------|---------------|
| cash_and_securities | CashCashEquivalentsAndShortTermInvestments | CashAndCashEquivalentsAtCarryingValue |
| short_term_debt | ShortTermBorrowings | DebtCurrent |

## Coverage

- Metrics defined: 21
- Metrics found: 20
- Coverage: 95.2%

- Fiscal years covered: [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
- Total parsed facts: 1037

## Validation Against Published Values

| Metric | FY | Parsed | Published | Diff % | Status |
|--------|----|--------|-----------|--------|--------|
| revenue | 2024 | 26,914,000,000 | 60,922,000,000 | 55.82% | ❌ fail |
| net_income | 2024 | 9,752,000,000 | 29,760,000,000 | 67.23% | ❌ fail |
| operating_cash_flow | 2024 | 9,108,000,000 | 28,090,000,000 | 67.58% | ❌ fail |
| capex | 2024 | N/A | 1,069,000,000 | N/A | ⚠️ missing |
| diluted_eps | 2024 | 4 | 12 | 67.73% | ❌ fail |
| total_debt | 2024 | N/A | 9,709,000,000 | N/A | ⚠️ missing |
| r_and_d | 2024 | 5,268,000,000 | 8,675,000,000 | 39.27% | ❌ fail |
| diluted_shares | 2024 | 2,535,000,000 | 24,940,000,000 | 89.84% | ❌ fail |
| revenue | 2025 | 26,974,000,000 | 130,497,000,000 | 79.33% | ❌ fail |
| net_income | 2025 | 4,368,000,000 | 72,880,000,000 | 94.01% | ❌ fail |
| operating_cash_flow | 2025 | 5,641,000,000 | 64,089,000,000 | 91.20% | ❌ fail |
| capex | 2025 | N/A | 3,233,000,000 | N/A | ⚠️ missing |
| diluted_eps | 2025 | 0 | 3 | 94.22% | ❌ fail |
| total_debt | 2025 | N/A | 8,462,000,000 | N/A | ⚠️ missing |
| r_and_d | 2025 | 7,339,000,000 | 12,893,000,000 | 43.08% | ❌ fail |
| diluted_shares | 2025 | 25,070,000,000 | 24,780,000,000 | 1.17% | ✅ pass |
| revenue | 2023 | 16,675,000,000 | 26,974,000,000 | 38.18% | ❌ fail |
| net_income | 2023 | 4,332,000,000 | 4,368,000,000 | 0.82% | ✅ pass |
| operating_cash_flow | 2023 | 5,822,000,000 | 5,641,000,000 | 3.21% | ❌ fail |
| capex | 2023 | N/A | 976,000,000 | N/A | ⚠️ missing |
| diluted_eps | 2023 | 2 | 2 | 0.57% | ✅ pass |
| total_debt | 2023 | N/A | 10,953,000,000 | N/A | ⚠️ missing |
| r_and_d | 2023 | 3,924,000,000 | 7,339,000,000 | 46.53% | ❌ fail |
| diluted_shares | 2023 | 2,510,000,000 | 24,960,000,000 | 89.94% | ❌ fail |

**Summary:** 3/24 passed, 15 failed, 6 missing

## Segment Revenue Normalization

**Ticker:** NVDA

### Label Mapping

| Original Label | Normalized Category |
|---------------|-------------------|
| all other | oem_and_other |
| automotive | automotive |
| compute & networking | compute_and_networking |
| compute and networking | compute_and_networking |
| data center | data_center |
| datacenter | data_center |
| gaming | gaming |
| gpu | gpu |
| gpu business | gpu |
| graphics | graphics |
| oem & other | oem_and_other |
| oem and ip | oem_and_other |
| oem and other | oem_and_other |
| professional visualization | professional_visualization |
| proviz | professional_visualization |
| tegra processor | tegra |
| tegra processor business | tegra |

### Extraction Methods Used

| Fiscal Year | Period | Method | Records |
|------------|--------|--------|---------|
| 2019 | FY | xbrl_dimension | 1 |
| 2020 | FY | xbrl_dimension | 1 |
| 2021 | FY | xbrl_dimension | 1 |
| 2022 | FY | xbrl_dimension | 1 |

### Unmapped / Changed Labels

- **segment_revenue** (FY2019, FY): unmapped label: 'segment_revenue'
- **segment_revenue** (FY2020, FY): unmapped label: 'segment_revenue'
- **segment_revenue** (FY2021, FY): unmapped label: 'segment_revenue'
- **segment_revenue** (FY2022, FY): unmapped label: 'segment_revenue'

## Filing Text Extraction Coverage

| accession     | form_type   | section      |   char_count | parse_status   | warning           |
|:--------------|:------------|:-------------|-------------:|:---------------|:------------------|
| test-coverage | 10-K        | business     |            0 | missing        | Section not found |
| test-coverage | 10-K        | risk_factors |            0 | missing        | Section not found |
| test-coverage | 10-K        | mda          |            0 | missing        | Section not found |
| test-coverage | 10-K        | quant        |            0 | missing        | Section not found |
