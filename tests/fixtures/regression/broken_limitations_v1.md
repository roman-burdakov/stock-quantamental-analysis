# Limitations

*Generated: 2026-04-29 22:59*

## Data Gaps and Quality Issues

No data quality issues recorded.

## Model Limitations

- ML driver model uses ElasticNet/Ridge regression — limited non-linear capture
- Walk-forward validation with limited historical periods may overfit
- NLP features based on TF-IDF keyword scoring — no semantic understanding
- DCF valuation relies on analyst-assumed growth rates and margins
- Peer multiples subject to comparability limitations across business models
- **Peer-panel ML (17.5):** The `train_peer_panel_model()` method in `src/ml_models.py` enables cross-sectional learning by pooling NVDA + semiconductor peer features. However, it requires 10-year peer historicals in the same feature format (fiscal-period-aligned financial metrics and NLP features) as NVDA. Currently, peer financials are snapshot-based (from yfinance) and lack the historical depth and granularity needed. To fully enable this feature, one would need to run the same XBRL parsing and metric computation pipeline for each peer company (AMD, AVGO, INTC, QCOM, MRVL) across the same fiscal year range.

## Assumptions

- WACC = 10.0% (analyst judgment)
- Terminal growth = 3.0% (analyst judgment)
- Projection horizon = 10 years
- Report date = 2026-04-29 (frozen analysis cutoff)
- Price date = 2026-04-29
- SBC treatment: included in FCF (not adjusted out)
- Peer staleness threshold: 90 days

## Unresolved Blockers

- No unresolved blockers.

## PDF Generation

PDF output was not generated. The `weasyprint` library is either not installed or encountered an error. An HTML fallback (`nvda_quantamental_report.html`) has been produced instead. Install `weasyprint` and its system dependencies for PDF output.
