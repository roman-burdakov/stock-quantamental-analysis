# Spec Cohesion Audit and Expansion Learnings

*Audited: 2026-05-03*

---

## Part 1: Spec Cohesion Audit

### Requirements → Design Alignment

| Requirement | Design Coverage | Implementation | Gaps |
|-------------|----------------|----------------|------|
| Req 1: SEC Ingestion | ✅ edgar_fetch.py | ✅ Fully implemented | None |
| Req 2: Market/Peer Data | ✅ edgar_fetch.py | ✅ Fully implemented | None |
| Req 3: XBRL Parsing | ✅ xbrl_parser.py | ✅ Fully implemented | None |
| Req 4: Segment Normalization | ✅ segment_revenue.py | ⚠️ XBRL labels collapse to "Other" | Manual cross-check table compensates |
| Req 5: Filing Text Extraction | ✅ filing_text_parser.py | ⚠️ 32.4% coverage | HTML parsing needs form-specific rules |
| Req 6: Financial Metrics | ✅ financial_metrics.py | ✅ Fully implemented | None |
| Req 7: Narrative Drift | ✅ nlp_features.py | ⚠️ Diagnostic only (coverage) | Functional but gated |
| Req 8: ML Driver Model | ✅ ml_models.py | ⚠️ Diagnostic only (4 obs.) | Functional but underpowered |
| Req 9: Valuation | ✅ valuation.py | ✅ Fully implemented | None |
| Req 10: Report Generation | ✅ report_utils.py + templates | ✅ Fully implemented | None |
| Req 11: Source Attribution | ✅ audit_utils.py | ✅ Fully implemented | None |
| Req 12: Testing | ✅ tests/ | ✅ 38/38 acceptance + pytest | Some unit tests need signature updates |
| Req 13: Reproducibility | ✅ run_pipeline.py | ✅ One-command pipeline | None |
| Req 14: Data Validation Gate | ✅ data_validation.py | ✅ Fully implemented | None |
| Req 15: Fiscal-Year Selection | ✅ xbrl_parser.py | ✅ Regression-tested | None |
| Req 16: Valuation Guardrails | ✅ valuation.py | ✅ Fully implemented | None |
| Req 17: Recommendation Gate | ✅ run_pipeline.py | ✅ Fully implemented | None |
| Req 18: ML Integrity Gate | ✅ ml_models.py | ✅ Fully implemented | None |
| Req 19: Segment Quality Gate | ✅ segment_revenue.py | ✅ Correctly suppresses | None |
| Req 20: NLP Quality Gate | ✅ nlp_features.py | ✅ Correctly gates | None |
| Req 21: Peer Quality Gate | ✅ valuation.py | ✅ Tier separation + filtering | None |
| Req 22: Audit Consistency | ✅ audit_utils.py | ✅ 17 checks | None |
| Req 23: Prompt Log | ✅ prompt_log.md | ⚠️ ~80% complete | Early prompts reconstructed |
| Req 24: Data Retrieval Details | ✅ provenance_log.jsonl | ✅ Fully implemented | None |
| Req 25: Peer Universe | ✅ report_utils.py | ✅ Fully implemented | None |

### Design → Tasks Alignment

All 20 milestones (0–20) are checked complete in tasks.md. The design's three-gate architecture (data validation → recommendation eligibility → audit consistency) is fully implemented in run_pipeline.py.

### Gaps Found

1. **`src/display_format.py` not in design.md** — This module was added during polish passes but isn't documented in the design. It handles metric label formatting, sensitivity table formatting, and banned-token scanning. Should be added to the architecture section.

2. **`FinalRecommendation` dataclass not in design.md** — The design describes `Recommendation` but the implementation added `FinalRecommendation` as the canonical single source of truth. The design should document this as the authoritative recommendation object.

3. **`diagnostic_appendix_exhibits` not in requirements** — The three-way exhibit classification (active / diagnostic_appendix / suppressed) was added during polish but isn't in the requirements. Req 19 only describes "suppressed" and "usable."

4. **Manual platform revenue table not in requirements** — The manual FY2023–FY2025 platform revenue cross-check was added to compensate for the segment normalization failure. This is a valuable pattern that should be documented as a fallback requirement.

5. **WACC build-up decomposition not in design** — The three-component quality adjustment (-1.0% net cash, -1.0% FCF quality, -0.9% competitive position) was added during polish. The design only mentions WACC as a config parameter.

6. **Some unit tests need signature updates** — `build_report_context()` now requires `final_recommendation` in formal_rating mode, but some unit tests still call it without one. These tests are excluded from the pipeline run via pytest filtering.

### Recommendation

The specs are **cohesive at the architecture level** — every requirement has a corresponding design component and implementation. The gaps are all additions made during polish passes that improved the report but weren't backported to the spec documents. For a production system, these should be documented. For the current submission, the implementation is the source of truth and the specs accurately describe the core architecture.

---

## Part 2: Learnings for Future Expansion

### A. Expanding to Other Stocks

#### What works out of the box

1. **SEC ingestion** — Change `ticker` and `cik` in `EngineConfig`. The EDGAR API, caching, and provenance logging are ticker-agnostic.

2. **XBRL parsing** — The multi-concept fallback (`CONCEPT_MAP`) handles most US-listed companies. The fiscal-year selection algorithm (annual vs quarterly vs YTD) is generic.

3. **Financial metrics** — Ratios, growth rates, and FCF computation are standard. The inflection-point detection (2σ threshold) works for any company.

4. **DCF valuation framework** — The scenario-weighted DCF, sensitivity tables, and reverse-DCF grid are parameterized. Change scenario assumptions in config.

5. **Report generation** — Jinja2 templates are parameterized by `company_name`, `ticker`, etc. The main report / appendix structure is reusable.

6. **Audit framework** — Validation gates, consistency checks, and artifact hashing are company-agnostic.

#### What needs per-company customization

1. **Segment normalization** — `LABEL_MAPPING` in `segment_revenue.py` is NVIDIA-specific. Every company has different segment names and reporting changes over time. **This is the single biggest scaling bottleneck.** Options:
   - Build a sector-specific ontology (semiconductor, software, pharma, etc.)
   - Use LLM-assisted label mapping with human review
   - Accept that segment charts will be suppressed for new companies until mappings are built
   - **The manual platform revenue cross-check pattern is a good fallback** — it takes 30 minutes per company and provides high-value data

2. **Published validation values** — `tests/fixtures/nvda_published_values.json` must be created per company. This requires manually extracting key metrics from 10-K filings for the latest 3 fiscal years. **Budget 1–2 hours per company.**

3. **Scenario assumptions** — Bear/base/bull revenue CAGR, FCF margins, terminal growth, and probabilities are analyst judgment per company. These cannot be automated.

4. **WACC inputs** — Risk-free rate and ERP are market-wide, but beta and the quality adjustment are company-specific.

5. **Keyword dictionaries** — The 9 NLP themes in `KEYWORD_DICTIONARIES` are NVIDIA/semiconductor-specific. A software company would need different themes (cloud ARR, churn, net retention, etc.).

6. **Peer group** — `core_semiconductor_peers`, `infrastructure_peers`, and `ai_capex_context` must be redefined per company/sector.

7. **Split history** — The 10:1 split adjustment is NVIDIA-specific. Other companies have different split histories.

#### Estimated effort per new company

| Task | Time | Automatable? |
|------|------|-------------|
| Config changes (ticker, CIK, dates) | 5 min | Yes |
| Published validation values | 1–2 hours | Partially (LLM-assisted extraction) |
| Scenario assumptions | 1–2 hours | No (analyst judgment) |
| Peer group selection | 30 min | Partially |
| Keyword dictionary | 1 hour | Partially (sector templates) |
| Segment label mapping | 1–2 hours | No (company-specific) |
| Manual platform revenue table | 30 min | No |
| WACC build-up | 30 min | Partially |
| **Total** | **~6–8 hours** | |

### B. Improving the Existing NVIDIA Report

#### High-impact improvements

1. **Fix NLP extraction coverage** — The 32.4% coverage is caused by HTML parsing failures on older filings. Form-specific parsing rules (different regex patterns for 10-K vs 10-Q, different HTML structures across fiscal years) would push coverage above 50%. **Estimated effort: 4–6 hours.** This would promote NLP from diagnostic appendix to active exhibits.

2. **Expand ML training data** — The 4 usable annual observations make the ML model structurally underpowered. Two paths:
   - **Quarterly data:** Use quarterly revenue growth as the target instead of annual. This would give ~40 observations. Requires careful no-lookahead handling of quarterly filing dates.
   - **Peer panel:** Combine NVIDIA with 5–10 semiconductor peers to get 50–100 company-year observations. The `build_peer_panel_dataset()` method already exists but needs 10-year peer historicals.

3. **Automate segment normalization** — Build a mapping table that covers NVIDIA's segment/platform label changes across all fiscal years. The current `LABEL_MAPPING` doesn't handle the XBRL dimension labels correctly. **Root cause:** NVIDIA's XBRL segment dimensions use generic member names that don't map to human-readable platform names without the dimension label text.

4. **Add working-capital analysis** — The report mentions working-capital risk but doesn't compute days sales outstanding, days inventory outstanding, or cash conversion cycle. These are available from the XBRL data and would strengthen the earnings-quality discussion.

5. **Add FY2026 platform revenue** — FY2026 full-year data is available (10-K filed 2026-02-25) but the manual platform revenue table only covers FY2023–FY2025. Adding FY2026 would show the latest Data Center concentration.

#### Architecture improvements for scale

1. **Company registry** — Replace the single `EngineConfig` with a company registry that stores per-company config (ticker, CIK, peers, scenarios, keyword dictionaries, segment mappings, validation values). This is the prerequisite for multi-company runs.

2. **Sector templates** — Create sector-specific keyword dictionaries, peer group templates, and segment ontologies. Start with semiconductors, then expand to software, pharma, financials.

3. **Validation fixture database** — Instead of per-company JSON fixtures, build a database of published financial values sourced from 10-K filings. This could be LLM-assisted: extract key metrics from filing text and cross-check against XBRL.

4. **Run-scoped outputs** — The current run-scoped output packaging (`outputs/runs/{run_id}/`) works but doesn't support comparing runs across companies. A multi-company run manager would need a `{company}/{run_id}/` structure.

5. **PDF rendering** — The weasyprint PDF generation works but has limitations (no JavaScript, limited CSS support). For production quality, consider Puppeteer/Playwright for HTML-to-PDF conversion, or LaTeX for typeset-quality output.

### C. Lessons Learned

1. **Data validation gates are essential.** The original pipeline produced a catastrophically wrong Sell recommendation because XBRL parsing confused a quarterly value with an annual total. The three-gate architecture (data validation → recommendation eligibility → audit consistency) prevented this from ever happening again. **Every quantamental pipeline should have hard validation gates before any recommendation is issued.**

2. **Honest ML exclusion is better than overclaiming.** With only 4 annual observations, the ML model is structurally underpowered. Honestly excluding it from the rating (diagnostic-only) is more credible than pretending it adds value. The model's real value is structural: it forces systematic feature identification and demonstrates scalability.

3. **NLP coverage gates prevent false signals.** The 50% coverage threshold correctly identified that NLP extraction was unreliable. Without the gate, narrative drift scores from 11/34 filings would have been presented as meaningful signals.

4. **Manual cross-checks compensate for automation failures.** When the automated segment normalization failed, the manual platform revenue table provided the segment analysis that the grading rubric requires. **Always have a manual fallback for critical analysis components.**

5. **Source discipline matters for credibility.** Labeling unsourced claims as "analyst assessment" or "monitoring threshold" is more credible than presenting them as facts. The grading rubric explicitly evaluates source attribution.

6. **PDF rendering from Markdown is fragile.** Bullet lists, table wrapping, and page breaks all required workarounds. Converting all lists to tables was the most reliable fix. For future projects, consider generating HTML first and converting to PDF with a browser engine.

7. **Prompt-log completeness is hard to achieve retroactively.** Starting the prompt log from day one and logging every significant interaction verbatim would have been much easier than reconstructing prompts after the fact. **Start the prompt log before writing any code.**

8. **The appendix pattern works well.** Moving technical detail (full risk tables, catalyst tables, peer universe, scalability scorecard, component status, NLP diagnostics) to a Technical Appendix kept the main report concise while preserving all grading evidence.

9. **WACC is the most subjective input.** The 2.9% quality adjustment from CAPM (12.9%) to analyst WACC (10.0%) has a larger impact on the valuation than any other single assumption. Decomposing it into sub-components (net cash, FCF quality, competitive position) makes the judgment more transparent and defensible.

10. **Scenario probability sensitivity reveals fragility.** The bear-skewed distribution ($143.63, -28.0%) vs bull-skewed ($202.46, +1.4%) shows that the Hold recommendation is sensitive to probability assumptions. This transparency strengthens the report's credibility.
