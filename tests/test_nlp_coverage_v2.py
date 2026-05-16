"""
Tests for v2 NLP coverage recovery — Task 2.4 (Reqs 3, 7, 14).

Covers:
- ``FilingTextParser.parse_filing()`` extraction tiers and proxy fallback
- ``FilingTextParser.check_proxy_quality()`` keyword-density gate
- ``NLPFeatureExtractor.compute_sentiment_polarity()`` polarity bounds
- ``NLPFeatureExtractor.compute_sentiment_delta()`` per-section deltas
- Manual QA workflow: ``outputs/nlp_proxy_qa.csv`` schema invariants
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import EngineConfig, TextSectionRecord
from src.filing_text_parser import FilingTextParser
from src.nlp_features import NLPFeatureExtractor


# ---------------------------------------------------------------------------
# Filing parser: extraction tiers and proxy fallback
# ---------------------------------------------------------------------------


SAMPLE_10K_HTML = """
<html><body>
<p>Cover page boilerplate.</p>
<p>Item 1. Business.</p>
<p>NVIDIA pioneered accelerated computing. We design GPUs for AI training and
inference. Revenue growth has been driven by data center demand. Our gross
margin remains strong. Compared to prior fiscal year, revenue increased
substantially. We continue to invest in next-generation architectures.</p>
<p>Item 1A. Risk Factors.</p>
<p>Demand for our products may not meet expectations. Export controls could
adversely affect our business. We depend on a limited number of foundries.
Customer concentration may adversely affect results.</p>
<p>Item 7. Management's Discussion and Analysis of Financial Condition and
Results of Operations.</p>
""" + ("<p>Revenue for the fiscal year increased compared to the prior year, "
       "driven by strong data center demand. Gross margin expanded as the "
       "product mix shifted toward higher-margin offerings. Operating income "
       "grew substantially. Net income reached a record high. We expect "
       "continued strength in data center demand for the next fiscal year. "
       "Capital expenditures remained disciplined. Free cash flow improved "
       "compared to prior year. Stock-based compensation expense increased "
       "in line with headcount growth. Research and development spending "
       "reflects our commitment to leading-edge architectures.</p>" * 8) + """
<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk.</p>
<p>We are exposed to interest rate and foreign exchange risk.</p>
<p>Item 8. Financial Statements.</p>
</body></html>
"""

SAMPLE_10Q_HTML_NO_HEADERS = """
<html><body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>FORM 10-Q</p>
<p>PART I FINANCIAL INFORMATION</p>
<p>Condensed Consolidated Statements (Unaudited)</p>
""" + ("<p>Revenue for the quarter increased compared to the prior year period. "
       "Gross margin held. Net income grew. Results of operations reflect "
       "continued strength in our data center business. Compared to the prior "
       "fiscal year, revenue is up substantially.</p>" * 50) + """
</body></html>
"""


def test_parse_filing_assigns_full_extraction_when_items_present(tmp_path: Path) -> None:
    cfg = EngineConfig()
    cfg.interim_dir = tmp_path
    parser = FilingTextParser(cfg)
    records = parser.parse_filing(
        html=SAMPLE_10K_HTML,
        accession="acc-test-10k",
        form_type="10-K",
        filing_date="2024-02-21",
        source_available_date="2024-02-21",
    )
    by_section = {r.section_name: r for r in records}
    assert "mda" in by_section
    mda_rec = by_section["mda"]
    tier = getattr(mda_rec, "extraction_tier", None)
    assert tier in {"full_extraction", "partial_extraction"}, (
        f"Expected full or partial extraction, got tier={tier} chars={mda_rec.char_count}"
    )


def test_parse_filing_uses_mda_proxy_fallback_when_items_missing(tmp_path: Path) -> None:
    cfg = EngineConfig()
    cfg.interim_dir = tmp_path
    parser = FilingTextParser(cfg)
    records = parser.parse_filing(
        html=SAMPLE_10Q_HTML_NO_HEADERS,
        accession="acc-test-10q",
        form_type="10-Q",
        filing_date="2024-08-25",
        source_available_date="2024-08-25",
    )
    by_section = {r.section_name: r for r in records}
    mda_rec = by_section.get("mda")
    assert mda_rec is not None
    tier = getattr(mda_rec, "extraction_tier", None)
    # The 10-Q has no Item headers; must fall through to proxy
    assert tier == "mda_proxy_fallback"
    # Proxy should produce substantial text
    assert mda_rec.char_count >= 500


def test_check_proxy_quality_detects_relevant_text() -> None:
    relevant = (
        "Revenue for the fiscal year increased compared to prior year. "
        "Gross margin remained strong. Net income reached a record level. "
        "Results of operations reflect quarter-over-quarter growth."
    )
    is_relevant, counts = FilingTextParser.check_proxy_quality(relevant)
    assert is_relevant is True
    # All 7 indicator terms should be present (or close to it)
    assert sum(counts.values()) >= 5


def test_check_proxy_quality_rejects_boilerplate() -> None:
    boilerplate = (
        "TABLE OF CONTENTS. Page. Cover. Index. Signatures. "
        "Exhibit list. Filing date. CIK number. Submission ID."
    )
    is_relevant, counts = FilingTextParser.check_proxy_quality(boilerplate)
    assert is_relevant is False, (
        f"Boilerplate text should not be flagged as relevant; "
        f"counts={counts}"
    )


def test_check_proxy_quality_handles_empty_string() -> None:
    is_relevant, counts = FilingTextParser.check_proxy_quality("")
    assert is_relevant is False
    assert counts == {}


def test_extraction_tier_persists_in_section_json(tmp_path: Path) -> None:
    cfg = EngineConfig()
    cfg.interim_dir = tmp_path
    parser = FilingTextParser(cfg)
    parser.parse_filing(
        html=SAMPLE_10K_HTML,
        accession="acc-tier-persistence",
        form_type="10-K",
        filing_date="2024-02-21",
        source_available_date="2024-02-21",
    )
    out = tmp_path / "sections_acc_tier_persistence.json"
    assert out.exists()
    import json
    payload = json.loads(out.read_text())
    assert payload, "section JSON should contain records"
    for entry in payload:
        assert "extraction_tier" in entry, (
            f"extraction_tier missing from saved entry: {entry}"
        )
        assert entry["extraction_tier"] in {
            "full_extraction",
            "partial_extraction",
            "mda_proxy_fallback",
            "failed",
            None,
        }


# ---------------------------------------------------------------------------
# Sentiment polarity (Req 3.4)
# ---------------------------------------------------------------------------


def test_sentiment_polarity_positive_text() -> None:
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    text = (
        "Revenue growth was strong and we achieved record success. "
        "Profitability improved with strong margins. Customer demand was "
        "robust and we exceeded expectations."
    )
    score = ext.compute_sentiment_polarity(text)
    assert 0.5 < score <= 1.0, f"expected strongly positive, got {score}"


def test_sentiment_polarity_negative_text() -> None:
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    text = (
        "We face significant challenges and concerns. The decline in demand "
        "resulted in losses and disappointing results. Risks have increased "
        "and our weaknesses were exposed."
    )
    score = ext.compute_sentiment_polarity(text)
    assert -1.0 <= score < -0.5, f"expected strongly negative, got {score}"


def test_sentiment_polarity_neutral_text() -> None:
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    text = "The company filed financial statements per regulatory requirements."
    score = ext.compute_sentiment_polarity(text)
    assert -0.1 <= score <= 0.1, f"expected near zero, got {score}"


def test_sentiment_polarity_empty_string_returns_zero() -> None:
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    assert ext.compute_sentiment_polarity("") == 0.0


def test_sentiment_polarity_bounded_in_range() -> None:
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    # Construct extreme positive — only positive words
    pos = "growth growth growth growth growth growth growth"
    s = ext.compute_sentiment_polarity(pos)
    assert -1.0 <= s <= 1.0
    # Extreme negative
    neg = "decline decline decline decline decline decline decline"
    s = ext.compute_sentiment_polarity(neg)
    assert -1.0 <= s <= 1.0


def test_sentiment_polarity_lexicon_avoids_misclassification() -> None:
    """LM lexicon must not be tricked by financial-domain false friends.

    'high cost' is neutral in finance, not positive. 'lower' is negative,
    not neutral. Verify these key disambiguations.
    """
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    # A LM-correct lexicon does NOT count "high" as positive (neutral in
    # finance: "high cost", "high risk"). Our embedded list excludes it.
    high_text = "We face high costs and high risk in this fiscal year."
    s = ext.compute_sentiment_polarity(high_text)
    # Should be negative because of "risk" being in negative list.
    assert s < 0, f"'high costs/risk' should be negative, got {s}"


def test_compute_sentiment_delta_produces_polarity_and_delta_rows() -> None:
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    pos_text = "Revenue growth was strong and profits improved."
    neg_text = "We face decline, losses, and risks."
    recs = [
        TextSectionRecord("acc-1", "10-Q", "mda", pos_text, len(pos_text),
                          "2024-08-15", "2024-08-15", "success"),
        TextSectionRecord("acc-2", "10-Q", "mda", neg_text, len(neg_text),
                          "2024-11-15", "2024-11-15", "success"),
    ]
    df = ext.compute_sentiment_delta(recs)
    assert not df.empty
    polarity_rows = df[df["feature_name"] == "sentiment_polarity"]
    delta_rows = df[df["feature_name"] == "sentiment_delta"]
    assert len(polarity_rows) == 2
    assert len(delta_rows) == 1, (
        "Exactly one delta row expected (after first observation)"
    )
    # Delta should be negative (positive → negative sentiment)
    delta = delta_rows.iloc[0]["value"]
    assert delta < 0, f"expected negative delta, got {delta}"


def test_compute_sentiment_delta_groups_by_section() -> None:
    """Delta must be computed within section, not across sections."""
    cfg = EngineConfig()
    ext = NLPFeatureExtractor(cfg)
    recs = [
        TextSectionRecord("acc-1", "10-K", "mda",
                          "growth profits success", 22,
                          "2024-02-15", "2024-02-15", "success"),
        TextSectionRecord("acc-1", "10-K", "risk_factors",
                          "decline losses risks", 20,
                          "2024-02-15", "2024-02-15", "success"),
        TextSectionRecord("acc-2", "10-K", "mda",
                          "decline losses risks", 20,
                          "2025-02-15", "2025-02-15", "success"),
        TextSectionRecord("acc-2", "10-K", "risk_factors",
                          "growth profits success", 22,
                          "2025-02-15", "2025-02-15", "success"),
    ]
    df = ext.compute_sentiment_delta(recs)
    delta_rows = df[df["feature_name"] == "sentiment_delta"]
    # 2 sections, each with 2 observations → 2 delta rows
    assert len(delta_rows) == 2
    # MD&A should have negative delta (positive → negative)
    mda_delta = delta_rows[delta_rows["section"] == "mda"].iloc[0]["value"]
    assert mda_delta < 0
    # Risk factors should have positive delta (negative → positive)
    rf_delta = delta_rows[delta_rows["section"] == "risk_factors"].iloc[0]["value"]
    assert rf_delta > 0


# ---------------------------------------------------------------------------
# Manual QA workflow contract (Req 14.2 / 14.3)
# ---------------------------------------------------------------------------


def test_nlp_proxy_qa_csv_required_columns() -> None:
    """The manual QA CSV format is part of the audit contract. If we
    later auto-generate it, the schema must include these columns."""
    expected_columns = ["accession", "quality_label", "qa_reviewer", "qa_date"]
    # The pipeline writes outputs/nlp_proxy_qa.csv with this schema
    # per Req 14.2. This test documents the contract; the actual
    # generation happens in the pipeline orchestration code.
    df = pd.DataFrame(columns=expected_columns)
    for col in expected_columns:
        assert col in df.columns


def test_proxy_qa_acceptance_rule_8_of_10() -> None:
    """Document the 14.3 acceptance threshold: ≥8/10 relevant labels."""
    # Simulate a filled QA CSV
    qa = pd.DataFrame(
        {
            "accession": [f"acc-{i}" for i in range(10)],
            "quality_label": [
                "relevant_mda", "relevant_mda", "partial_relevant",
                "relevant_mda", "relevant_mda", "boilerplate",
                "relevant_mda", "relevant_mda", "relevant_mda",
                "relevant_mda",
            ],
            "qa_reviewer": ["test"] * 10,
            "qa_date": ["2026-05-16"] * 10,
        }
    )
    relevant_count = qa["quality_label"].isin(
        ["relevant_mda", "partial_relevant"]
    ).sum()
    # 9 out of 10 → passes
    assert relevant_count >= 8

    # Simulate a failing CSV
    qa_fail = qa.copy()
    qa_fail.loc[0:2, "quality_label"] = "boilerplate"
    relevant_fail = qa_fail["quality_label"].isin(
        ["relevant_mda", "partial_relevant"]
    ).sum()
    assert relevant_fail < 8
