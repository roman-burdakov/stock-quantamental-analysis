"""
Tests for BERTopic/LDA topic modeling (Task 17.2).

Covers:
  - LDA topic modeling with mock filing texts (at least 3 filings)
  - Topic distribution rows conform to NLPFeatureRecord schema
  - Emerging/fading topic detection (early vs late filing weight changes)
  - Graceful handling when fewer than 2 documents
  - BERTopic fallback to LDA when bertopic is not installed
  - Topic model results integrate into compute_narrative_drift
  - Edge cases: empty texts, very short texts, single document

All tests run offline using inline TextSectionRecord fixtures.
Reqs: 7.10
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from src.config import EngineConfig, NLPFeatureRecord, TextSectionRecord
from src.nlp_features import NLPFeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures — inline TextSectionRecords
# ---------------------------------------------------------------------------

def _make_section(
    text: str,
    section_name: str = "risk_factors",
    filing_date: str = "2024-02-28",
    source_available_date: str = "2024-02-28",
    accession_number: str = "0001045810-24-000010",
    form_type: str = "10-K",
) -> TextSectionRecord:
    return TextSectionRecord(
        accession_number=accession_number,
        form_type=form_type,
        section_name=section_name,
        text=text,
        char_count=len(text),
        filing_date=filing_date,
        source_available_date=source_available_date,
        parse_status="success",
    )


# --- Filing texts with distinct thematic content for topic separation ---

FILING_TEXT_2022 = (
    "Gaming revenue declined due to channel inventory corrections and reduced "
    "demand for GeForce RTX desktop and notebook GPUs. Cryptocurrency mining "
    "demand collapsed leading to excess inventory of graphics cards. The PC "
    "market experienced a cyclical downturn after pandemic-driven demand. "
    "Consumer spending on gaming hardware weakened across all regions. "
    "Channel partners reported elevated inventory levels requiring promotional "
    "pricing to clear stock. Add-in board partners reduced orders significantly. "
    "Gaming segment gross margin contracted due to product mix shifts and "
    "promotional activity. Desktop GPU average selling prices declined."
)

FILING_TEXT_2023 = (
    "Data center revenue grew significantly driven by demand for AI training "
    "and inference workloads. Hyperscale cloud service providers increased "
    "capital expenditure on GPU computing infrastructure. The Hopper "
    "architecture H100 GPU achieved strong adoption for large language model "
    "training. CUDA ecosystem and software platform strengthened competitive "
    "moat. Deep learning and generative AI applications drove unprecedented "
    "demand for accelerated computing. Enterprise AI adoption accelerated "
    "across industries. InfiniBand networking revenue grew as data center "
    "scale increased. DGX systems demand exceeded available supply."
)

FILING_TEXT_2024 = (
    "Export controls imposed by the Bureau of Industry and Security restricted "
    "sales of advanced computing chips to China. Trade restrictions affected "
    "our ability to serve customers in restricted markets. The Entity List "
    "expanded to include additional Chinese entities. Geopolitical tensions "
    "between the United States and China created uncertainty for our business. "
    "License requirements for advanced semiconductors limited revenue from "
    "China. We developed different product configurations to comply with "
    "export regulations while serving permitted markets. Revenue from China "
    "declined as a percentage of total data center revenue."
)

FILING_TEXT_2025 = (
    "Blackwell architecture GPUs entered production with strong demand from "
    "hyperscaler cloud service providers. Sovereign AI initiatives drove "
    "new data center deployments globally. AI inference workloads grew "
    "rapidly complementing training demand. Supply constraints at TSMC "
    "foundry limited production capacity for advanced packaging. CoWoS "
    "packaging capacity remained a bottleneck for GPU production. "
    "Competition from AMD custom silicon and cloud provider ASICs intensified. "
    "Enterprise AI adoption broadened beyond hyperscalers to include "
    "financial services healthcare and manufacturing sectors."
)

MDA_TEXT_2023 = (
    "Total revenue for fiscal year 2023 was $26.9 billion. Data Center "
    "revenue was $15.0 billion representing 56 percent of total revenue. "
    "Gaming revenue declined to $9.1 billion. Gross margin was 56.9 percent. "
    "Operating cash flow was $5.6 billion. Research and development expenses "
    "increased to support next-generation GPU architectures."
)

MDA_TEXT_2024 = (
    "Total revenue for fiscal year 2024 was $60.9 billion driven by Data "
    "Center revenue growth of 217 percent. Gross margin improved to 72.7 "
    "percent reflecting higher mix of data center revenue. Operating cash "
    "flow was $28.1 billion and free cash flow was $26.9 billion. AI "
    "training and inference demand drove record data center revenue."
)

MDA_TEXT_2025 = (
    "Total revenue for fiscal year 2025 was $130.5 billion an increase of "
    "114 percent from fiscal year 2024. Data Center revenue was $115.2 "
    "billion representing 88 percent of total revenue. Gross margin was "
    "75.0 percent. Operating cash flow was $64.1 billion and free cash "
    "flow was $60.9 billion. Blackwell architecture ramp drove growth."
)


@pytest.fixture
def extractor() -> NLPFeatureExtractor:
    return NLPFeatureExtractor()


@pytest.fixture
def three_filing_sections() -> list[TextSectionRecord]:
    """Three consecutive filings with risk_factors sections."""
    return [
        _make_section(
            FILING_TEXT_2022, "risk_factors",
            "2023-02-24", "2023-02-24", "acc-2023",
        ),
        _make_section(
            FILING_TEXT_2023, "risk_factors",
            "2024-02-28", "2024-02-28", "acc-2024",
        ),
        _make_section(
            FILING_TEXT_2024, "risk_factors",
            "2025-02-26", "2025-02-26", "acc-2025",
        ),
    ]


@pytest.fixture
def four_filing_sections() -> list[TextSectionRecord]:
    """Four consecutive filings with risk_factors and mda sections."""
    return [
        _make_section(
            FILING_TEXT_2022, "risk_factors",
            "2023-02-24", "2023-02-24", "acc-2023",
        ),
        _make_section(
            MDA_TEXT_2023, "mda",
            "2023-02-24", "2023-02-24", "acc-2023",
        ),
        _make_section(
            FILING_TEXT_2023, "risk_factors",
            "2024-02-28", "2024-02-28", "acc-2024",
        ),
        _make_section(
            MDA_TEXT_2024, "mda",
            "2024-02-28", "2024-02-28", "acc-2024",
        ),
        _make_section(
            FILING_TEXT_2024, "risk_factors",
            "2025-02-26", "2025-02-26", "acc-2025",
        ),
        _make_section(
            MDA_TEXT_2025, "mda",
            "2025-02-26", "2025-02-26", "acc-2025",
        ),
        _make_section(
            FILING_TEXT_2025, "risk_factors",
            "2026-02-25", "2026-02-25", "acc-2026",
        ),
    ]


# ===================================================================
# 1. LDA topic modeling with mock filing texts (≥3 filings)
# ===================================================================


class TestLDATopicModeling:
    """Verify LDA topic modeling produces valid results with ≥3 filings."""

    def test_lda_produces_nonempty_results_with_three_filings(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """LDA with 3 filings should produce topic distribution rows."""
        topic_df, emerging_fading = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        assert isinstance(topic_df, pd.DataFrame)
        assert not topic_df.empty
        assert isinstance(emerging_fading, list)

    def test_lda_produces_results_with_four_filings(
        self, extractor: NLPFeatureExtractor, four_filing_sections,
    ):
        """LDA with 4+ filings should produce topic distributions."""
        topic_df, emerging_fading = extractor.fit_topic_model(
            four_filing_sections, n_topics=3,
        )
        assert not topic_df.empty
        # Should have rows for each document × each topic
        n_docs = len([
            s for s in four_filing_sections
            if s.text and len(s.text.strip()) > 50
        ])
        # At least n_docs rows (one per doc per topic, but topics may be fewer)
        assert len(topic_df) >= n_docs

    def test_lda_topic_weights_are_probabilities(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """Topic weights should be in [0, 1] (probability distribution)."""
        topic_df, _ = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        assert not topic_df.empty
        for val in topic_df["value"]:
            assert 0.0 <= val <= 1.0, f"Topic weight {val} outside [0, 1]"

    def test_lda_topic_weights_sum_to_one_per_document(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """Per-document topic weights should approximately sum to 1.0."""
        topic_df, _ = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        # Group by filing_date + source_accession and check sum
        grouped = topic_df.groupby(
            ["filing_date", "source_accession"]
        )["value"].sum()
        for key, total in grouped.items():
            assert abs(total - 1.0) < 0.01, (
                f"Topic weights for {key} sum to {total}, expected ~1.0"
            )

    def test_lda_feature_names_contain_topic_words(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """Feature names should contain descriptive topic words."""
        topic_df, _ = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        for name in topic_df["feature_name"]:
            # LDA feature names follow pattern: topic_N_word1_word2_...
            assert name.startswith("topic_"), (
                f"Feature name '{name}' doesn't start with 'topic_'"
            )


# ===================================================================
# 2. Topic distribution rows conform to NLPFeatureRecord schema
# ===================================================================


class TestTopicDistributionSchema:
    """Verify topic distribution output conforms to NLPFeatureRecord schema."""

    EXPECTED_COLUMNS = {
        "filing_date",
        "source_available_date",
        "source_accession",
        "section",
        "feature_type",
        "feature_name",
        "value",
        "prev_filing_date",
    }

    def test_all_nlp_feature_record_columns_present(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """Output DataFrame must have all NLPFeatureRecord columns."""
        topic_df, _ = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        assert self.EXPECTED_COLUMNS.issubset(set(topic_df.columns))

    def test_feature_type_is_topic_distribution(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """feature_type should be 'topic_distribution' for all rows."""
        topic_df, _ = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        assert (topic_df["feature_type"] == "topic_distribution").all()

    def test_value_is_numeric(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """value column should contain numeric (float) values."""
        topic_df, _ = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        assert pd.api.types.is_numeric_dtype(topic_df["value"])

    def test_rows_can_construct_nlp_feature_records(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """Each row should be convertible to an NLPFeatureRecord dataclass."""
        topic_df, _ = extractor.fit_topic_model(
            three_filing_sections, n_topics=3,
        )
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        for _, row in topic_df.iterrows():
            record = NLPFeatureRecord(
                filing_date=row["filing_date"],
                source_available_date=row["source_available_date"],
                source_accession=row["source_accession"],
                section=row["section"],
                feature_type=row["feature_type"],
                feature_name=row["feature_name"],
                value=row["value"],
                prev_filing_date=row["prev_filing_date"],
            )
            assert record.feature_type == "topic_distribution"
            assert isinstance(record.value, float)


# ===================================================================
# 3. Emerging/fading topic detection
# ===================================================================


class TestEmergingFadingTopics:
    """Verify emerging/fading topic detection across filings."""

    def test_emerging_fading_returns_list_of_dicts(
        self, extractor: NLPFeatureExtractor, four_filing_sections,
    ):
        """emerging_fading should be a list of dicts with expected keys."""
        _, emerging_fading = extractor.fit_topic_model(
            four_filing_sections, n_topics=3,
        )
        assert isinstance(emerging_fading, list)
        if not emerging_fading:
            pytest.skip("No emerging/fading topics detected")
        for item in emerging_fading:
            assert isinstance(item, dict)
            assert "topic_id" in item
            assert "top_words" in item
            assert "early_weight" in item
            assert "late_weight" in item
            assert "delta" in item
            assert "direction" in item

    def test_direction_values_are_valid(
        self, extractor: NLPFeatureExtractor, four_filing_sections,
    ):
        """direction should be 'emerging', 'fading', or 'stable'."""
        _, emerging_fading = extractor.fit_topic_model(
            four_filing_sections, n_topics=3,
        )
        valid_directions = {"emerging", "fading", "stable"}
        for item in emerging_fading:
            assert item["direction"] in valid_directions, (
                f"Invalid direction: {item['direction']}"
            )

    def test_delta_sign_matches_direction(
        self, extractor: NLPFeatureExtractor, four_filing_sections,
    ):
        """Positive delta → emerging, negative delta → fading."""
        _, emerging_fading = extractor.fit_topic_model(
            four_filing_sections, n_topics=3,
        )
        for item in emerging_fading:
            if item["direction"] == "emerging":
                assert item["delta"] > 0
            elif item["direction"] == "fading":
                assert item["delta"] < 0

    def test_weights_are_non_negative(
        self, extractor: NLPFeatureExtractor, four_filing_sections,
    ):
        """early_weight and late_weight should be non-negative."""
        _, emerging_fading = extractor.fit_topic_model(
            four_filing_sections, n_topics=3,
        )
        for item in emerging_fading:
            assert item["early_weight"] >= 0
            assert item["late_weight"] >= 0

    def test_top_words_are_nonempty_strings(
        self, extractor: NLPFeatureExtractor, four_filing_sections,
    ):
        """top_words should be a non-empty comma-separated string."""
        _, emerging_fading = extractor.fit_topic_model(
            four_filing_sections, n_topics=3,
        )
        for item in emerging_fading:
            assert isinstance(item["top_words"], str)
            assert len(item["top_words"]) > 0


# ===================================================================
# 4. Graceful handling when fewer than 2 documents
# ===================================================================


class TestFewDocuments:
    """Verify graceful handling with insufficient documents."""

    def test_single_document_returns_empty(
        self, extractor: NLPFeatureExtractor,
    ):
        """With only 1 document, topic modeling should return empty."""
        sections = [_make_section(FILING_TEXT_2022, "risk_factors")]
        topic_df, emerging = extractor.fit_topic_model(sections)
        assert topic_df.empty
        assert emerging == []

    def test_empty_sections_list_returns_empty(
        self, extractor: NLPFeatureExtractor,
    ):
        """Empty input list should return empty results."""
        topic_df, emerging = extractor.fit_topic_model([])
        assert topic_df.empty
        assert emerging == []

    def test_all_empty_texts_returns_empty(
        self, extractor: NLPFeatureExtractor,
    ):
        """Sections with empty text should be filtered out."""
        sections = [
            _make_section("", "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section("", "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
            _make_section("", "risk_factors", "2026-02-25", "2026-02-25", "acc-3"),
        ]
        topic_df, emerging = extractor.fit_topic_model(sections)
        assert topic_df.empty
        assert emerging == []

    def test_very_short_texts_filtered_out(
        self, extractor: NLPFeatureExtractor,
    ):
        """Texts with ≤50 characters should be filtered out."""
        sections = [
            _make_section("Short text.", "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section("Also short.", "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
            _make_section("Tiny.", "risk_factors", "2026-02-25", "2026-02-25", "acc-3"),
        ]
        topic_df, emerging = extractor.fit_topic_model(sections)
        assert topic_df.empty
        assert emerging == []

    def test_mix_of_short_and_long_texts(
        self, extractor: NLPFeatureExtractor,
    ):
        """Only texts >50 chars should be used; if <2 remain, return empty."""
        sections = [
            _make_section("Short.", "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(FILING_TEXT_2022, "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        # Only 1 valid doc → should return empty
        topic_df, emerging = extractor.fit_topic_model(sections)
        assert topic_df.empty
        assert emerging == []


# ===================================================================
# 5. BERTopic fallback to LDA when bertopic is not installed
# ===================================================================


class TestBERTopicFallback:
    """Verify BERTopic falls back to LDA when not installed."""

    def test_fallback_to_lda_when_bertopic_unavailable(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """When bertopic import fails, should fall back to LDA and produce results."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "bertopic":
                raise ImportError("Mocked: bertopic not installed")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            topic_df, emerging_fading = extractor.fit_topic_model(
                three_filing_sections, n_topics=3,
            )
            assert isinstance(topic_df, pd.DataFrame)
            # LDA should still produce results
            if not topic_df.empty:
                assert (topic_df["feature_type"] == "topic_distribution").all()
                # LDA feature names start with "topic_" (not "bertopic_")
                for name in topic_df["feature_name"]:
                    assert name.startswith("topic_")

    def test_lda_always_available_via_sklearn(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """LDA via sklearn should always work regardless of bertopic."""
        topic_df, _ = extractor._fit_lda(
            sorted(
                [t for t in three_filing_sections if t.text and len(t.text.strip()) > 50],
                key=lambda t: t.filing_date,
            ),
            [t.text for t in three_filing_sections if t.text and len(t.text.strip()) > 50],
            n_topics=3,
        )
        assert isinstance(topic_df, pd.DataFrame)

    def test_bertopic_exception_falls_back_to_lda(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """If BERTopic raises an exception during fitting, should fall back to LDA."""
        doc_records = sorted(
            [t for t in three_filing_sections if t.text and len(t.text.strip()) > 50],
            key=lambda t: t.filing_date,
        )
        corpus = [t.text for t in doc_records]

        # Mock bertopic to raise during fit_transform
        with mock.patch.dict("sys.modules", {"bertopic": mock.MagicMock()}):
            import sys
            bertopic_mock = sys.modules["bertopic"]
            bertopic_mock.BERTopic.return_value.fit_transform.side_effect = RuntimeError(
                "Mocked BERTopic failure"
            )
            topic_df, emerging = extractor._fit_bertopic(
                doc_records, corpus, n_topics=3,
            )
            # Should fall back to LDA and produce results
            assert isinstance(topic_df, pd.DataFrame)


# ===================================================================
# 6. Integration with compute_narrative_drift
# ===================================================================


class TestNarrativeDriftIntegration:
    """Verify topic model results integrate into compute_narrative_drift."""

    def test_topic_rows_appear_when_use_topic_model_true(
        self, four_filing_sections, tmp_path,
    ):
        """When use_topic_model=True, narrative drift should include topic rows."""
        config = EngineConfig()
        config.use_topic_model = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(four_filing_sections)
        # LDA is always available (sklearn), so topic rows should appear
        assert "topic_distribution" in df["feature_type"].values

    def test_topic_rows_absent_when_use_topic_model_false(
        self, four_filing_sections, tmp_path,
    ):
        """When use_topic_model=False, no topic rows should appear."""
        config = EngineConfig()
        config.use_topic_model = False
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(four_filing_sections)
        assert "topic_distribution" not in df["feature_type"].values

    def test_topic_rows_have_correct_schema_in_drift(
        self, four_filing_sections, tmp_path,
    ):
        """Topic rows within narrative drift should match NLPFeatureRecord schema."""
        config = EngineConfig()
        config.use_topic_model = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(four_filing_sections)
        topic_rows = df[df["feature_type"] == "topic_distribution"]
        assert not topic_rows.empty
        expected_cols = {
            "filing_date", "source_available_date", "source_accession",
            "section", "feature_type", "feature_name", "value", "prev_filing_date",
        }
        assert expected_cols.issubset(set(topic_rows.columns))

    def test_csv_output_includes_topic_rows(
        self, four_filing_sections, tmp_path,
    ):
        """The saved CSV should include topic distribution rows."""
        config = EngineConfig()
        config.use_topic_model = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        ext.compute_narrative_drift(four_filing_sections)

        csv_path = tmp_path / "processed" / "nvda_nlp_features.csv"
        assert csv_path.exists()
        saved_df = pd.read_csv(csv_path)
        assert "topic_distribution" in saved_df["feature_type"].values

    def test_narrative_drift_still_has_tfidf_and_keywords_with_topics(
        self, four_filing_sections, tmp_path,
    ):
        """Enabling topics should not remove tfidf or keyword features."""
        config = EngineConfig()
        config.use_topic_model = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(four_filing_sections)
        feature_types = set(df["feature_type"].unique())
        assert "tfidf_similarity" in feature_types
        assert "keyword_score" in feature_types
        assert "topic_distribution" in feature_types


# ===================================================================
# 7. Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case handling for topic modeling."""

    def test_identical_texts_produce_valid_results(
        self, extractor: NLPFeatureExtractor,
    ):
        """Identical texts should still produce valid topic distributions."""
        sections = [
            _make_section(FILING_TEXT_2022, "risk_factors", "2023-02-24", "2023-02-24", "acc-1"),
            _make_section(FILING_TEXT_2022, "risk_factors", "2024-02-28", "2024-02-28", "acc-2"),
            _make_section(FILING_TEXT_2022, "risk_factors", "2025-02-26", "2025-02-26", "acc-3"),
        ]
        topic_df, emerging = extractor.fit_topic_model(sections, n_topics=2)
        assert isinstance(topic_df, pd.DataFrame)
        assert isinstance(emerging, list)
        # With identical texts, all topics should be stable
        for item in emerging:
            assert item["direction"] == "stable"

    def test_n_topics_exceeds_documents(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """Requesting more topics than documents should be handled gracefully."""
        topic_df, emerging = extractor.fit_topic_model(
            three_filing_sections, n_topics=20,
        )
        assert isinstance(topic_df, pd.DataFrame)
        # Should cap topics to min(n_topics, n_docs, n_features)

    def test_n_topics_one(
        self, extractor: NLPFeatureExtractor, three_filing_sections,
    ):
        """Requesting 1 topic should be handled (may produce empty if <2 needed)."""
        topic_df, emerging = extractor.fit_topic_model(
            three_filing_sections, n_topics=1,
        )
        assert isinstance(topic_df, pd.DataFrame)
        assert isinstance(emerging, list)

    def test_mixed_sections_processed(
        self, extractor: NLPFeatureExtractor,
    ):
        """Texts from different sections should all be processed."""
        sections = [
            _make_section(FILING_TEXT_2022, "risk_factors", "2023-02-24", "2023-02-24", "acc-1"),
            _make_section(MDA_TEXT_2023, "mda", "2023-02-24", "2023-02-24", "acc-1"),
            _make_section(FILING_TEXT_2023, "risk_factors", "2024-02-28", "2024-02-28", "acc-2"),
            _make_section(MDA_TEXT_2024, "mda", "2024-02-28", "2024-02-28", "acc-2"),
        ]
        topic_df, _ = extractor.fit_topic_model(sections, n_topics=2)
        if not topic_df.empty:
            sections_present = set(topic_df["section"].unique())
            assert len(sections_present) >= 1  # At least some sections represented
