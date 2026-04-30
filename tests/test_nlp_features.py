"""
Tests for src.nlp_features — NLP Feature Extraction.

Covers:
  - Keyword scoring returns all 9 themes from KEYWORD_DICTIONARIES
  - TF-IDF cosine similarity produces valid [0, 1] scores
  - compute_narrative_drift output DataFrame has source_available_date column

All tests run offline using inline TextSectionRecord fixtures.
Reqs: 7.1, 7.2
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import KEYWORD_DICTIONARIES, TextSectionRecord
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


RISK_TEXT_2024 = (
    "The company faces significant risks from export controls imposed by the "
    "Bureau of Industry and Security on advanced computing chips to China. "
    "Supply constraints at TSMC foundry limit our capacity for Hopper and "
    "Blackwell architecture GPUs. Competition from AMD, Intel, and custom "
    "silicon such as Google TPU and Amazon Trainium continues to intensify. "
    "Customer concentration remains high as a limited number of hyperscaler "
    "cloud service providers account for a significant portion of Data Center "
    "revenue. Gaming cyclicality and channel inventory fluctuations affect "
    "GeForce RTX demand. Gross margin and pricing pressure from product mix "
    "shifts could impact profitability. Inventory demand cyclicality and "
    "backlog visibility remain ongoing concerns. Artificial intelligence and "
    "accelerated computing drive our growth but also increase execution risk."
)

RISK_TEXT_2025 = (
    "Export controls and trade restrictions continue to adversely affect our "
    "business in China and other restricted markets. The Bureau of Industry "
    "and Security has imposed additional restrictions on advanced computing "
    "semiconductors. Supply chain disruptions and capacity constraints at our "
    "foundry partners including TSMC could limit production of Blackwell "
    "architecture products. Competition from AMD custom silicon TPU and "
    "Trainium alternatives is increasing. Customer concentration risk persists "
    "with major cloud service providers representing significant revenue. "
    "Gaming market cyclicality affects GeForce demand patterns. Margin "
    "pressure from pricing and product mix changes is a concern. Inventory "
    "and demand cyclicality create backlog uncertainty. AI and deep learning "
    "inference workloads are central to our data center growth strategy."
)

MDA_TEXT_2024 = (
    "Total revenue for fiscal year 2024 was $60.9 billion driven by Data "
    "Center revenue growth. Gross margin improved to 72.7 percent reflecting "
    "higher mix of data center revenue. Research and development expenses "
    "increased to support our Hopper and Blackwell product roadmap. Operating "
    "cash flow was $28.1 billion and free cash flow was $26.9 billion."
)

MDA_TEXT_2025 = (
    "Total revenue for fiscal year 2025 was $130.5 billion an increase of "
    "114 percent from fiscal year 2024. Data Center revenue was $115.2 billion "
    "representing 88 percent of total revenue. Gross margin was 75.0 percent. "
    "Operating cash flow was $64.1 billion and free cash flow was $60.9 billion. "
    "AI training and inference demand drove unprecedented growth."
)


@pytest.fixture
def extractor() -> NLPFeatureExtractor:
    return NLPFeatureExtractor()


@pytest.fixture
def sample_sections() -> list[TextSectionRecord]:
    """Two consecutive filings with risk_factors and mda sections."""
    return [
        _make_section(RISK_TEXT_2024, "risk_factors", "2024-02-28", "2024-02-28", "0001045810-24-000010"),
        _make_section(MDA_TEXT_2024, "mda", "2024-02-28", "2024-02-28", "0001045810-24-000010"),
        _make_section(RISK_TEXT_2025, "risk_factors", "2025-02-26", "2025-02-26", "0001045810-25-000013"),
        _make_section(MDA_TEXT_2025, "mda", "2025-02-26", "2025-02-26", "0001045810-25-000013"),
    ]


# ===================================================================
# 1. Keyword scoring returns all 9 themes
# ===================================================================


class TestKeywordScoring:
    """Verify compute_keyword_scores returns a dict with all 9 theme keys."""

    def test_returns_all_nine_themes(self, extractor: NLPFeatureExtractor):
        scores = extractor.compute_keyword_scores(RISK_TEXT_2024)
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(KEYWORD_DICTIONARIES.keys())
        assert len(scores) == 9

    def test_scores_are_numeric(self, extractor: NLPFeatureExtractor):
        scores = extractor.compute_keyword_scores(RISK_TEXT_2024)
        for theme, score in scores.items():
            assert isinstance(score, (int, float)), f"{theme} score is not numeric"

    def test_scores_are_non_negative(self, extractor: NLPFeatureExtractor):
        scores = extractor.compute_keyword_scores(RISK_TEXT_2024)
        for theme, score in scores.items():
            assert score >= 0, f"{theme} has negative score: {score}"

    def test_relevant_themes_have_positive_scores(self, extractor: NLPFeatureExtractor):
        """Text mentioning export controls, supply, competition should score > 0."""
        scores = extractor.compute_keyword_scores(RISK_TEXT_2024)
        assert scores["export_controls_china"] > 0
        assert scores["supply_constraints"] > 0
        assert scores["competition"] > 0

    def test_empty_text_returns_all_themes_zero(self, extractor: NLPFeatureExtractor):
        scores = extractor.compute_keyword_scores("")
        assert len(scores) == 9
        for score in scores.values():
            assert score == 0.0


# ===================================================================
# 2. TF-IDF produces valid [0, 1] scores
# ===================================================================


class TestTfidfSimilarity:
    """Verify compute_tfidf_similarity returns similarity scores in [0, 1]."""

    def test_similarity_scores_between_zero_and_one(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        df = extractor.compute_tfidf_similarity(sample_sections)
        assert not df.empty
        for score in df["similarity_score"]:
            assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"

    def test_returns_dataframe_with_expected_columns(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        df = extractor.compute_tfidf_similarity(sample_sections)
        expected_cols = {
            "filing_date", "source_available_date", "source_accession",
            "section", "similarity_score", "prev_filing_date",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_identical_texts_have_high_similarity(self, extractor: NLPFeatureExtractor):
        """Two identical sections should produce similarity close to 1.0."""
        sections = [
            _make_section(RISK_TEXT_2024, "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(RISK_TEXT_2024, "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        df = extractor.compute_tfidf_similarity(sections)
        assert not df.empty
        assert df["similarity_score"].iloc[0] > 0.99

    def test_single_filing_returns_empty(self, extractor: NLPFeatureExtractor):
        """Need at least 2 filings for pairwise comparison."""
        sections = [_make_section(RISK_TEXT_2024, "risk_factors")]
        df = extractor.compute_tfidf_similarity(sections)
        assert df.empty


# ===================================================================
# 3. source_available_date present in narrative drift output
# ===================================================================


class TestNarrativeDrift:
    """Verify compute_narrative_drift output has source_available_date."""

    def test_source_available_date_column_present(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord],
        tmp_path,
    ):
        # Point processed_dir to tmp to avoid polluting project data
        extractor.config.processed_dir = tmp_path / "processed"
        df = extractor.compute_narrative_drift(sample_sections)
        assert "source_available_date" in df.columns

    def test_source_available_date_populated(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord],
        tmp_path,
    ):
        extractor.config.processed_dir = tmp_path / "processed"
        df = extractor.compute_narrative_drift(sample_sections)
        assert not df.empty
        non_null = df["source_available_date"].dropna()
        assert len(non_null) > 0, "source_available_date has no non-null values"

    def test_output_contains_both_feature_types(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord],
        tmp_path,
    ):
        extractor.config.processed_dir = tmp_path / "processed"
        df = extractor.compute_narrative_drift(sample_sections)
        feature_types = set(df["feature_type"].unique())
        assert "tfidf_similarity" in feature_types
        assert "keyword_score" in feature_types


# ===================================================================
# 4. Sentence-transformer embedding distances (17.1)
# ===================================================================


class TestEmbeddingDistance:
    """Verify compute_embedding_distance returns valid cosine distances."""

    def test_graceful_degradation_without_library(self, extractor, sample_sections):
        """Should return empty DataFrame when sentence-transformers is missing.

        We test the actual method — if the library IS installed, it will
        produce results; if not, it returns empty.  Either outcome is valid.
        """
        df = extractor.compute_embedding_distance(sample_sections)
        assert isinstance(df, pd.DataFrame)

    def test_embedding_distance_values_in_valid_range(self, extractor, sample_sections):
        """If sentence-transformers is available, distances should be in [0, 2]."""
        df = extractor.compute_embedding_distance(sample_sections)
        if df.empty:
            pytest.skip("sentence-transformers not installed")
        for val in df["value"]:
            assert 0.0 <= val <= 2.0, f"Embedding distance {val} outside [0, 2]"

    def test_embedding_distance_has_expected_columns(self, extractor, sample_sections):
        """Output should have NLPFeatureRecord-compatible columns."""
        df = extractor.compute_embedding_distance(sample_sections)
        if df.empty:
            pytest.skip("sentence-transformers not installed")
        expected = {"filing_date", "source_available_date", "source_accession",
                    "section", "feature_type", "feature_name", "value", "prev_filing_date"}
        assert expected.issubset(set(df.columns))

    def test_embedding_distance_feature_type(self, extractor, sample_sections):
        """feature_type should be 'embedding_distance'."""
        df = extractor.compute_embedding_distance(sample_sections)
        if df.empty:
            pytest.skip("sentence-transformers not installed")
        assert (df["feature_type"] == "embedding_distance").all()

    def test_single_filing_returns_empty(self, extractor):
        """Need at least 2 filings for pairwise comparison."""
        sections = [_make_section(RISK_TEXT_2024, "risk_factors")]
        df = extractor.compute_embedding_distance(sections)
        assert df.empty or len(df) == 0

    def test_integration_with_narrative_drift(self, tmp_path, sample_sections):
        """When use_embeddings=True, narrative drift should include embedding rows."""
        from src.config import EngineConfig
        config = EngineConfig()
        config.use_embeddings = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(sample_sections)
        # If library is installed, we should see embedding_distance rows
        try:
            import sentence_transformers  # noqa: F401
            assert "embedding_distance" in df["feature_type"].values
        except ImportError:
            # Without the library, embedding rows won't appear — that's fine
            pass


# ===================================================================
# 5. Topic modeling — LDA / BERTopic (17.2)
# ===================================================================


class TestTopicModeling:
    """Verify fit_topic_model produces topic distributions."""

    def test_lda_returns_dataframe_and_emerging_fading(self, extractor, sample_sections):
        """LDA should produce a non-empty DataFrame and emerging/fading list."""
        topic_df, emerging_fading = extractor.fit_topic_model(sample_sections, n_topics=3)
        assert isinstance(topic_df, pd.DataFrame)
        assert isinstance(emerging_fading, list)

    def test_lda_topic_distribution_columns(self, extractor, sample_sections):
        """Topic distribution rows should have NLPFeatureRecord-compatible columns."""
        topic_df, _ = extractor.fit_topic_model(sample_sections, n_topics=3)
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        expected = {"filing_date", "source_available_date", "source_accession",
                    "section", "feature_type", "feature_name", "value"}
        assert expected.issubset(set(topic_df.columns))

    def test_lda_feature_type_is_topic_distribution(self, extractor, sample_sections):
        """feature_type should be 'topic_distribution'."""
        topic_df, _ = extractor.fit_topic_model(sample_sections, n_topics=3)
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        assert (topic_df["feature_type"] == "topic_distribution").all()

    def test_lda_values_are_probabilities(self, extractor, sample_sections):
        """Topic weights should be in [0, 1]."""
        topic_df, _ = extractor.fit_topic_model(sample_sections, n_topics=3)
        if topic_df.empty:
            pytest.skip("Not enough data for topic modeling")
        for val in topic_df["value"]:
            assert 0.0 <= val <= 1.0, f"Topic weight {val} outside [0, 1]"

    def test_too_few_documents_returns_empty(self, extractor):
        """With only 1 document, topic modeling should return empty."""
        sections = [_make_section(RISK_TEXT_2024, "risk_factors")]
        topic_df, emerging = extractor.fit_topic_model(sections)
        assert topic_df.empty
        assert emerging == []

    def test_integration_with_narrative_drift(self, tmp_path, sample_sections):
        """When use_topic_model=True, narrative drift should include topic rows."""
        from src.config import EngineConfig
        config = EngineConfig()
        config.use_topic_model = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(sample_sections)
        # LDA is always available (sklearn), so topic rows should appear
        assert "topic_distribution" in df["feature_type"].values


# ===================================================================
# 6. Change-point detection with ruptures (17.3)
# ===================================================================


class TestChangePointDetection:
    """Verify detect_change_points returns valid date lists."""

    def test_graceful_degradation_without_library(self, extractor):
        """Should return empty list when ruptures is missing."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                           index=["2020-01-01", "2021-01-01", "2022-01-01",
                                  "2023-01-01", "2024-01-01"])
        result = extractor.detect_change_points(series)
        assert isinstance(result, list)

    def test_returns_list_of_date_strings(self, extractor):
        """Change points should be date strings from the series index."""
        # Create a series with an obvious structural break
        values = [0.9, 0.91, 0.92, 0.93, 0.1, 0.11, 0.12, 0.13]
        dates = [f"202{i}-01-01" for i in range(len(values))]
        series = pd.Series(values, index=dates)
        result = extractor.detect_change_points(series, pen=1.0)
        assert isinstance(result, list)
        for cp in result:
            assert isinstance(cp, str)

    def test_short_series_returns_empty(self, extractor):
        """Series with fewer than 4 points should return empty."""
        series = pd.Series([1.0, 2.0], index=["2023-01-01", "2024-01-01"])
        result = extractor.detect_change_points(series)
        assert result == []

    def test_nan_handling(self, extractor):
        """Series with NaN values should be handled gracefully."""
        import numpy as np
        series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
                           index=["2019-01-01", "2020-01-01", "2021-01-01",
                                  "2022-01-01", "2023-01-01", "2024-01-01"])
        result = extractor.detect_change_points(series)
        assert isinstance(result, list)

    def test_integration_with_narrative_drift(self, tmp_path):
        """When use_change_point_detection=True, narrative drift should include
        change_point rows (if ruptures is installed and enough data)."""
        from src.config import EngineConfig
        config = EngineConfig()
        config.use_change_point_detection = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        # Build enough sections for change-point detection (need ≥4 filing pairs)
        sections = []
        for year in range(2019, 2025):
            sections.append(_make_section(
                RISK_TEXT_2024 + f" Year {year} specific content.",
                "risk_factors",
                f"{year}-02-28",
                f"{year}-02-28",
                f"acc-{year}",
            ))
            sections.append(_make_section(
                MDA_TEXT_2024 + f" Year {year} specific content.",
                "mda",
                f"{year}-02-28",
                f"{year}-02-28",
                f"acc-{year}",
            ))

        df = ext.compute_narrative_drift(sections)
        # change_point rows may or may not appear depending on whether
        # ruptures is installed and whether breaks are detected
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


# ===================================================================
# 7. Peer-panel ML (17.5) — tests in test_nlp_features for convenience
# ===================================================================


class TestPeerPanelML:
    """Verify train_peer_panel_model handles missing/present peer data."""

    def test_no_peer_data_returns_none(self):
        """Without peer data, should return (None, {})."""
        from src.ml_models import MLDriverModel
        model_mgr = MLDriverModel()
        features = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6]})
        target = pd.Series([0.1, 0.2, 0.3])
        model, coeffs = model_mgr.train_peer_panel_model(features, target, peer_data=None)
        assert model is None
        assert coeffs == {}

    def test_empty_peer_data_returns_none(self):
        """Empty peer DataFrame should return (None, {})."""
        from src.ml_models import MLDriverModel
        model_mgr = MLDriverModel()
        features = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6]})
        target = pd.Series([0.1, 0.2, 0.3])
        model, coeffs = model_mgr.train_peer_panel_model(
            features, target, peer_data=pd.DataFrame()
        )
        assert model is None
        assert coeffs == {}

    def test_peer_data_missing_target_column(self):
        """Peer data without 'target_value' should return (None, {})."""
        from src.ml_models import MLDriverModel
        model_mgr = MLDriverModel()
        features = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6]})
        target = pd.Series([0.1, 0.2, 0.3])
        peer_data = pd.DataFrame({"feat_a": [7, 8], "feat_b": [9, 10], "ticker": ["AMD", "INTC"]})
        model, coeffs = model_mgr.train_peer_panel_model(features, target, peer_data=peer_data)
        assert model is None
        assert coeffs == {}

    def test_valid_peer_data_trains_model(self):
        """With valid peer data, should return a fitted model."""
        from src.ml_models import MLDriverModel
        model_mgr = MLDriverModel()
        features = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        target = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        peer_data = pd.DataFrame({
            "ticker": ["AMD"] * 5 + ["INTC"] * 5,
            "feat_a": [1.5, 2.5, 3.5, 4.5, 5.5, 1.2, 2.2, 3.2, 4.2, 5.2],
            "feat_b": [15.0, 25.0, 35.0, 45.0, 55.0, 12.0, 22.0, 32.0, 42.0, 52.0],
            "target_value": [0.12, 0.22, 0.32, 0.42, 0.52, 0.11, 0.21, 0.31, 0.41, 0.51],
        })
        model, coeffs = model_mgr.train_peer_panel_model(features, target, peer_data=peer_data)
        assert model is not None
        assert len(coeffs) == 2  # feat_a, feat_b
        assert "feat_a" in coeffs
        assert "feat_b" in coeffs

    def test_no_common_features_returns_none(self):
        """If NVDA and peer features have no overlap, should return (None, {})."""
        from src.ml_models import MLDriverModel
        model_mgr = MLDriverModel()
        features = pd.DataFrame({"nvda_only_feat": [1, 2, 3]})
        target = pd.Series([0.1, 0.2, 0.3])
        peer_data = pd.DataFrame({
            "peer_only_feat": [4, 5, 6],
            "target_value": [0.4, 0.5, 0.6],
            "ticker": ["AMD", "AMD", "AMD"],
        })
        model, coeffs = model_mgr.train_peer_panel_model(features, target, peer_data=peer_data)
        assert model is None
        assert coeffs == {}
