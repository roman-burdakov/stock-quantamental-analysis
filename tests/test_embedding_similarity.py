"""
Tests for sentence-transformer embedding distances (Task 17.1).

Covers:
  - Embedding similarity computation with mock/real data
  - Graceful fallback when sentence-transformers is not installed
  - Output format matches NLPFeatureRecord schema
  - Integration with compute_narrative_drift when use_embeddings=True
  - Edge cases: single filing, empty text, identical texts

All tests run offline using inline TextSectionRecord fixtures.
Reqs: 7.10
"""

from __future__ import annotations

import importlib
from unittest import mock

import pandas as pd
import pytest

from src.config import EngineConfig, NLPFeatureRecord, TextSectionRecord
from src.nlp_features import NLPFeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures
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


RISK_TEXT_A = (
    "The company faces significant risks from export controls imposed by the "
    "Bureau of Industry and Security on advanced computing chips to China. "
    "Supply constraints at TSMC foundry limit our capacity for Hopper and "
    "Blackwell architecture GPUs. Competition from AMD, Intel, and custom "
    "silicon such as Google TPU and Amazon Trainium continues to intensify."
)

RISK_TEXT_B = (
    "Export controls and trade restrictions continue to adversely affect our "
    "business in China and other restricted markets. The Bureau of Industry "
    "and Security has imposed additional restrictions on advanced computing "
    "semiconductors. Supply chain disruptions and capacity constraints at our "
    "foundry partners including TSMC could limit production of Blackwell."
)

MDA_TEXT_A = (
    "Total revenue for fiscal year 2024 was $60.9 billion driven by Data "
    "Center revenue growth. Gross margin improved to 72.7 percent reflecting "
    "higher mix of data center revenue. Operating cash flow was $28.1 billion."
)

MDA_TEXT_B = (
    "Total revenue for fiscal year 2025 was $130.5 billion an increase of "
    "114 percent from fiscal year 2024. Data Center revenue was $115.2 billion "
    "representing 88 percent of total revenue. Gross margin was 75.0 percent."
)

# Completely different text for testing dissimilarity
UNRELATED_TEXT = (
    "The weather forecast for tomorrow calls for sunny skies with temperatures "
    "reaching 75 degrees Fahrenheit. Winds will be light from the southwest "
    "at 5 to 10 miles per hour. No precipitation is expected this week."
)


@pytest.fixture
def extractor() -> NLPFeatureExtractor:
    return NLPFeatureExtractor()


@pytest.fixture
def sample_sections() -> list[TextSectionRecord]:
    """Two consecutive filings with risk_factors and mda sections."""
    return [
        _make_section(RISK_TEXT_A, "risk_factors", "2024-02-28", "2024-02-28", "acc-2024"),
        _make_section(MDA_TEXT_A, "mda", "2024-02-28", "2024-02-28", "acc-2024"),
        _make_section(RISK_TEXT_B, "risk_factors", "2025-02-26", "2025-02-26", "acc-2025"),
        _make_section(MDA_TEXT_B, "mda", "2025-02-26", "2025-02-26", "acc-2025"),
    ]


def _sentence_transformers_available() -> bool:
    """Check if sentence-transformers is importable."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


# ===================================================================
# 1. Embedding similarity computation with mock data
# ===================================================================


class TestEmbeddingSimilarityComputation:
    """Verify compute_embedding_distance produces valid results."""

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_produces_results_for_consecutive_filings(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """Should produce embedding distance rows for risk_factors and mda."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        # Should have rows for both sections (1 pair each)
        sections_present = set(df["section"].unique())
        assert "risk_factors" in sections_present
        assert "mda" in sections_present

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_cosine_distance_values_in_valid_range(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """Cosine distance = 1 - cosine_similarity, should be in [0, 2]."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        for val in df["value"]:
            assert 0.0 <= val <= 2.0, f"Embedding distance {val} outside [0, 2]"

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_similar_texts_have_low_distance(
        self, extractor: NLPFeatureExtractor
    ):
        """Two very similar texts should produce a low cosine distance."""
        sections = [
            _make_section(RISK_TEXT_A, "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(RISK_TEXT_B, "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        df = extractor.compute_embedding_distance(sections)
        assert not df.empty
        # Similar SEC risk factor texts should have distance < 0.5
        assert df["value"].iloc[0] < 0.5

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_dissimilar_texts_have_higher_distance(
        self, extractor: NLPFeatureExtractor
    ):
        """Completely unrelated texts should have higher distance than similar ones."""
        similar_sections = [
            _make_section(RISK_TEXT_A, "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(RISK_TEXT_B, "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        dissimilar_sections = [
            _make_section(RISK_TEXT_A, "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(UNRELATED_TEXT, "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        df_similar = extractor.compute_embedding_distance(similar_sections)
        df_dissimilar = extractor.compute_embedding_distance(dissimilar_sections)
        assert not df_similar.empty
        assert not df_dissimilar.empty
        assert df_dissimilar["value"].iloc[0] > df_similar["value"].iloc[0]

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_identical_texts_have_near_zero_distance(
        self, extractor: NLPFeatureExtractor
    ):
        """Identical texts should produce distance very close to 0."""
        sections = [
            _make_section(RISK_TEXT_A, "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(RISK_TEXT_A, "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        df = extractor.compute_embedding_distance(sections)
        assert not df.empty
        assert df["value"].iloc[0] < 0.01

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_multiple_filing_pairs(self, extractor: NLPFeatureExtractor):
        """Three consecutive filings should produce 2 distance pairs per section."""
        sections = [
            _make_section(RISK_TEXT_A, "risk_factors", "2023-02-28", "2023-02-28", "acc-2023"),
            _make_section(RISK_TEXT_B, "risk_factors", "2024-02-28", "2024-02-28", "acc-2024"),
            _make_section(
                RISK_TEXT_A + " Additional new content for 2025.",
                "risk_factors", "2025-02-26", "2025-02-26", "acc-2025",
            ),
        ]
        df = extractor.compute_embedding_distance(sections)
        assert len(df) == 2  # 2 consecutive pairs
        # Verify chronological ordering via prev_filing_date
        assert df.iloc[0]["prev_filing_date"] == "2023-02-28"
        assert df.iloc[1]["prev_filing_date"] == "2024-02-28"


# ===================================================================
# 2. Graceful fallback when sentence-transformers is not installed
# ===================================================================


class TestGracefulFallback:
    """Verify graceful degradation when sentence-transformers is unavailable."""

    def test_returns_empty_dataframe_when_import_fails(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """Mocking ImportError for sentence_transformers should return empty DF."""
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            # Force re-import failure inside the method
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "sentence_transformers":
                    raise ImportError("Mocked: sentence-transformers not installed")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=mock_import):
                df = extractor.compute_embedding_distance(sample_sections)
                assert isinstance(df, pd.DataFrame)
                assert df.empty

    def test_narrative_drift_works_without_embeddings(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord],
        tmp_path,
    ):
        """compute_narrative_drift should work fine with use_embeddings=False."""
        extractor.config.use_embeddings = False
        extractor.config.processed_dir = tmp_path / "processed"
        df = extractor.compute_narrative_drift(sample_sections)
        assert not df.empty
        # Should have tfidf and keyword features but NOT embedding
        feature_types = set(df["feature_type"].unique())
        assert "tfidf_similarity" in feature_types
        assert "keyword_score" in feature_types
        assert "embedding_distance" not in feature_types

    def test_narrative_drift_skips_embeddings_when_library_missing(
        self, sample_sections: list[TextSectionRecord], tmp_path
    ):
        """With use_embeddings=True but library missing, should skip gracefully."""
        config = EngineConfig()
        config.use_embeddings = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("Mocked: sentence-transformers not installed")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            df = ext.compute_narrative_drift(sample_sections)
            assert not df.empty
            # Should still have tfidf and keyword features
            feature_types = set(df["feature_type"].unique())
            assert "tfidf_similarity" in feature_types
            assert "keyword_score" in feature_types


# ===================================================================
# 3. Output format matches NLPFeatureRecord schema
# ===================================================================


class TestNLPFeatureRecordSchema:
    """Verify embedding distance output conforms to NLPFeatureRecord schema."""

    # NLPFeatureRecord fields:
    # filing_date, source_available_date, source_accession, section,
    # feature_type, feature_name, value, prev_filing_date

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

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_all_nlp_feature_record_columns_present(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """Output DataFrame must have all NLPFeatureRecord columns."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        assert self.EXPECTED_COLUMNS.issubset(set(df.columns))

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_feature_type_is_embedding_distance(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """feature_type should be 'embedding_distance' for all rows."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        assert (df["feature_type"] == "embedding_distance").all()

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_feature_name_is_cosine_distance(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """feature_name should be 'cosine_distance' for all rows."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        assert (df["feature_name"] == "cosine_distance").all()

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_value_is_numeric(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """value column should contain numeric (float) values."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        assert pd.api.types.is_numeric_dtype(df["value"])

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_prev_filing_date_populated(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """prev_filing_date should be populated for all embedding distance rows."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        assert df["prev_filing_date"].notna().all()

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_section_values_are_target_sections(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """section should be one of 'risk_factors' or 'mda'."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        valid_sections = {"risk_factors", "mda"}
        assert set(df["section"].unique()).issubset(valid_sections)

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_rows_can_construct_nlp_feature_records(
        self, extractor: NLPFeatureExtractor, sample_sections: list[TextSectionRecord]
    ):
        """Each row should be convertible to an NLPFeatureRecord dataclass."""
        df = extractor.compute_embedding_distance(sample_sections)
        assert not df.empty
        for _, row in df.iterrows():
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
            assert record.feature_type == "embedding_distance"
            assert isinstance(record.value, float)


# ===================================================================
# 4. Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case handling for embedding distance computation."""

    def test_single_filing_returns_empty(self, extractor: NLPFeatureExtractor):
        """Need at least 2 filings for pairwise comparison."""
        sections = [_make_section(RISK_TEXT_A, "risk_factors")]
        df = extractor.compute_embedding_distance(sections)
        assert isinstance(df, pd.DataFrame)
        assert df.empty or len(df) == 0

    def test_empty_text_sections_skipped(self, extractor: NLPFeatureExtractor):
        """Sections with empty text should be filtered out."""
        sections = [
            _make_section("", "risk_factors", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(RISK_TEXT_A, "risk_factors", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        df = extractor.compute_embedding_distance(sections)
        # Only 1 non-empty section → no pairs → empty
        assert df.empty or len(df) == 0

    def test_no_target_sections_returns_empty(self, extractor: NLPFeatureExtractor):
        """Sections that aren't risk_factors or mda should produce no results."""
        sections = [
            _make_section(RISK_TEXT_A, "business", "2024-02-28", "2024-02-28", "acc-1"),
            _make_section(RISK_TEXT_B, "business", "2025-02-26", "2025-02-26", "acc-2"),
        ]
        df = extractor.compute_embedding_distance(sections)
        assert df.empty or len(df) == 0

    def test_empty_sections_list(self, extractor: NLPFeatureExtractor):
        """Empty input list should return empty DataFrame."""
        df = extractor.compute_embedding_distance([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ===================================================================
# 5. Integration with narrative drift pipeline
# ===================================================================


class TestNarrativeDriftIntegration:
    """Verify embedding distances integrate correctly into compute_narrative_drift."""

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_embedding_rows_appear_in_narrative_drift(
        self, sample_sections: list[TextSectionRecord], tmp_path
    ):
        """When use_embeddings=True, narrative drift output includes embedding rows."""
        config = EngineConfig()
        config.use_embeddings = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(sample_sections)
        assert "embedding_distance" in df["feature_type"].values

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_embedding_rows_have_correct_schema_in_drift(
        self, sample_sections: list[TextSectionRecord], tmp_path
    ):
        """Embedding rows within narrative drift should match NLPFeatureRecord schema."""
        config = EngineConfig()
        config.use_embeddings = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        df = ext.compute_narrative_drift(sample_sections)
        emb_rows = df[df["feature_type"] == "embedding_distance"]
        assert not emb_rows.empty
        expected_cols = {
            "filing_date", "source_available_date", "source_accession",
            "section", "feature_type", "feature_name", "value", "prev_filing_date",
        }
        assert expected_cols.issubset(set(emb_rows.columns))

    @pytest.mark.skipif(
        not _sentence_transformers_available(),
        reason="sentence-transformers not installed",
    )
    def test_csv_output_includes_embedding_rows(
        self, sample_sections: list[TextSectionRecord], tmp_path
    ):
        """The saved CSV should include embedding distance rows."""
        config = EngineConfig()
        config.use_embeddings = True
        config.processed_dir = tmp_path / "processed"
        ext = NLPFeatureExtractor(config)
        ext.compute_narrative_drift(sample_sections)

        csv_path = tmp_path / "processed" / "nvda_nlp_features.csv"
        assert csv_path.exists()
        saved_df = pd.read_csv(csv_path)
        assert "embedding_distance" in saved_df["feature_type"].values
