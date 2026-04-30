"""
Tests for segment revenue normalization.

Validates: Requirements 4.5
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import EngineConfig
from src.segment_revenue import SegmentRevenueNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_NORMALIZED_CATEGORIES = {
    "compute_and_networking",
    "graphics",
    "gpu",
    "tegra",
    "data_center",
    "gaming",
    "professional_visualization",
    "automotive",
    "oem_and_other",
}


def _make_xbrl_row(
    label: str,
    fiscal_year: int = 2024,
    fiscal_period: str = "FY",
    value: float = 1000.0,
) -> dict:
    """Build a single XBRL-style row for testing."""
    return {
        "ticker": "NVDA",
        "fiscal_period": fiscal_period,
        "fiscal_year": fiscal_year,
        "filing_date": f"{fiscal_year}-02-21",
        "source_available_date": f"{fiscal_year}-02-21",
        "accession_number": f"0001045810-{fiscal_year}-000001",
        "original_label": label,
        "value": value,
        "unit": "USD",
        "form_type": "10-K",
    }


# ---------------------------------------------------------------------------
# 1. Label mapping consistency
# ---------------------------------------------------------------------------


class TestLabelMappingConsistency:
    """Every entry in LABEL_MAPPING must resolve to a valid normalized category,
    and common Nvidia labels must map correctly."""

    def test_all_mappings_target_valid_categories(self):
        """Every value in LABEL_MAPPING is a recognized normalized category."""
        for original, normalized in SegmentRevenueNormalizer.LABEL_MAPPING.items():
            assert normalized in VALID_NORMALIZED_CATEGORIES, (
                f"'{original}' maps to unknown category '{normalized}'"
            )

    def test_mapping_keys_are_lowercase(self):
        """Keys should be lowercase for case-insensitive lookup."""
        for key in SegmentRevenueNormalizer.LABEL_MAPPING:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    @pytest.mark.parametrize(
        "label, expected",
        [
            ("Data Center", "data_center"),
            ("Gaming", "gaming"),
            ("Professional Visualization", "professional_visualization"),
            ("Automotive", "automotive"),
            ("OEM and Other", "oem_and_other"),
            ("Compute & Networking", "compute_and_networking"),
        ],
    )
    def test_common_nvidia_labels(self, label, expected):
        """Common Nvidia segment/platform labels normalize correctly."""
        normalizer = SegmentRevenueNormalizer()
        category, _ = normalizer._normalize_label(label)
        assert category == expected

    def test_case_insensitive_lookup(self):
        """Labels with different casing should still resolve."""
        normalizer = SegmentRevenueNormalizer()
        cat_lower, _ = normalizer._normalize_label("data center")
        cat_upper, _ = normalizer._normalize_label("Data Center")
        cat_mixed, _ = normalizer._normalize_label("DATA CENTER")
        assert cat_lower == cat_upper == cat_mixed == "data_center"

    def test_unmapped_label_returns_other(self):
        """A completely unknown label should map to 'other'."""
        normalizer = SegmentRevenueNormalizer()
        category, notes = normalizer._normalize_label("Quantum Computing Division")
        assert category == "other"
        assert "unmapped" in notes.lower()


# ---------------------------------------------------------------------------
# 2. Handling changed labels across fiscal years
# ---------------------------------------------------------------------------


class TestChangedLabelsAcrossFiscalYears:
    """Nvidia renamed segments over time. Both old and new labels must
    normalize correctly, even when mixed in the same dataset."""

    def test_older_gpu_label_normalizes(self):
        """'GPU' (pre-FY2023 segment name) maps to 'gpu'."""
        normalizer = SegmentRevenueNormalizer()
        cat, _ = normalizer._normalize_label("GPU")
        assert cat == "gpu"

    def test_newer_compute_networking_label_normalizes(self):
        """'Compute & Networking' (current segment name) maps correctly."""
        normalizer = SegmentRevenueNormalizer()
        cat, _ = normalizer._normalize_label("Compute & Networking")
        assert cat == "compute_and_networking"

    def test_mixed_old_and_new_labels_in_same_dataset(self, tmp_path):
        """A dataset containing both old ('GPU') and new ('Compute & Networking')
        labels should normalize all rows without errors."""
        rows = [
            # Older fiscal years used "GPU"
            _make_xbrl_row("GPU", fiscal_year=2019, value=3000.0),
            _make_xbrl_row("Tegra Processor", fiscal_year=2019, value=500.0),
            # Newer fiscal years use "Compute & Networking"
            _make_xbrl_row("Compute & Networking", fiscal_year=2024, value=47500.0),
            _make_xbrl_row("Graphics", fiscal_year=2024, value=15000.0),
        ]
        df = pd.DataFrame(rows)

        config = EngineConfig()
        config.processed_dir = tmp_path
        normalizer = SegmentRevenueNormalizer(config)
        result = normalizer.normalize(df)

        assert len(result) == 4
        categories = set(result["normalized_category"])
        assert "gpu" in categories
        assert "compute_and_networking" in categories
        assert "tegra" in categories
        assert "graphics" in categories

    def test_datacenter_alias_consistency(self):
        """Both 'Data Center' and 'Datacenter' map to the same category."""
        normalizer = SegmentRevenueNormalizer()
        cat1, _ = normalizer._normalize_label("Data Center")
        cat2, _ = normalizer._normalize_label("Datacenter")
        assert cat1 == cat2 == "data_center"

    def test_oem_label_variants(self):
        """Multiple OEM label variants all map to 'oem_and_other'."""
        normalizer = SegmentRevenueNormalizer()
        for label in ["OEM and Other", "OEM & Other", "OEM and IP", "All Other"]:
            cat, _ = normalizer._normalize_label(label)
            assert cat == "oem_and_other", f"'{label}' mapped to '{cat}'"

    def test_normalize_output_schema(self, tmp_path):
        """Output DataFrame has the expected SegmentRevenueRecord columns."""
        rows = [_make_xbrl_row("Gaming", fiscal_year=2024, value=10000.0)]
        df = pd.DataFrame(rows)

        config = EngineConfig()
        config.processed_dir = tmp_path
        normalizer = SegmentRevenueNormalizer(config)
        result = normalizer.normalize(df)

        expected_cols = [
            "ticker", "fiscal_period", "fiscal_year", "filing_date",
            "source_available_date", "source_accession", "original_label",
            "normalized_category", "value", "unit", "extraction_method",
            "mapping_notes",
        ]
        assert list(result.columns) == expected_cols
