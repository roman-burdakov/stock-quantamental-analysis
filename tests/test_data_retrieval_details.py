"""
Tests for data retrieval detail logging and source attribution.

Validates that:
  - provenance_log.jsonl entries contain required fields
    (timestamp, step, source_url, http_status, cache_path)
  - Enhanced fields are present: rows_returned, cache_hit, retrieval_duration_ms
  - SEC EDGAR entries include api_endpoint and cik
  - Market data entries include tickers, date_range, provider
  - Peer financial entries include ticker, fields_retrieved, source_date, staleness_days
  - Error entries include error and fallback_action
  - source_attribution.md contains a Data Retrieval Summary table
  - Cache hits are distinguished from live retrievals

Reqs: 24.1–24.7
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.audit_utils import AuditModule
from src.config import EngineConfig
from src.edgar_fetch import EdgarFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fetcher_config(tmp_path: Path, **overrides) -> EngineConfig:
    """Return an EngineConfig with raw_dir and provenance_log in tmp_path."""
    defaults = dict(
        ticker="NVDA",
        cik="0001045810",
        start_fiscal_year=2024,
        end_fiscal_year=2026,
        report_date="2026-04-25",
        price_date="2026-04-25",
        raw_dir=tmp_path / "raw",
        force_refresh=True,
        peer_staleness_threshold_days=90,
    )
    defaults.update(overrides)
    cfg = EngineConfig(**defaults)
    cfg.provenance_log = tmp_path / "raw" / "provenance_log.jsonl"
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    return cfg


def _make_audit_config(tmp_path: Path) -> EngineConfig:
    """Return an EngineConfig with outputs_dir in tmp_path."""
    cfg = EngineConfig(
        ticker="NVDA",
        cik="0001045810",
        report_date="2026-04-25",
        price_date="2026-04-25",
    )
    cfg.outputs_dir = tmp_path / "outputs"
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _read_provenance(cfg: EngineConfig) -> list[dict[str, Any]]:
    """Read all provenance entries from the JSONL log."""
    entries: list[dict[str, Any]] = []
    if cfg.provenance_log.exists():
        for line in cfg.provenance_log.read_text().strip().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Required fields for every provenance entry (Req 24.4)
# ---------------------------------------------------------------------------

_REQUIRED_BASE_FIELDS = {"timestamp", "step", "source_url", "http_status", "cache_path"}
_ENHANCED_FIELDS = {"rows_returned", "cache_hit", "retrieval_duration_ms"}


# ===================================================================
# 1. Provenance entries contain required base fields
# ===================================================================


class TestProvenanceRequiredFields:
    """Verify that every provenance entry has the required base fields."""

    def test_base_fields_present_on_live_retrieval(self, tmp_path):
        """A live retrieval entry has timestamp, step, source_url, http_status, cache_path."""
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_submissions",
            "https://data.sec.gov/submissions/CIK0001045810.json",
            {"cik": "0001045810", "api_endpoint": "https://data.sec.gov/submissions/CIK0001045810.json"},
            http_status=200,
            cache_path="data/raw/submissions.json",
            rows_returned=45,
            cache_hit=False,
            retrieval_duration_ms=350,
        )

        entries = _read_provenance(cfg)
        assert len(entries) == 1
        entry = entries[0]

        for field in _REQUIRED_BASE_FIELDS:
            assert field in entry, f"Missing required field: {field}"

    def test_enhanced_fields_present(self, tmp_path):
        """Enhanced fields (rows_returned, cache_hit, retrieval_duration_ms) are present."""
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_companyfacts",
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
            http_status=200,
            cache_path="data/raw/companyfacts.json",
            rows_returned=1200,
            cache_hit=False,
            retrieval_duration_ms=800,
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        for field in _ENHANCED_FIELDS:
            assert field in entry, f"Missing enhanced field: {field}"

    def test_cache_hit_entry_has_required_fields(self, tmp_path):
        """A cache-hit entry still has all required fields (http_status may be None)."""
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_submissions",
            "https://data.sec.gov/submissions/CIK0001045810.json",
            http_status=None,
            cache_path="data/raw/submissions.json",
            rows_returned=45,
            cache_hit=True,
            retrieval_duration_ms=5,
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        for field in _REQUIRED_BASE_FIELDS:
            assert field in entry, f"Missing required field on cache hit: {field}"
        assert entry["cache_hit"] is True
        assert entry["http_status"] is None


# ===================================================================
# 2. SEC EDGAR entries include api_endpoint and cik (Req 24.1)
# ===================================================================


class TestSECEdgarProvenanceFields:
    """Verify SEC EDGAR provenance entries include api_endpoint and cik."""

    def test_submissions_entry_has_cik_and_endpoint(self, tmp_path):
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_submissions",
            "https://data.sec.gov/submissions/CIK0001045810.json",
            {"cik": "0001045810", "api_endpoint": "https://data.sec.gov/submissions/CIK0001045810.json"},
            http_status=200,
            cache_path="data/raw/submissions.json",
            rows_returned=45,
            cache_hit=False,
            retrieval_duration_ms=350,
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        assert "cik" in entry, "SEC entry missing cik field"
        assert entry["cik"] == "0001045810"
        assert "api_endpoint" in entry, "SEC entry missing api_endpoint field"
        assert "data.sec.gov" in entry["api_endpoint"]

    def test_companyfacts_entry_has_cik_and_endpoint(self, tmp_path):
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_companyfacts",
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
            {"cik": "0001045810", "api_endpoint": "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json"},
            http_status=200,
            cache_path="data/raw/companyfacts.json",
            rows_returned=5000,
            cache_hit=False,
            retrieval_duration_ms=1200,
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        assert entry["cik"] == "0001045810"
        assert "companyfacts" in entry["api_endpoint"]


# ===================================================================
# 3. Market data entries include tickers, date_range, provider (Req 24.2)
# ===================================================================


class TestMarketDataProvenanceFields:
    """Verify market data provenance entries include tickers, date_range, provider."""

    def test_market_prices_entry_has_required_metadata(self, tmp_path):
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_market_prices",
            "yfinance",
            {
                "tickers": ["NVDA", "AMD", "AVGO"],
                "date_range_start": "2016-01-01",
                "date_range_end": "2026-04-25",
                "provider": "yfinance",
            },
            http_status=200,
            cache_path="data/raw/market_prices.csv",
            rows_returned=7500,
            cache_hit=False,
            retrieval_duration_ms=3000,
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        assert "tickers" in entry, "Market data entry missing tickers"
        assert isinstance(entry["tickers"], list)
        assert "NVDA" in entry["tickers"]
        assert "date_range_start" in entry, "Market data entry missing date_range_start"
        assert "date_range_end" in entry, "Market data entry missing date_range_end"
        assert "provider" in entry, "Market data entry missing provider"
        assert entry["provider"] == "yfinance"


# ===================================================================
# 4. Peer financial entries include ticker, fields_retrieved,
#    source_date, staleness_days (Req 24.3)
# ===================================================================


class TestPeerFinancialProvenanceFields:
    """Verify peer financial provenance entries include required metadata."""

    def test_peer_entry_has_required_metadata(self, tmp_path):
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_peer_financials",
            "yfinance",
            {
                "ticker": "AMD",
                "tickers": ["AMD"],
                "provider": "yfinance",
                "fields_retrieved": [
                    "market_cap", "total_debt", "total_cash",
                    "revenue", "ebitda", "net_income", "free_cash_flow",
                ],
                "source_date": "2026-03-31",
                "staleness_days": 25,
            },
            http_status=200,
            cache_path="data/raw/peer_financials.csv",
            rows_returned=1,
            cache_hit=False,
            retrieval_duration_ms=500,
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        assert "ticker" in entry, "Peer entry missing ticker"
        assert entry["ticker"] == "AMD"
        assert "fields_retrieved" in entry, "Peer entry missing fields_retrieved"
        assert isinstance(entry["fields_retrieved"], list)
        assert "market_cap" in entry["fields_retrieved"]
        assert "source_date" in entry, "Peer entry missing source_date"
        assert "staleness_days" in entry, "Peer entry missing staleness_days"
        assert entry["staleness_days"] == 25


# ===================================================================
# 5. Error entries include error and fallback_action (Req 24.7)
# ===================================================================


class TestErrorProvenanceFields:
    """Verify that error provenance entries include error and fallback_action."""

    def test_error_entry_has_error_and_fallback(self, tmp_path):
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_market_prices",
            "yfinance",
            {"tickers": ["NVDA"], "provider": "yfinance"},
            http_status=None,
            cache_path="data/raw/market_prices.csv",
            rows_returned=0,
            cache_hit=False,
            retrieval_duration_ms=100,
            error="API rate limit exceeded",
            fallback_action="skipped ticker NVDA",
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        assert "error" in entry, "Error entry missing error field"
        assert entry["error"] == "API rate limit exceeded"
        assert "fallback_action" in entry, "Error entry missing fallback_action field"
        assert entry["fallback_action"] == "skipped ticker NVDA"

    def test_success_entry_omits_error_fields(self, tmp_path):
        """Successful entries should NOT have error or fallback_action fields."""
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "fetch_submissions",
            "https://data.sec.gov/submissions/CIK0001045810.json",
            http_status=200,
            cache_path="data/raw/submissions.json",
            rows_returned=45,
            cache_hit=False,
            retrieval_duration_ms=350,
        )

        entries = _read_provenance(cfg)
        entry = entries[0]

        assert "error" not in entry
        assert "fallback_action" not in entry


# ===================================================================
# 6. source_attribution.md contains Data Retrieval Summary (Req 24.5, 24.6)
# ===================================================================


class TestDataRetrievalSummaryInAttribution:
    """Verify that _build_data_retrieval_summary produces a proper summary table."""

    def _make_provenance(self) -> list[dict[str, Any]]:
        """Create a representative set of provenance entries."""
        return [
            {
                "timestamp": "2026-05-01T10:00:00+00:00",
                "step": "fetch_submissions",
                "source_url": "https://data.sec.gov/submissions/CIK0001045810.json",
                "http_status": 200,
                "cache_path": "data/raw/submissions_CIK0001045810.json",
                "rows_returned": 45,
                "cache_hit": False,
                "retrieval_duration_ms": 350,
                "cik": "0001045810",
                "api_endpoint": "https://data.sec.gov/submissions/CIK0001045810.json",
            },
            {
                "timestamp": "2026-05-01T10:01:00+00:00",
                "step": "fetch_companyfacts",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
                "http_status": None,
                "cache_path": "data/raw/companyfacts_CIK0001045810.json",
                "rows_returned": 5000,
                "cache_hit": True,
                "retrieval_duration_ms": 10,
                "cik": "0001045810",
            },
            {
                "timestamp": "2026-05-01T10:02:00+00:00",
                "step": "fetch_market_prices",
                "source_url": "yfinance",
                "http_status": 200,
                "cache_path": "data/raw/market_prices.csv",
                "rows_returned": 7500,
                "cache_hit": False,
                "retrieval_duration_ms": 3000,
                "tickers": ["NVDA", "AMD"],
                "provider": "yfinance",
            },
            {
                "timestamp": "2026-05-01T10:03:00+00:00",
                "step": "fetch_peer_financials",
                "source_url": "yfinance",
                "http_status": 200,
                "cache_path": "data/raw/peer_financials.csv",
                "rows_returned": 1,
                "cache_hit": False,
                "retrieval_duration_ms": 500,
                "ticker": "AMD",
                "provider": "yfinance",
                "fields_retrieved": ["market_cap", "revenue"],
                "source_date": "2026-03-31",
                "staleness_days": 25,
            },
        ]

    def test_summary_contains_header(self, tmp_path):
        """Data Retrieval Summary section header is present."""
        audit = AuditModule(_make_audit_config(tmp_path))
        lines = audit._build_data_retrieval_summary(self._make_provenance())
        text = "\n".join(lines)
        assert "## Data Retrieval Summary" in text

    def test_summary_contains_table_columns(self, tmp_path):
        """Summary table has expected column headers."""
        audit = AuditModule(_make_audit_config(tmp_path))
        lines = audit._build_data_retrieval_summary(self._make_provenance())
        text = "\n".join(lines)

        assert "Data Source" in text
        assert "API Calls" in text
        assert "Rows Retrieved" in text
        assert "Cache Hit Rate" in text
        assert "Failures" in text

    def test_summary_includes_all_data_sources(self, tmp_path):
        """Summary table includes rows for each data source type."""
        audit = AuditModule(_make_audit_config(tmp_path))
        lines = audit._build_data_retrieval_summary(self._make_provenance())
        text = "\n".join(lines)

        assert "SEC Submissions" in text
        assert "CompanyFacts" in text or "XBRL" in text
        assert "Market Prices" in text
        assert "Peer Financials" in text

    def test_summary_has_totals_row(self, tmp_path):
        """Summary table includes a totals row."""
        audit = AuditModule(_make_audit_config(tmp_path))
        lines = audit._build_data_retrieval_summary(self._make_provenance())
        text = "\n".join(lines)

        assert "**Total**" in text


# ===================================================================
# 7. Cache hits distinguished from live retrievals (Req 24.5)
# ===================================================================


class TestCacheHitDistinction:
    """Verify that cache hits are distinguished from live retrievals."""

    def test_cache_hit_true_vs_false(self, tmp_path):
        """Provenance entries correctly record cache_hit=True vs False."""
        cfg = _make_fetcher_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        # Live retrieval
        fetcher.log_provenance(
            "fetch_submissions",
            "https://data.sec.gov/submissions/CIK0001045810.json",
            http_status=200,
            cache_path="data/raw/submissions.json",
            rows_returned=45,
            cache_hit=False,
            retrieval_duration_ms=350,
        )

        # Cache hit
        fetcher.log_provenance(
            "fetch_submissions",
            "https://data.sec.gov/submissions/CIK0001045810.json",
            http_status=None,
            cache_path="data/raw/submissions.json",
            rows_returned=45,
            cache_hit=True,
            retrieval_duration_ms=5,
        )

        entries = _read_provenance(cfg)
        assert len(entries) == 2

        live_entry = entries[0]
        cache_entry = entries[1]

        assert live_entry["cache_hit"] is False
        assert cache_entry["cache_hit"] is True

    def test_summary_distinguishes_live_vs_cache(self, tmp_path):
        """The Data Retrieval Summary distinguishes live retrievals from cache loads."""
        audit = AuditModule(_make_audit_config(tmp_path))

        provenance = [
            {
                "timestamp": "2026-05-01T10:00:00+00:00",
                "step": "fetch_submissions",
                "source_url": "https://data.sec.gov/submissions/CIK0001045810.json",
                "http_status": 200,
                "cache_path": "data/raw/submissions.json",
                "rows_returned": 45,
                "cache_hit": False,
                "retrieval_duration_ms": 350,
            },
            {
                "timestamp": "2026-05-01T10:01:00+00:00",
                "step": "fetch_companyfacts",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
                "http_status": None,
                "cache_path": "data/raw/companyfacts.json",
                "rows_returned": 5000,
                "cache_hit": True,
                "retrieval_duration_ms": 10,
            },
        ]

        lines = audit._build_data_retrieval_summary(provenance)
        text = "\n".join(lines)

        # Summary table should show Live Retrievals and Cache Loads columns
        assert "Live Retrieval" in text
        assert "Cache Load" in text

    def test_summary_shows_cache_hit_rate(self, tmp_path):
        """The summary table shows a cache hit rate percentage."""
        audit = AuditModule(_make_audit_config(tmp_path))

        provenance = [
            {
                "timestamp": "2026-05-01T10:00:00+00:00",
                "step": "fetch_submissions",
                "source_url": "https://data.sec.gov/submissions/CIK0001045810.json",
                "http_status": 200,
                "cache_path": "data/raw/submissions.json",
                "rows_returned": 45,
                "cache_hit": False,
                "retrieval_duration_ms": 350,
            },
            {
                "timestamp": "2026-05-01T10:01:00+00:00",
                "step": "fetch_submissions",
                "source_url": "https://data.sec.gov/submissions/CIK0001045810.json",
                "http_status": None,
                "cache_path": "data/raw/submissions.json",
                "rows_returned": 45,
                "cache_hit": True,
                "retrieval_duration_ms": 5,
            },
        ]

        lines = audit._build_data_retrieval_summary(provenance)
        text = "\n".join(lines)

        # 1 out of 2 is a cache hit → 50%
        assert "50%" in text

    def test_all_cache_hits_shows_100_percent(self, tmp_path):
        """When all entries are cache hits, rate should be 100%."""
        audit = AuditModule(_make_audit_config(tmp_path))

        provenance = [
            {
                "timestamp": "2026-05-01T10:00:00+00:00",
                "step": "fetch_submissions",
                "source_url": "https://data.sec.gov/submissions/CIK0001045810.json",
                "http_status": None,
                "cache_path": "data/raw/submissions.json",
                "rows_returned": 45,
                "cache_hit": True,
                "retrieval_duration_ms": 5,
            },
        ]

        lines = audit._build_data_retrieval_summary(provenance)
        text = "\n".join(lines)

        assert "100%" in text
