"""
Tests for retrieval failure logging in EdgarFetcher and source_attribution.md.

Verifies that:
  - Failed retrievals log provenance entries with error and fallback_action fields
  - Partial market price failures log per-ticker errors
  - Partial peer financial failures log per-ticker errors
  - source_attribution.md Retrieval Failures table includes fallback_action column
  - The _build_data_retrieval_summary correctly counts failures

Reqs: 24.7
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import EngineConfig
from src.edgar_fetch import EdgarFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides) -> EngineConfig:
    """Return an EngineConfig pointing at tmp_path for raw_dir."""
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


def _read_provenance(cfg: EngineConfig) -> list[dict[str, Any]]:
    """Read all provenance entries from the log."""
    entries = []
    if cfg.provenance_log.exists():
        for line in cfg.provenance_log.read_text().strip().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


# ===================================================================
# 1. log_provenance accepts error and fallback_action
# ===================================================================


class TestLogProvenanceErrorFields:
    """Verify that log_provenance writes error and fallback_action fields."""

    def test_error_and_fallback_written(self, tmp_path):
        cfg = _make_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "test_step", "https://example.com",
            {"ticker": "NVDA"},
            http_status=500,
            error="Internal Server Error",
            fallback_action="skipped",
        )

        entries = _read_provenance(cfg)
        assert len(entries) == 1
        assert entries[0]["error"] == "Internal Server Error"
        assert entries[0]["fallback_action"] == "skipped"
        assert entries[0]["http_status"] == 500

    def test_no_error_fields_when_success(self, tmp_path):
        cfg = _make_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        fetcher.log_provenance(
            "test_step", "https://example.com",
            http_status=200,
        )

        entries = _read_provenance(cfg)
        assert len(entries) == 1
        assert "error" not in entries[0]
        assert "fallback_action" not in entries[0]


# ===================================================================
# 2. Market price failures log error provenance
# ===================================================================


class TestMarketPriceFailureLogging:
    """Verify that failed market price retrievals log error provenance."""

    @patch("src.edgar_fetch.yf.Ticker")
    def test_single_ticker_failure_logs_error(self, mock_ticker_cls, tmp_path):
        cfg = _make_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        # Make yfinance raise an exception
        mock_ticker_cls.side_effect = RuntimeError("API rate limit exceeded")

        df = fetcher.fetch_market_prices(["NVDA"], "2024-01-01", "2026-04-25")

        # Should return empty DataFrame
        assert df.empty

        entries = _read_provenance(cfg)
        # Should have at least one error entry for the ticker failure
        # and one for the "no data retrieved" overall failure
        error_entries = [e for e in entries if e.get("error")]
        assert len(error_entries) >= 1

        ticker_error = [e for e in error_entries if "NVDA" in str(e.get("fallback_action", "")) or "NVDA" in str(e.get("tickers", ""))]
        assert len(ticker_error) >= 1
        assert "error" in ticker_error[0]
        assert "fallback_action" in ticker_error[0]

    @patch("src.edgar_fetch.yf.Ticker")
    def test_partial_failure_logs_missing_tickers(self, mock_ticker_cls, tmp_path):
        cfg = _make_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        # First ticker succeeds, second fails
        def side_effect(ticker):
            mock = MagicMock()
            if ticker == "NVDA":
                dates = pd.date_range("2024-01-02", periods=3, freq="D")
                mock.history.return_value = pd.DataFrame({
                    "Date": dates,
                    "Close": [100.0, 101.0, 102.0],
                })
                return mock
            else:
                raise RuntimeError(f"Failed for {ticker}")
        mock_ticker_cls.side_effect = side_effect

        df = fetcher.fetch_market_prices(["NVDA", "AMD"], "2024-01-01", "2026-04-25")

        # NVDA should be present
        assert not df.empty
        assert "NVDA" in df["ticker"].values

        entries = _read_provenance(cfg)
        error_entries = [e for e in entries if e.get("error")]
        # Should have error for AMD
        amd_errors = [e for e in error_entries if "AMD" in str(e.get("fallback_action", "")) or "AMD" in str(e.get("tickers", []))]
        assert len(amd_errors) >= 1

    @patch("src.edgar_fetch.yf.Ticker")
    def test_all_tickers_fail_logs_overall_error(self, mock_ticker_cls, tmp_path):
        cfg = _make_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        mock_ticker_cls.side_effect = RuntimeError("Network error")

        df = fetcher.fetch_market_prices(["NVDA", "AMD"], "2024-01-01", "2026-04-25")
        assert df.empty

        entries = _read_provenance(cfg)
        error_entries = [e for e in entries if e.get("error")]
        # Should have per-ticker errors plus the overall "no data" error
        assert len(error_entries) >= 2
        overall = [e for e in error_entries if "returned empty" in str(e.get("fallback_action", ""))]
        assert len(overall) >= 1


# ===================================================================
# 3. Peer financial failures log error provenance
# ===================================================================


class TestPeerFinancialFailureLogging:
    """Verify that failed peer financial retrievals log error provenance."""

    @patch("src.edgar_fetch.yf.Ticker")
    def test_peer_failure_logs_error(self, mock_ticker_cls, tmp_path):
        cfg = _make_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        mock_ticker_cls.side_effect = RuntimeError("Peer API error")

        df = fetcher.fetch_peer_financials(["AMD"])
        assert df.empty

        entries = _read_provenance(cfg)
        error_entries = [e for e in entries if e.get("error")]
        assert len(error_entries) >= 1
        assert "AMD" in str(error_entries[0].get("fallback_action", ""))
        assert error_entries[0]["error"] == "Peer API error"

    @patch("src.edgar_fetch.yf.Ticker")
    def test_partial_peer_failure_logs_per_ticker(self, mock_ticker_cls, tmp_path):
        cfg = _make_config(tmp_path)
        fetcher = EdgarFetcher(cfg)

        def side_effect(ticker):
            mock = MagicMock()
            if ticker == "AMD":
                mock.info = {
                    "marketCap": 200e9,
                    "totalDebt": 5e9,
                    "totalCash": 10e9,
                    "totalRevenue": 25e9,
                    "ebitda": 5e9,
                    "netIncomeToCommon": 3e9,
                    "freeCashflow": 4e9,
                    "mostRecentQuarter": "2026-03-31",
                }
                return mock
            else:
                raise RuntimeError(f"Failed for {ticker}")
        mock_ticker_cls.side_effect = side_effect

        df = fetcher.fetch_peer_financials(["AMD", "INTC"])

        # AMD should succeed
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AMD"

        entries = _read_provenance(cfg)
        error_entries = [e for e in entries if e.get("error")]
        assert len(error_entries) >= 1
        intc_errors = [e for e in error_entries if "INTC" in str(e.get("fallback_action", ""))]
        assert len(intc_errors) == 1


# ===================================================================
# 4. Source attribution Retrieval Failures table includes fallback_action
# ===================================================================


class TestSourceAttributionFailureDisclosure:
    """Verify that source_attribution.md discloses failures with fallback_action."""

    def test_failure_table_includes_fallback_action_column(self, tmp_path):
        from src.audit_utils import AuditModule

        cfg = _make_config(tmp_path)
        cfg.outputs_dir = tmp_path / "outputs"
        cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

        audit = AuditModule(cfg)

        provenance = [
            {
                "timestamp": "2026-05-01T12:00:00+00:00",
                "step": "fetch_market_prices",
                "source_url": "yfinance",
                "http_status": None,
                "cache_path": "data/raw/market_prices.csv",
                "rows_returned": 0,
                "cache_hit": False,
                "retrieval_duration_ms": 100,
                "error": "API rate limit exceeded",
                "fallback_action": "skipped ticker NVDA",
                "tickers": ["NVDA"],
                "provider": "yfinance",
            },
        ]

        lines = audit._build_data_retrieval_summary(provenance)
        text = "\n".join(lines)

        assert "Retrieval Failures" in text
        assert "Fallback Action" in text
        assert "API rate limit exceeded" in text
        assert "skipped ticker NVDA" in text

    def test_no_failures_shows_no_failures_message(self, tmp_path):
        from src.audit_utils import AuditModule

        cfg = _make_config(tmp_path)
        cfg.outputs_dir = tmp_path / "outputs"
        cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

        audit = AuditModule(cfg)

        provenance = [
            {
                "timestamp": "2026-05-01T12:00:00+00:00",
                "step": "fetch_submissions",
                "source_url": "https://data.sec.gov/submissions/CIK0001045810.json",
                "http_status": 200,
                "cache_path": "data/raw/submissions.json",
                "rows_returned": 10,
                "cache_hit": False,
                "retrieval_duration_ms": 500,
            },
        ]

        lines = audit._build_data_retrieval_summary(provenance)
        text = "\n".join(lines)

        assert "No retrieval failures recorded" in text

    def test_http_error_counted_as_failure(self, tmp_path):
        from src.audit_utils import AuditModule

        cfg = _make_config(tmp_path)
        cfg.outputs_dir = tmp_path / "outputs"
        cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

        audit = AuditModule(cfg)

        provenance = [
            {
                "timestamp": "2026-05-01T12:00:00+00:00",
                "step": "fetch_companyfacts",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
                "http_status": 500,
                "cache_path": "data/raw/companyfacts.json",
                "rows_returned": 0,
                "cache_hit": False,
                "retrieval_duration_ms": 200,
                "error": "Internal Server Error",
                "fallback_action": "retried and failed",
            },
        ]

        lines = audit._build_data_retrieval_summary(provenance)
        text = "\n".join(lines)

        assert "Retrieval Failures" in text
        assert "Internal Server Error" in text
        assert "retried and failed" in text
