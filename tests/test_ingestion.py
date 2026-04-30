"""
Tests for src.edgar_fetch — SEC & market data ingestion.

Covers:
  - Submissions filtering (form type, fiscal year, report_date)
  - Caching logic (_is_cached)
  - LOCF ≤ 3 calendar days
  - Peer staleness check skips EV multiples

All tests run offline using fixtures in tests/fixtures/.
Reqs: 12.6, 12.7
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import EngineConfig
from src.edgar_fetch import EdgarFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures")


def _make_config(**overrides) -> EngineConfig:
    """Return an EngineConfig with sensible test defaults."""
    defaults = dict(
        ticker="NVDA",
        cik="0001045810",
        start_fiscal_year=2024,
        end_fiscal_year=2026,
        report_date="2026-04-25",
        price_date="2026-04-25",
        raw_dir=Path("data/raw"),
        force_refresh=False,
        peer_staleness_threshold_days=90,
    )
    defaults.update(overrides)
    return EngineConfig(**defaults)


def _load_submissions_fixture() -> dict:
    with open(FIXTURES / "sample_submissions.json") as f:
        return json.load(f)


# ===================================================================
# 1. Submissions filtering
# ===================================================================


class TestSubmissionsFiltering:
    """Verify fetch_submissions filters by form type, fiscal year, and report_date."""

    def test_filters_to_10k_and_10q_only(self, tmp_path):
        """Only 10-K and 10-Q forms should survive filtering."""
        # Inject an 8-K into the fixture data
        data = _load_submissions_fixture()
        data["filings"]["recent"]["form"].append("8-K")
        data["filings"]["recent"]["accessionNumber"].append("0001045810-24-999999")
        data["filings"]["recent"]["filingDate"].append("2024-06-01")
        data["filings"]["recent"]["reportDate"].append("2024-05-15")
        data["filings"]["recent"]["primaryDocument"].append("doc.htm")
        data["filings"]["recent"]["primaryDocDescription"].append("8-K")
        data["filings"]["recent"]["isXBRL"].append(1)
        data["filings"]["recent"]["items"].append("")

        cache_path = tmp_path / "submissions_CIK0001045810.json"
        cache_path.write_text(json.dumps(data))

        config = _make_config(raw_dir=tmp_path)
        fetcher = EdgarFetcher(config)
        df = fetcher.fetch_submissions("1045810")

        assert not df.empty
        assert set(df["form_type"].unique()).issubset({"10-K", "10-Q"})

    def test_fiscal_year_filter(self, tmp_path):
        """Filings outside the configured fiscal-year range are excluded."""
        data = _load_submissions_fixture()
        cache_path = tmp_path / "submissions_CIK0001045810.json"
        cache_path.write_text(json.dumps(data))

        # Narrow range to FY2025 only
        config = _make_config(raw_dir=tmp_path, start_fiscal_year=2025, end_fiscal_year=2025)
        fetcher = EdgarFetcher(config)
        df = fetcher.fetch_submissions("1045810")

        # The fixture has report dates in 2024 and 2025.
        # FY filter uses rp.year and rp.year+1 — 2025-01-26 → year=2025 matches.
        # 2024-10-27 → year=2024, year+1=2025 → matches.
        # 2024-07-28 → year=2024, year+1=2025 → matches.
        # 2024-01-28 → year=2024, year+1=2025 → matches.
        # All four should match because year+1 == 2025 for 2024 dates.
        for _, row in df.iterrows():
            rp = datetime.strptime(row["report_period"], "%Y-%m-%d")
            fy = rp.year
            assert (2025 <= fy <= 2025) or (2025 <= fy + 1 <= 2025), (
                f"Filing with report_period={row['report_period']} should not pass FY filter"
            )

    def test_report_date_filter(self, tmp_path):
        """Filings with source_available_date > report_date are excluded."""
        data = _load_submissions_fixture()
        cache_path = tmp_path / "submissions_CIK0001045810.json"
        cache_path.write_text(json.dumps(data))

        # Set report_date before the latest filing (2025-02-26)
        config = _make_config(raw_dir=tmp_path, report_date="2025-02-25")
        fetcher = EdgarFetcher(config)
        df = fetcher.fetch_submissions("1045810")

        # The 2025-02-26 filing should be excluded
        for _, row in df.iterrows():
            assert row["source_available_date"] <= "2025-02-25", (
                f"Filing dated {row['source_available_date']} should be excluded"
            )

    def test_all_fixture_filings_returned_with_wide_config(self, tmp_path):
        """With a wide fiscal range and future report_date, all fixture filings pass."""
        data = _load_submissions_fixture()
        cache_path = tmp_path / "submissions_CIK0001045810.json"
        cache_path.write_text(json.dumps(data))

        config = _make_config(
            raw_dir=tmp_path,
            start_fiscal_year=2020,
            end_fiscal_year=2030,
            report_date="2030-01-01",
        )
        fetcher = EdgarFetcher(config)
        df = fetcher.fetch_submissions("1045810")

        # Fixture has 4 filings, all 10-K/10-Q
        assert len(df) == 4


# ===================================================================
# 2. Caching logic
# ===================================================================


class TestCachingLogic:
    """Verify _is_cached behaviour with force_refresh flag."""

    def test_cached_returns_true_when_file_exists(self, tmp_path):
        """_is_cached returns True when file exists and force_refresh=False."""
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text('{"ok": true}')

        config = _make_config(raw_dir=tmp_path, force_refresh=False)
        fetcher = EdgarFetcher(config)

        assert fetcher._is_cached(cache_file) is True

    def test_cached_returns_false_when_force_refresh(self, tmp_path):
        """_is_cached returns False when force_refresh=True even if file exists."""
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text('{"ok": true}')

        config = _make_config(raw_dir=tmp_path, force_refresh=True)
        fetcher = EdgarFetcher(config)

        assert fetcher._is_cached(cache_file) is False

    def test_cached_returns_false_when_file_missing(self, tmp_path):
        """_is_cached returns False when file does not exist."""
        config = _make_config(raw_dir=tmp_path, force_refresh=False)
        fetcher = EdgarFetcher(config)

        assert fetcher._is_cached(tmp_path / "nonexistent.json") is False

    def test_cached_returns_false_for_empty_file(self, tmp_path):
        """_is_cached returns False when file exists but is empty (0 bytes)."""
        cache_file = tmp_path / "empty.json"
        cache_file.write_text("")

        config = _make_config(raw_dir=tmp_path, force_refresh=False)
        fetcher = EdgarFetcher(config)

        assert fetcher._is_cached(cache_file) is False


# ===================================================================
# 3. LOCF only ≤ 3 days
# ===================================================================


class TestLOCF:
    """Verify _apply_locf fills gaps ≤ 3 calendar days and excludes longer gaps."""

    def _build_price_df(self, dates: list[str], ticker: str = "NVDA") -> pd.DataFrame:
        """Helper: build a minimal price DataFrame from date strings."""
        return pd.DataFrame({
            "date": pd.to_datetime(dates),
            "ticker": ticker,
            "adj_close": [100.0 + i for i in range(len(dates))],
            "source_available_date": dates,
        })

    def test_weekend_gap_filled(self):
        """A 2-day weekend gap (Sat-Sun) should be forward-filled."""
        # Fri → Mon (2-day gap)
        df = self._build_price_df(["2025-04-11", "2025-04-14"])

        config = _make_config()
        fetcher = EdgarFetcher(config)
        result = fetcher._apply_locf(df, max_gap_days=3)

        # Should have Fri, Sat, Sun, Mon = 4 rows
        assert len(result) == 4
        # Sat and Sun should carry Friday's close
        sat = result[result["date"] == pd.Timestamp("2025-04-12")]
        assert len(sat) == 1
        assert sat.iloc[0]["adj_close"] == 100.0  # Friday's value

    def test_3_day_gap_filled(self):
        """A 3-day gap should be forward-filled (max allowed)."""
        # Day 0 → Day 4 (3-day gap: days 1, 2, 3 missing)
        df = self._build_price_df(["2025-04-07", "2025-04-11"])

        config = _make_config()
        fetcher = EdgarFetcher(config)
        result = fetcher._apply_locf(df, max_gap_days=3)

        # Days 7, 8, 9, 10, 11 = 5 rows (gap of 3 days: 8, 9, 10)
        assert len(result) == 5

    def test_4_day_gap_excluded(self):
        """A 4-day gap should have the 4th gap day excluded."""
        # Day 0 → Day 5 (4-day gap: days 1, 2, 3, 4 missing)
        df = self._build_price_df(["2025-04-07", "2025-04-12"])

        config = _make_config()
        fetcher = EdgarFetcher(config)
        result = fetcher._apply_locf(df, max_gap_days=3)

        # Days 7, 8, 9, 10 are filled (3-day gap OK), day 11 is excluded (4th gap day)
        # So we get: 7, 8, 9, 10, 12 = 5 rows (day 11 dropped)
        assert len(result) == 5
        dates = set(result["date"].dt.strftime("%Y-%m-%d"))
        assert "2025-04-11" not in dates  # 4th gap day excluded

    def test_long_gap_excluded(self):
        """A gap > 3 days should have excess days excluded."""
        # 7-day gap
        df = self._build_price_df(["2025-04-01", "2025-04-09"])

        config = _make_config()
        fetcher = EdgarFetcher(config)
        result = fetcher._apply_locf(df, max_gap_days=3)

        # Days 1, 2, 3, 4 filled (3-day gap), days 5-8 excluded
        # Result: Apr 1, 2, 3, 4, 9 = 5 rows
        assert len(result) == 5
        dates = sorted(result["date"].dt.strftime("%Y-%m-%d").tolist())
        assert dates == ["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-09"]

    def test_empty_df_returns_empty(self):
        """Empty input should return empty output."""
        df = pd.DataFrame(columns=["date", "ticker", "adj_close", "source_available_date"])
        config = _make_config()
        fetcher = EdgarFetcher(config)
        result = fetcher._apply_locf(df, max_gap_days=3)
        assert result.empty


# ===================================================================
# 4. Peer staleness check skips EV multiples
# ===================================================================


class TestPeerStaleness:
    """Verify stale_ev flag is True when source_date > 90 days before report_date."""

    def test_stale_peer_flagged(self, tmp_path):
        """Peer with source_date > 90 days before report_date gets stale_ev=True."""
        config = _make_config(
            raw_dir=tmp_path,
            report_date="2025-06-01",
            peer_staleness_threshold_days=90,
        )
        fetcher = EdgarFetcher(config)

        # Mock yfinance to return a peer with old source_date
        mock_info = {
            "marketCap": 100_000_000_000,
            "totalDebt": 5_000_000_000,
            "totalCash": 3_000_000_000,
            "totalRevenue": 20_000_000_000,
            "ebitda": 5_000_000_000,
            "netIncomeToCommon": 2_000_000_000,
            "freeCashflow": 3_000_000_000,
            "mostRecentQuarter": "2025-01-15",  # 137 days before 2025-06-01 → stale
        }

        with patch("src.edgar_fetch.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_yf.return_value = mock_ticker

            df = fetcher.fetch_peer_financials(["STALE_PEER"])

        assert len(df) == 1
        assert bool(df.iloc[0]["stale_ev"]) is True

    def test_fresh_peer_not_flagged(self, tmp_path):
        """Peer with source_date within 90 days of report_date gets stale_ev=False."""
        config = _make_config(
            raw_dir=tmp_path,
            report_date="2025-06-01",
            peer_staleness_threshold_days=90,
        )
        fetcher = EdgarFetcher(config)

        mock_info = {
            "marketCap": 100_000_000_000,
            "totalDebt": 5_000_000_000,
            "totalCash": 3_000_000_000,
            "totalRevenue": 20_000_000_000,
            "ebitda": 5_000_000_000,
            "netIncomeToCommon": 2_000_000_000,
            "freeCashflow": 3_000_000_000,
            "mostRecentQuarter": "2025-04-15",  # 47 days before 2025-06-01 → fresh
        }

        with patch("src.edgar_fetch.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_yf.return_value = mock_ticker

            df = fetcher.fetch_peer_financials(["FRESH_PEER"])

        assert len(df) == 1
        assert bool(df.iloc[0]["stale_ev"]) is False

    def test_missing_source_date_flagged_stale(self, tmp_path):
        """Peer with missing source_date gets stale_ev=True."""
        config = _make_config(
            raw_dir=tmp_path,
            report_date="2025-06-01",
        )
        fetcher = EdgarFetcher(config)

        mock_info = {
            "marketCap": 100_000_000_000,
            "totalDebt": 5_000_000_000,
            "totalCash": 3_000_000_000,
            "totalRevenue": 20_000_000_000,
            "ebitda": 5_000_000_000,
            "netIncomeToCommon": 2_000_000_000,
            "freeCashflow": 3_000_000_000,
            # No mostRecentQuarter → source_date = ""
        }

        with patch("src.edgar_fetch.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_yf.return_value = mock_ticker

            df = fetcher.fetch_peer_financials(["NO_DATE_PEER"])

        assert len(df) == 1
        assert bool(df.iloc[0]["stale_ev"]) is True

    def test_boundary_exactly_90_days_not_stale(self, tmp_path):
        """Peer with source_date exactly 90 days before report_date is NOT stale."""
        config = _make_config(
            raw_dir=tmp_path,
            report_date="2025-06-01",
            peer_staleness_threshold_days=90,
        )
        fetcher = EdgarFetcher(config)

        # 90 days before 2025-06-01 = 2025-03-03
        mock_info = {
            "marketCap": 100_000_000_000,
            "totalDebt": 5_000_000_000,
            "totalCash": 3_000_000_000,
            "totalRevenue": 20_000_000_000,
            "ebitda": 5_000_000_000,
            "netIncomeToCommon": 2_000_000_000,
            "freeCashflow": 3_000_000_000,
            "mostRecentQuarter": "2025-03-03",  # exactly 90 days → NOT stale
        }

        with patch("src.edgar_fetch.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_yf.return_value = mock_ticker

            df = fetcher.fetch_peer_financials(["BOUNDARY_PEER"])

        assert len(df) == 1
        assert bool(df.iloc[0]["stale_ev"]) is False
