"""
NVDA Quantamental Engine — SEC EDGAR & Market Data Fetcher.

Handles all external data retrieval: SEC submissions, companyfacts,
filing documents, market prices (yfinance), and peer financials.
Every download is cached, rate-limited, and logged to provenance.

Reqs: 1.1–1.8, 2.1–2.8
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from src.config import EngineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEC_BASE = "https://data.sec.gov"
_SEC_ARCHIVES_BASE = "https://www.sec.gov"  # Filing documents are on www.sec.gov
_SUBMISSIONS_URL = f"{_SEC_BASE}/submissions/CIK{{cik}}.json"
_COMPANYFACTS_URL = f"{_SEC_BASE}/api/xbrl/companyfacts/CIK{{cik}}.json"
_FILING_ARCHIVES_URL = f"{_SEC_ARCHIVES_BASE}/Archives/edgar/data/{{cik}}/{{accession_no_dash}}/{{primary_doc}}"

_ALLOWED_FORMS = {"10-K", "10-Q"}


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to *path* atomically via temp file + rename.

    This prevents cache corruption if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to *path* atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class EdgarFetcher:
    """Retrieves and caches SEC filings, XBRL facts, market prices, and peer data."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.0 / max(config.sec_rate_limit_rps, 1)
        self._session = requests.Session()
        ua = config.sec_user_agent
        if not ua or "@" not in ua:
            logger.warning(
                "SEC_USER_AGENT is not set or missing contact email. "
                "SEC EDGAR requires a valid User-Agent with contact info. "
                "Set SEC_USER_AGENT='YourApp your@email.com' in .env or environment. "
                "Using placeholder — requests may be rate-limited or rejected."
            )
            ua = ua or "NVDAQuantamentalEngine contact@example.com"
        self._session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_with_retry(self, url: str) -> requests.Response:
        """GET *url* with rate-limiting, exponential back-off on 429/5xx.

        Enforces max ``sec_rate_limit_rps`` requests per second and retries
        up to ``sec_max_retries`` times with exponential back-off.
        """
        max_retries = self.config.sec_max_retries

        for attempt in range(max_retries + 1):
            # Rate-limit: wait until min_interval has elapsed
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            self._last_request_time = time.monotonic()

            try:
                resp = self._session.get(url, timeout=30)
            except requests.RequestException as exc:
                logger.warning("Request error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                return resp

            if resp.status_code in (429,) or resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "HTTP %d from %s — retrying in %ds (attempt %d/%d)",
                    resp.status_code, url, wait, attempt + 1, max_retries + 1,
                )
                if attempt == max_retries:
                    resp.raise_for_status()
                time.sleep(wait)
            else:
                resp.raise_for_status()

        # Should not reach here, but just in case
        raise requests.HTTPError(f"Failed after {max_retries + 1} attempts: {url}")

    def _is_cached(self, filepath: Path) -> bool:
        """Return True if *filepath* exists and force_refresh is False."""
        if self.config.force_refresh:
            return False
        return filepath.exists() and filepath.stat().st_size > 0

    # ------------------------------------------------------------------
    # SEC Submissions  (Req 1.1, 1.2, 1.5–1.8)
    # ------------------------------------------------------------------

    def fetch_submissions(self, cik: str) -> pd.DataFrame:
        """Retrieve filing metadata from SEC submissions endpoint.

        Filters to 10-K / 10-Q within the configured fiscal-year range
        where ``source_available_date <= report_date``.

        Returns a DataFrame with columns matching :class:`FilingRecord`.
        """
        cik_padded = cik.lstrip("0").zfill(10)
        cache_path = self.config.raw_dir / f"submissions_CIK{cik_padded}.json"
        url = _SUBMISSIONS_URL.format(cik=cik_padded)

        t0 = time.monotonic()
        if self._is_cached(cache_path):
            logger.info("Using cached submissions: %s", cache_path)
            with open(cache_path, "r") as f:
                data = json.load(f)
            duration_ms = int((time.monotonic() - t0) * 1000)
            self.log_provenance(
                "fetch_submissions", url,
                {"cik": cik_padded, "api_endpoint": url},
                http_status=None,
                cache_path=str(cache_path),
                rows_returned=None,
                cache_hit=True,
                retrieval_duration_ms=duration_ms,
            )
        else:
            resp = self._request_with_retry(url)
            data = resp.json()
            _atomic_write_json(cache_path, data)
            duration_ms = int((time.monotonic() - t0) * 1000)
            self.log_provenance(
                "fetch_submissions", url,
                {"cik": cik_padded, "api_endpoint": url},
                http_status=resp.status_code,
                cache_path=str(cache_path),
                rows_returned=None,
                cache_hit=False,
                retrieval_duration_ms=duration_ms,
            )

        # Parse the "recent" filings block
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning("No recent filings found for CIK %s", cik_padded)
            return pd.DataFrame()

        # Also fetch older filing files referenced in filings.files
        older_files = data.get("filings", {}).get("files", [])
        all_blocks = [recent]
        for file_ref in older_files:
            fname = file_ref.get("name")
            if not fname:
                continue
            older_cache = self.config.raw_dir / fname
            older_url = f"https://data.sec.gov/submissions/{fname}"
            t1 = time.monotonic()
            if self._is_cached(older_cache):
                logger.info("Using cached older submissions: %s", older_cache)
                with open(older_cache, "r") as f:
                    older_data = json.load(f)
                dur = int((time.monotonic() - t1) * 1000)
                self.log_provenance(
                    "fetch_older_submissions", older_url,
                    {"cik": cik_padded, "file": fname, "api_endpoint": older_url},
                    http_status=None,
                    cache_path=str(older_cache),
                    cache_hit=True,
                    retrieval_duration_ms=dur,
                )
            else:
                try:
                    older_resp = self._request_with_retry(older_url)
                    older_data = older_resp.json()
                    _atomic_write_json(older_cache, older_data)
                    dur = int((time.monotonic() - t1) * 1000)
                    self.log_provenance(
                        "fetch_older_submissions", older_url,
                        {"cik": cik_padded, "file": fname, "api_endpoint": older_url},
                        http_status=older_resp.status_code,
                        cache_path=str(older_cache),
                        cache_hit=False,
                        retrieval_duration_ms=dur,
                    )
                except Exception as exc:
                    logger.warning("Failed to fetch older submissions %s: %s", fname, exc)
                    dur = int((time.monotonic() - t1) * 1000)
                    self.log_provenance(
                        "fetch_older_submissions", older_url,
                        {"cik": cik_padded, "file": fname, "api_endpoint": older_url},
                        http_status=None,
                        cache_path=str(older_cache),
                        cache_hit=False,
                        retrieval_duration_ms=dur,
                        error=str(exc),
                        fallback_action="skipped older submissions file",
                    )
                    continue
            all_blocks.append(older_data)

        n_total = 0
        rows: list[dict[str, Any]] = []
        for block in all_blocks:
            n = len(block.get("accessionNumber", []))
            for i in range(n):
                form = block["form"][i]
                if form not in _ALLOWED_FORMS:
                    continue

                filing_date = block["filingDate"][i]
                report_period = block["reportDate"][i]

                # source_available_date = filing_date for SEC filings
                source_available_date = filing_date

                # Filter: source_available_date <= report_date
                if source_available_date > self.config.report_date:
                    continue

                # Fiscal year filter — derive FY from report_period
                try:
                    rp = datetime.strptime(report_period, "%Y-%m-%d")
                    fy = rp.year
                    if not (self.config.start_fiscal_year <= fy <= self.config.end_fiscal_year
                            or self.config.start_fiscal_year <= fy + 1 <= self.config.end_fiscal_year):
                        continue
                except ValueError:
                    pass  # keep if we can't parse

                accession = block["accessionNumber"][i]
                primary_doc = block.get("primaryDocument", [""] * n)[i]
                accession_no_dash = accession.replace("-", "")
                cik_numeric = cik_padded.lstrip("0") or "0"
                sec_url = f"{_SEC_ARCHIVES_BASE}/Archives/edgar/data/{cik_numeric}/{accession_no_dash}/{primary_doc}"

                rows.append({
                    "ticker": self.config.ticker,
                    "cik": cik_padded,
                    "accession_number": accession,
                    "filing_date": filing_date,
                    "source_available_date": source_available_date,
                    "report_period": report_period,
                    "form_type": form,
                    "sec_url": sec_url,
                    "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                })

        df = pd.DataFrame(rows)
        logger.info("Fetched %d submissions for CIK %s", len(df), cik_padded)
        return df

    # ------------------------------------------------------------------
    # Company Facts  (Req 1.3)
    # ------------------------------------------------------------------

    def fetch_companyfacts(self, cik: str) -> dict:
        """Retrieve structured XBRL facts, preserving ``source_available_date`` per fact.

        The raw JSON is cached. Each fact's ``filed`` date is treated as
        its ``source_available_date``.
        """
        cik_padded = cik.lstrip("0").zfill(10)
        cache_path = self.config.raw_dir / f"companyfacts_CIK{cik_padded}.json"
        url = _COMPANYFACTS_URL.format(cik=cik_padded)

        t0 = time.monotonic()
        if self._is_cached(cache_path):
            logger.info("Using cached companyfacts: %s", cache_path)
            with open(cache_path, "r") as f:
                data = json.load(f)
            duration_ms = int((time.monotonic() - t0) * 1000)
            # Count total facts for rows_returned
            fact_count = 0
            for taxonomy in data.get("facts", {}).values():
                for concept_data in taxonomy.values():
                    for unit_entries in concept_data.get("units", {}).values():
                        fact_count += len(unit_entries)
            self.log_provenance(
                "fetch_companyfacts", url,
                {"cik": cik_padded, "api_endpoint": url},
                http_status=None,
                cache_path=str(cache_path),
                rows_returned=fact_count,
                cache_hit=True,
                retrieval_duration_ms=duration_ms,
            )
            return data

        resp = self._request_with_retry(url)
        data = resp.json()

        # Annotate each fact with source_available_date = filed date
        fact_count = 0
        for taxonomy in data.get("facts", {}).values():
            for concept_data in taxonomy.values():
                for unit_entries in concept_data.get("units", {}).values():
                    for entry in unit_entries:
                        entry["source_available_date"] = entry.get("filed", "")
                    fact_count += len(unit_entries)

        _atomic_write_json(cache_path, data)
        duration_ms = int((time.monotonic() - t0) * 1000)

        self.log_provenance(
            "fetch_companyfacts", url,
            {"cik": cik_padded, "api_endpoint": url},
            http_status=resp.status_code,
            cache_path=str(cache_path),
            rows_returned=fact_count,
            cache_hit=False,
            retrieval_duration_ms=duration_ms,
        )
        logger.info("Fetched and cached companyfacts for CIK %s", cik_padded)
        return data

    # ------------------------------------------------------------------
    # Filing Document  (Req 1.4)
    # ------------------------------------------------------------------

    def fetch_filing_document(self, accession: str, cik: str) -> str:
        """Download filing HTML and attach metadata.

        Returns the HTML string. Metadata (accession, filing_date,
        source_available_date) is stored in a sidecar JSON file.
        """
        cik_padded = cik.lstrip("0").zfill(10)
        accession_no_dash = accession.replace("-", "")
        safe_accession = accession.replace("/", "_")

        filings_dir = self.config.raw_dir / "filings"
        filings_dir.mkdir(parents=True, exist_ok=True)
        html_path = filings_dir / f"{safe_accession}.html"
        meta_path = filings_dir / f"{safe_accession}_meta.json"

        t0 = time.monotonic()
        if self._is_cached(html_path):
            logger.info("Using cached filing: %s", html_path)
            html = html_path.read_text(encoding="utf-8")
            duration_ms = int((time.monotonic() - t0) * 1000)
            # Resolve URL for provenance even on cache hit
            primary_doc = self._resolve_primary_doc(accession, cik_padded)
            cik_numeric = cik_padded.lstrip("0") or "0"
            url = f"{_SEC_ARCHIVES_BASE}/Archives/edgar/data/{cik_numeric}/{accession_no_dash}/{primary_doc}"
            filing_date = self._resolve_filing_date(accession, cik_padded)
            self.log_provenance(
                "fetch_filing_document", url,
                {
                    "accession_number": accession,
                    "cik": cik_padded,
                    "api_endpoint": url,
                    "filing_date": filing_date,
                    "source_available_date": filing_date,
                },
                http_status=None,
                cache_path=str(html_path),
                rows_returned=1,
                cache_hit=True,
                retrieval_duration_ms=duration_ms,
            )
            return html

        # We need the primary document name — try to look it up from submissions
        primary_doc = self._resolve_primary_doc(accession, cik_padded)
        cik_numeric = cik_padded.lstrip("0") or "0"
        url = f"{_SEC_ARCHIVES_BASE}/Archives/edgar/data/{cik_numeric}/{accession_no_dash}/{primary_doc}"

        resp = self._request_with_retry(url)
        html = resp.text

        _atomic_write_text(html_path, html)

        # Resolve filing_date from submissions cache if available
        filing_date = self._resolve_filing_date(accession, cik_padded)
        duration_ms = int((time.monotonic() - t0) * 1000)
        metadata = {
            "accession_number": accession,
            "cik": cik_padded,
            "api_endpoint": url,
            "filing_date": filing_date,
            "source_available_date": filing_date,
            "url": url,
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(meta_path, metadata)

        self.log_provenance(
            "fetch_filing_document", url, metadata,
            http_status=resp.status_code,
            cache_path=str(html_path),
            rows_returned=1,
            cache_hit=False,
            retrieval_duration_ms=duration_ms,
        )
        logger.info("Fetched filing document: %s", accession)
        return html

    def _resolve_primary_doc(self, accession: str, cik_padded: str) -> str:
        """Best-effort lookup of the primary document filename."""
        cache_path = self.config.raw_dir / f"submissions_CIK{cik_padded}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
            recent = data.get("filings", {}).get("recent", {})
            accessions = recent.get("accessionNumber", [])
            docs = recent.get("primaryDocument", [])
            for i, acc in enumerate(accessions):
                if acc == accession and i < len(docs):
                    return docs[i]
        # Fallback: construct a plausible name
        return "index.html"

    def _resolve_filing_date(self, accession: str, cik_padded: str) -> str:
        """Best-effort lookup of filing_date from cached submissions."""
        cache_path = self.config.raw_dir / f"submissions_CIK{cik_padded}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
            recent = data.get("filings", {}).get("recent", {})
            accessions = recent.get("accessionNumber", [])
            dates = recent.get("filingDate", [])
            for i, acc in enumerate(accessions):
                if acc == accession and i < len(dates):
                    return dates[i]
        return ""

    # ------------------------------------------------------------------
    # Market Prices  (Req 2.1–2.4)
    # ------------------------------------------------------------------

    def fetch_market_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Retrieve daily adjusted close via yfinance.

        Applies LOCF for gaps ≤ 3 calendar days (weekends/holidays).
        Longer gaps are logged and excluded. Caches result as CSV.

        Returns a DataFrame with columns: date, ticker, adj_close,
        source_available_date.
        """
        cache_path = self.config.raw_dir / "market_prices.csv"

        t0 = time.monotonic()
        if self._is_cached(cache_path):
            logger.info("Using cached market prices: %s", cache_path)
            df = pd.read_csv(cache_path, parse_dates=["date"])
            duration_ms = int((time.monotonic() - t0) * 1000)
            self.log_provenance(
                "fetch_market_prices", "yfinance",
                {
                    "tickers": tickers,
                    "date_range_start": start,
                    "date_range_end": end,
                    "provider": "yfinance",
                },
                http_status=None,
                cache_path=str(cache_path),
                rows_returned=len(df),
                cache_hit=True,
                retrieval_duration_ms=duration_ms,
            )
            return df

        all_frames: list[pd.DataFrame] = []

        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start=start, end=end, auto_adjust=True)
                if hist.empty:
                    logger.warning("No price data for %s", ticker)
                    continue

                hist = hist.reset_index()
                hist = hist.rename(columns={"Date": "date", "Close": "adj_close"})
                hist["ticker"] = ticker
                hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
                hist["source_available_date"] = hist["date"].dt.strftime("%Y-%m-%d")
                hist = hist[["date", "ticker", "adj_close", "source_available_date"]]
                all_frames.append(hist)
            except Exception as exc:
                logger.warning("Failed to fetch prices for %s: %s", ticker, exc)
                self.log_provenance(
                    "fetch_market_prices", "yfinance",
                    {
                        "tickers": [ticker],
                        "date_range_start": start,
                        "date_range_end": end,
                        "provider": "yfinance",
                    },
                    http_status=None,
                    cache_path=str(cache_path),
                    rows_returned=0,
                    cache_hit=False,
                    retrieval_duration_ms=int((time.monotonic() - t0) * 1000),
                    error=str(exc),
                    fallback_action=f"skipped ticker {ticker}",
                )

        if not all_frames:
            logger.warning("No market price data retrieved")
            self.log_provenance(
                "fetch_market_prices", "yfinance",
                {
                    "tickers": tickers,
                    "date_range_start": start,
                    "date_range_end": end,
                    "provider": "yfinance",
                },
                http_status=None,
                cache_path=str(cache_path),
                rows_returned=0,
                cache_hit=False,
                retrieval_duration_ms=int((time.monotonic() - t0) * 1000),
                error="No market price data retrieved for any ticker",
                fallback_action="returned empty DataFrame",
            )
            return pd.DataFrame(columns=["date", "ticker", "adj_close", "source_available_date"])

        df = pd.concat(all_frames, ignore_index=True)

        # Log partial failure if some tickers were missing
        retrieved_tickers = set(df["ticker"].unique())
        missing_tickers = [t for t in tickers if t not in retrieved_tickers]
        if missing_tickers:
            self.log_provenance(
                "fetch_market_prices", "yfinance",
                {
                    "tickers": missing_tickers,
                    "date_range_start": start,
                    "date_range_end": end,
                    "provider": "yfinance",
                },
                http_status=None,
                cache_path=str(cache_path),
                rows_returned=0,
                cache_hit=False,
                retrieval_duration_ms=int((time.monotonic() - t0) * 1000),
                error=f"No price data returned for tickers: {', '.join(missing_tickers)}",
                fallback_action=f"continued with {len(retrieved_tickers)}/{len(tickers)} tickers",
            )

        # Apply LOCF with max 3 calendar-day gap
        df = self._apply_locf(df, max_gap_days=3)

        # Filter: source_available_date <= price_date
        df = df[df["source_available_date"] <= self.config.price_date].copy()

        _atomic_write_text(cache_path, df.to_csv(index=False))
        duration_ms = int((time.monotonic() - t0) * 1000)

        self.log_provenance(
            "fetch_market_prices", "yfinance",
            {
                "tickers": tickers,
                "date_range_start": start,
                "date_range_end": end,
                "provider": "yfinance",
            },
            http_status=200,
            cache_path=str(cache_path),
            rows_returned=len(df),
            cache_hit=False,
            retrieval_duration_ms=duration_ms,
        )
        logger.info("Fetched market prices: %d rows for %d tickers", len(df), len(tickers))
        return df

    def _apply_locf(self, df: pd.DataFrame, max_gap_days: int = 3) -> pd.DataFrame:
        """Last-observation-carried-forward for gaps ≤ *max_gap_days*.

        Longer gaps are logged and the interpolated rows are dropped.
        """
        if df.empty:
            return df

        result_frames: list[pd.DataFrame] = []

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date").copy()

            # Build a full calendar-day index between min and max date
            full_idx = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
            group = group.set_index("date").reindex(full_idx)
            group.index.name = "date"
            group["ticker"] = ticker

            # Identify gaps
            is_original = group["adj_close"].notna()
            group["adj_close"] = group["adj_close"].ffill()
            group["source_available_date"] = group["source_available_date"].ffill()

            # Check gap lengths — drop rows where gap > max_gap_days
            gap_start = None
            rows_to_drop: list = []
            for idx in group.index:
                if is_original.loc[idx]:
                    gap_start = None
                else:
                    if gap_start is None:
                        gap_start = idx
                    gap_len = (idx - gap_start).days + 1
                    if gap_len > max_gap_days:
                        rows_to_drop.append(idx)
                        logger.info(
                            "Price gap > %d days for %s at %s (gap day %d) — excluding",
                            max_gap_days, ticker, idx.strftime("%Y-%m-%d"), gap_len,
                        )

            if rows_to_drop:
                group = group.drop(rows_to_drop)

            group = group.dropna(subset=["adj_close"]).reset_index()
            group = group.rename(columns={"index": "date"})
            group = group[["date", "ticker", "adj_close", "source_available_date"]]
            result_frames.append(group)

        return pd.concat(result_frames, ignore_index=True) if result_frames else df

    # ------------------------------------------------------------------
    # Peer Financials  (Req 2.5–2.7)
    # ------------------------------------------------------------------

    def fetch_peer_financials(self, tickers: list[str]) -> pd.DataFrame:
        """Retrieve peer financial snapshots via yfinance.

        Collects: market_cap, total_debt, total_cash, revenue, EBITDA,
        net_income, free_cash_flow, source_date, source_available_date.

        Applies staleness check: if ``source_date`` is more than
        ``peer_staleness_threshold_days`` before ``report_date``,
        EV-based multiples are flagged for exclusion.
        """
        cache_path = self.config.raw_dir / "peer_financials.csv"

        _PEER_FIELDS = [
            "market_cap", "total_debt", "total_cash", "revenue",
            "ebitda", "net_income", "free_cash_flow",
        ]

        t0 = time.monotonic()
        if self._is_cached(cache_path):
            logger.info("Using cached peer financials: %s", cache_path)
            df = pd.read_csv(cache_path)
            duration_ms = int((time.monotonic() - t0) * 1000)
            # Log per-ticker provenance for cached data
            for _, row in df.iterrows():
                tk = row.get("ticker", "")
                sd = row.get("source_date", "")
                staleness = None
                if sd:
                    try:
                        sd_dt = datetime.strptime(str(sd), "%Y-%m-%d")
                        rd_dt = datetime.strptime(self.config.report_date, "%Y-%m-%d")
                        staleness = (rd_dt - sd_dt).days
                    except ValueError:
                        pass
                self.log_provenance(
                    "fetch_peer_financials", "yfinance",
                    {
                        "ticker": tk,
                        "tickers": [tk],
                        "provider": "yfinance",
                        "fields_retrieved": _PEER_FIELDS,
                        "source_date": str(sd),
                        "staleness_days": staleness,
                    },
                    http_status=None,
                    cache_path=str(cache_path),
                    rows_returned=1,
                    cache_hit=True,
                    retrieval_duration_ms=duration_ms,
                )
            return df

        rows: list[dict[str, Any]] = []
        retrieval_ts = datetime.now(timezone.utc).isoformat()

        for ticker in tickers:
            t_peer = time.monotonic()
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info or {}

                market_cap = info.get("marketCap")
                total_debt = info.get("totalDebt")
                total_cash = info.get("totalCash")
                revenue = info.get("totalRevenue")
                ebitda = info.get("ebitda")
                net_income = info.get("netIncomeToCommon")
                fcf = info.get("freeCashflow")

                # source_date: most recent fiscal data date from yfinance
                source_date = info.get("mostRecentQuarter", "")
                if source_date and isinstance(source_date, (int, float)):
                    source_date = datetime.utcfromtimestamp(source_date).strftime("%Y-%m-%d")
                elif not isinstance(source_date, str):
                    source_date = ""

                source_available_date = retrieval_ts[:10]  # retrieval date

                # Staleness check
                stale_ev = False
                staleness_days: int | None = None
                if source_date:
                    try:
                        sd = datetime.strptime(source_date, "%Y-%m-%d")
                        rd = datetime.strptime(self.config.report_date, "%Y-%m-%d")
                        staleness_days = (rd - sd).days
                        if staleness_days > self.config.peer_staleness_threshold_days:
                            stale_ev = True
                            logger.warning(
                                "Peer %s EV data stale: source_date=%s, report_date=%s (>%d days)",
                                ticker, source_date, self.config.report_date,
                                self.config.peer_staleness_threshold_days,
                            )
                    except ValueError:
                        stale_ev = True
                else:
                    stale_ev = True

                rows.append({
                    "ticker": ticker,
                    "market_cap": market_cap,
                    "total_debt": total_debt,
                    "total_cash": total_cash,
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "net_income": net_income,
                    "free_cash_flow": fcf,
                    "source_date": source_date,
                    "source_available_date": source_available_date,
                    "stale_ev": stale_ev,
                    "retrieval_timestamp": retrieval_ts,
                })

                peer_dur = int((time.monotonic() - t_peer) * 1000)
                self.log_provenance(
                    "fetch_peer_financials", "yfinance",
                    {
                        "ticker": ticker,
                        "tickers": [ticker],
                        "provider": "yfinance",
                        "fields_retrieved": _PEER_FIELDS,
                        "source_date": source_date,
                        "staleness_days": staleness_days,
                    },
                    http_status=200,
                    cache_path=str(cache_path),
                    rows_returned=1,
                    cache_hit=False,
                    retrieval_duration_ms=peer_dur,
                )
            except Exception as exc:
                logger.warning("Failed to fetch peer financials for %s: %s", ticker, exc)
                peer_dur = int((time.monotonic() - t_peer) * 1000)
                self.log_provenance(
                    "fetch_peer_financials", "yfinance",
                    {
                        "ticker": ticker,
                        "tickers": [ticker],
                        "provider": "yfinance",
                        "fields_retrieved": _PEER_FIELDS,
                    },
                    http_status=None,
                    cache_path=str(cache_path),
                    rows_returned=0,
                    cache_hit=False,
                    retrieval_duration_ms=peer_dur,
                    error=str(exc),
                    fallback_action=f"skipped peer {ticker}",
                )

        df = pd.DataFrame(rows)

        _atomic_write_text(cache_path, df.to_csv(index=False))

        logger.info("Fetched peer financials: %d peers", len(df))
        return df

    # ------------------------------------------------------------------
    # Provenance logging  (Req 1.8, 24.1–24.4)
    # ------------------------------------------------------------------

    def log_provenance(
        self,
        step: str,
        url: str,
        metadata: dict[str, Any] | None = None,
        *,
        http_status: int | None = None,
        cache_path: str | None = None,
        rows_returned: int | None = None,
        cache_hit: bool = False,
        retrieval_duration_ms: int | None = None,
        error: str | None = None,
        fallback_action: str | None = None,
    ) -> None:
        """Append a provenance entry to ``provenance_log.jsonl``.

        Each line records: timestamp, step, source_url, and the standard
        retrieval-detail fields required by Req 24.4:

        - **http_status** (int | None): HTTP status code (None for cache hits)
        - **cache_path** (str | None): local path to cached file
        - **rows_returned** (int | None): number of rows/records returned
        - **cache_hit** (bool): whether data was loaded from cache
        - **retrieval_duration_ms** (int | None): wall-clock retrieval time
        - **error** (str | None): error message if retrieval failed (Req 24.7)
        - **fallback_action** (str | None): what the pipeline did instead
          on failure (Req 24.7)

        Plus any extra *metadata* (accession_number, filing_date,
        source_available_date, report_period, cik, tickers, etc.).

        Uses write-then-flush to minimize corruption risk on crash.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "source_url": url,
            "http_status": http_status,
            "cache_path": cache_path,
            "rows_returned": rows_returned,
            "cache_hit": cache_hit,
            "retrieval_duration_ms": retrieval_duration_ms,
        }
        if error is not None:
            entry["error"] = error
        if fallback_action is not None:
            entry["fallback_action"] = fallback_action
        if metadata:
            entry.update(metadata)

        log_path = self.config.provenance_log
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
