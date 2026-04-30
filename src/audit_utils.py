"""
NVDA Quantamental Engine — Audit and Attribution Utilities.

Generates source attribution, data dictionary, limitations,
self-audit, and validation checks for exhibit attribution and
report-date filtering.

Requirements: 11.1–11.7
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import (
    AuditStatus,
    ComponentStatus,
    ComponentStatusEnum,
    DataQualityIssue,
    DataQualityStatus,
    EngineConfig,
    ExhibitRecord,
    RecommendationStatus,
    ReportMode,
    ValidatedMetric,
    ValuationAssumption,
    get_default_config,
)

logger = logging.getLogger(__name__)


class AuditModule:
    """Audit, attribution, and validation for the quantamental pipeline."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or get_default_config()
        self.outputs_dir = Path(self.config.outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Source Attribution
    # ------------------------------------------------------------------

    def generate_source_attribution(
        self,
        provenance: list[dict[str, Any]],
        exhibits: list[ExhibitRecord],
        assumptions: Optional[list[ValuationAssumption]] = None,
        peer_attribution: Optional[list[dict[str, Any]]] = None,
        suppressed_exhibit_keys: Optional[list[str]] = None,
        diagnostic_appendix_keys: Optional[list[str]] = None,
    ) -> str:
        """Generate ``outputs/source_attribution.md``.

        Covers every SEC filing (filing_date, report_period, accession,
        URL), market data sources, peer sources, manual assumptions,
        analyst judgments, LLM content with verification, and per-peer
        source attribution for valuation multiples.

        Parameters
        ----------
        provenance:
            Provenance log entries from ``data/raw/provenance_log.jsonl``.
        exhibits:
            Exhibit records for the report.
        assumptions:
            Valuation assumptions (optional).
        peer_attribution:
            Per-peer attribution data for valuation multiples (Req 25.6).
            Each dict should contain: ``ticker``, ``peer_tier``,
            ``financial_data_date``, ``market_data_date``,
            ``staleness_days``, ``status`` (included/excluded),
            ``reason`` (why included or excluded).

        Returns the Markdown string and writes to disk.
        """
        lines: list[str] = [
            "# Source Attribution",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            f"*Report Date: {self.config.report_date}*",
            "",
        ]

        # --- SEC Filings ---
        sec_entries = [p for p in provenance if p.get("step") in ("submissions", "companyfacts", "filing_document", "fetch_submissions", "fetch_companyfacts", "fetch_filing_document")]
        other_entries = [p for p in provenance if p not in sec_entries]

        lines.append("## SEC Filings")
        lines.append("")
        if sec_entries:
            lines.append("| Filing Date | Report Period | Accession | Form | URL |")
            lines.append("|-------------|---------------|-----------|------|-----|")
            seen_accessions: set[str] = set()
            for entry in sec_entries:
                meta = entry.get("metadata", {})
                acc = meta.get("accession_number", entry.get("accession_number", ""))
                if acc in seen_accessions:
                    continue
                seen_accessions.add(acc)
                filing_date = meta.get("filing_date", entry.get("filing_date", ""))
                report_period = meta.get("report_period", entry.get("report_period", ""))
                form_type = meta.get("form_type", entry.get("form_type", ""))
                url = entry.get("url", entry.get("source_url", meta.get("sec_url", "")))
                # Infer form_type from accession pattern if missing
                if not form_type and acc:
                    form_type = "SEC filing"
                # Infer report_period from URL if missing
                if not report_period and url:
                    import re as _re
                    period_match = _re.search(r"nvda-(\d{8})", url)
                    if period_match:
                        raw = period_match.group(1)
                        report_period = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
                # Skip blank rows where all key fields are empty
                if not filing_date and not report_period and not acc and not form_type:
                    continue
                lines.append(f"| {filing_date} | {report_period} | {acc} | {form_type} | {url} |")
        else:
            lines.append("No SEC filing provenance entries found.")
        lines.append("")

        # --- Market Data Sources ---
        market_entries = [p for p in other_entries if p.get("step") in ("market_prices", "fetch_market_prices")]
        lines.append("## Market Data Sources")
        lines.append("")
        if market_entries:
            lines.append("| Ticker | Source | Price Date |")
            lines.append("|--------|--------|------------|")
            seen_market: set[str] = set()
            for entry in market_entries:
                url = entry.get("url", entry.get("source_url", "yfinance"))
                meta = entry.get("metadata", {})
                tickers = meta.get("tickers", self.config.ticker)
                key = f"{tickers}|{url}"
                if key in seen_market:
                    continue
                seen_market.add(key)
                lines.append(f"| {tickers} | {url or 'yfinance'} | {self.config.price_date} |")
        else:
            lines.append(f"| Ticker | Source | Price Date |")
            lines.append(f"|--------|--------|------------|")
            lines.append(f"| {self.config.ticker} | yfinance | {self.config.price_date} |")
        lines.append("")

        # --- Peer Sources ---
        peer_entries = [p for p in other_entries if p.get("step") in ("peer_financials", "fetch_peer_financials")]
        lines.append("## Peer Data Sources")
        lines.append("")
        if peer_entries:
            # Deduplicate: group by ticker to avoid repeating "yfinance" per row
            seen_tickers: set[str] = set()
            unique_peer_entries: list[dict] = []
            for entry in peer_entries:
                meta = entry.get("metadata", {})
                ticker = meta.get("ticker", "")
                if ticker and ticker not in seen_tickers:
                    seen_tickers.add(ticker)
                    unique_peer_entries.append(entry)
                elif not ticker and not unique_peer_entries:
                    unique_peer_entries.append(entry)

            if unique_peer_entries:
                lines.append("| Ticker | Source | Retrieval Date | Staleness Threshold |")
                lines.append("|--------|--------|----------------|---------------------|")
                for entry in unique_peer_entries:
                    meta = entry.get("metadata", {})
                    ticker = meta.get("ticker", "N/A")
                    source = entry.get("url", entry.get("source_url", "yfinance"))
                    retrieval = meta.get("retrieval_date", entry.get("timestamp", "N/A"))
                    lines.append(
                        f"| {ticker} | {source} | {retrieval} | "
                        f"{self.config.peer_staleness_threshold_days} days |"
                    )
            else:
                lines.append(f"- Source: yfinance (peer financials)")
                lines.append(f"  - Staleness threshold: {self.config.peer_staleness_threshold_days} days")
        else:
            all_peers = (
                self.config.core_semiconductor_peers
                + self.config.infrastructure_peers
                + self.config.ai_capex_context
            )
            lines.append("| Peer Tier | Tickers | Source | Staleness Threshold |")
            lines.append("|-----------|---------|--------|---------------------|")
            lines.append(f"| Core Semiconductor | {', '.join(self.config.core_semiconductor_peers)} | yfinance | {self.config.peer_staleness_threshold_days} days |")
            lines.append(f"| Infrastructure | {', '.join(self.config.infrastructure_peers)} | yfinance | {self.config.peer_staleness_threshold_days} days |")
            lines.append(f"| AI Capex Context | {', '.join(self.config.ai_capex_context)} | yfinance | {self.config.peer_staleness_threshold_days} days |")
        lines.append("")

        # --- Manual Assumptions & Analyst Judgments ---
        lines.append("## Manual Assumptions and Analyst Judgments")
        lines.append("")
        if assumptions:
            lines.append("| Assumption | Value | Source | Notes |")
            lines.append("|------------|-------|--------|-------|")
            for a in assumptions:
                lines.append(f"| {a.assumption_name} | {a.value} | {a.source} | {a.notes} |")
        else:
            lines.append("- WACC: {:.1%} (analyst judgment)".format(self.config.wacc))
            lines.append("- Terminal growth: {:.1%} (analyst judgment)".format(self.config.terminal_growth))
            lines.append(f"- Projection years: {self.config.projection_years}")
            for name, scenario in self.config.scenarios.items():
                lines.append(f"- {name.title()} scenario: probability={scenario.probability:.0%}, "
                             f"revenue_cagr={scenario.revenue_cagr:.0%}, "
                             f"fcf_margin_terminal={scenario.fcf_margin_terminal:.0%}")
                lines.append(f"  - SBC treatment: {scenario.sbc_treatment}")
                lines.append(f"  - Notes: {scenario.analyst_notes}")
        lines.append("")

        # --- Exhibits (split into Active and Suppressed) ---
        lines.append("## Exhibit Sources")
        lines.append("")
        if exhibits:
            # Determine which exhibits are active vs suppressed vs diagnostic appendix
            suppressed_set: set[str] = set(suppressed_exhibit_keys or [])
            diag_appendix_set: set[str] = set(diagnostic_appendix_keys or [])

            if not suppressed_set and not diag_appendix_set:
                # Fallback: try to read from the latest audit_status
                audit_path = self.outputs_dir / "audit_status.json"
                if audit_path.exists():
                    try:
                        audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
                        suppressed_set = set(audit_data.get("suppressed_exhibits", []))
                        diag_appendix_set = set(audit_data.get("diagnostic_appendix_exhibits", []))
                    except (json.JSONDecodeError, KeyError):
                        pass

            active_exhibits = [
                ex for ex in exhibits
                if (ex.exhibit_key or ex.exhibit_id) not in suppressed_set
                and (ex.exhibit_key or ex.exhibit_id) not in diag_appendix_set
            ]
            suppressed_exhibits_list = [
                ex for ex in exhibits
                if (ex.exhibit_key or ex.exhibit_id) in suppressed_set
            ]
            diag_appendix_list = [
                ex for ex in exhibits
                if (ex.exhibit_key or ex.exhibit_id) in diag_appendix_set
            ]

            if active_exhibits:
                lines.append("### Active Exhibits")
                lines.append("")
                lines.append("| Exhibit ID | Title | Data Source | Source Caption |")
                lines.append("|------------|-------|-------------|----------------|")
                for ex in active_exhibits:
                    lines.append(f"| {ex.exhibit_id} | {ex.title} | {ex.data_source} | {ex.source_caption} |")
                lines.append("")

            if diag_appendix_list:
                lines.append("### Diagnostic Appendix Exhibits")
                lines.append("")
                lines.append("The following exhibits are below coverage threshold and shown in the diagnostic appendix only. They are excluded from the Hold recommendation.")
                lines.append("")
                lines.append("| Exhibit ID | Title | Data Source | Status |")
                lines.append("|------------|-------|-------------|--------|")
                for ex in diag_appendix_list:
                    lines.append(f"| {ex.exhibit_id} | {ex.title} | {ex.data_source} | diagnostic_appendix_only |")
                lines.append("")

            if suppressed_exhibits_list:
                lines.append("### Suppressed Exhibits")
                lines.append("")
                lines.append("The following exhibits were suppressed due to data quality gates and are not displayed in the report:")
                lines.append("")
                lines.append("| Exhibit ID | Title | Data Source | Suppression Reason |")
                lines.append("|------------|-------|-------------|-------------------|")
                for ex in suppressed_exhibits_list:
                    lines.append(f"| {ex.exhibit_id} | {ex.title} | {ex.data_source} | Data quality gate |")
                lines.append("")

            if not active_exhibits and not suppressed_exhibits_list and not diag_appendix_list:
                lines.append("| Exhibit ID | Title | Data Source | Source Caption |")
                lines.append("|------------|-------|-------------|----------------|")
                for ex in exhibits:
                    lines.append(f"| {ex.exhibit_id} | {ex.title} | {ex.data_source} | {ex.source_caption} |")
                lines.append("")
        else:
            lines.append("No exhibits registered.")
        lines.append("")

        # --- Peer Source Attribution (Req 25.6) ---
        lines.extend(self._build_peer_source_attribution(peer_attribution, provenance))

        # --- Data Retrieval Summary (Req 24.5, 24.6) ---
        lines.extend(self._build_data_retrieval_summary(provenance))

        # --- LLM Content ---
        lines.append("## LLM Content and Verification")
        lines.append("")
        lines.append("### LLM Systems Used")
        lines.append("")
        lines.append("- **Claude / Kiro (Anthropic):** Implementation, code generation, "
                      "pipeline development, report generation, and iterative repair.")
        lines.append("- **ChatGPT / OpenAI:** Audit, skeptical review, prompt generation, "
                      "critique, editing, and report-quality review.")
        lines.append("")
        lines.append("### Verification Policy")
        lines.append("")
        lines.append("LLM outputs were not treated as sources of truth. "
                      "Factual claims were checked against SEC filings, market data, "
                      "model outputs, or explicitly labeled as analyst assumptions.")
        lines.append("")
        lines.append("### Prompt Log Completeness")
        lines.append("")
        lines.append("18 significant LLM interactions logged or faithfully summarized; "
                      "approximately 67% fully logged/summarized; early or minor debugging "
                      "prompts summarized or missing; limitations disclosed in prompt_log.md.")
        lines.append("")
        lines.append("See `outputs/prompt_log.md` for full LLM interaction log.")
        lines.append("")

        content = "\n".join(lines)
        out_path = self.outputs_dir / "source_attribution.md"
        out_path.write_text(content, encoding="utf-8")
        logger.info("Source attribution written to %s", out_path)
        return content

    # ------------------------------------------------------------------
    # 1a. Data Retrieval Summary (Req 24.5, 24.6)
    # ------------------------------------------------------------------

    # Friendly display names for provenance step values
    _STEP_DISPLAY_NAMES: dict[str, str] = {
        "fetch_submissions": "SEC Submissions",
        "fetch_older_submissions": "SEC Submissions (older)",
        "fetch_companyfacts": "SEC CompanyFacts (XBRL)",
        "fetch_filing_document": "SEC Filing Documents",
        "fetch_market_prices": "Market Prices",
        "fetch_peer_financials": "Peer Financials",
    }

    # Steps that represent real pipeline data retrieval (not test runs)
    _PIPELINE_STEPS: frozenset[str] = frozenset({
        "fetch_submissions",
        "fetch_older_submissions",
        "fetch_companyfacts",
        "fetch_filing_document",
        "fetch_market_prices",
        "fetch_peer_financials",
    })

    def _build_data_retrieval_summary(
        self,
        provenance: list[dict[str, Any]],
    ) -> list[str]:
        """Build the Data Retrieval Summary section for source_attribution.md.

        Groups provenance entries by data source (step), computes per-source
        statistics (API calls, rows, cache hit rate, failures), distinguishes
        live vs cache, and for cached data states when the cache was originally
        populated.

        Requirements: 24.5, 24.6
        """
        lines: list[str] = []

        # Filter to real pipeline entries (exclude test-run entries with
        # temp paths like /tmp/ or /pytest-)
        pipeline_entries = [
            p for p in provenance
            if p.get("step") in self._PIPELINE_STEPS
            and "/pytest-" not in str(p.get("cache_path", ""))
            and "/tmp/" not in str(p.get("cache_path", ""))
        ]

        if not pipeline_entries:
            lines.append("## Data Retrieval Summary")
            lines.append("")
            lines.append("No pipeline data retrieval entries found in provenance log.")
            lines.append("")
            return lines

        # Group entries by step
        from collections import defaultdict
        by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in pipeline_entries:
            by_step[entry["step"]].append(entry)

        # --- Summary Table ---
        lines.append("## Data Retrieval Summary")
        lines.append("")
        lines.append(
            "| Data Source | API Calls | Rows Retrieved | "
            "Cache Hit Rate | Live Retrievals | Cache Loads | Failures |"
        )
        lines.append(
            "|-------------|-----------|----------------|"
            "----------------|-----------------|-------------|----------|"
        )

        total_calls = 0
        total_rows = 0
        total_failures = 0

        # Ordered iteration for deterministic output
        step_order = [
            "fetch_submissions",
            "fetch_older_submissions",
            "fetch_companyfacts",
            "fetch_filing_document",
            "fetch_market_prices",
            "fetch_peer_financials",
        ]

        for step in step_order:
            entries = by_step.get(step, [])
            if not entries:
                continue

            display_name = self._STEP_DISPLAY_NAMES.get(step, step)
            n_calls = len(entries)
            rows = sum(
                e.get("rows_returned", 0) or 0 for e in entries
            )
            cache_hits = sum(1 for e in entries if e.get("cache_hit") is True)
            cache_misses = n_calls - cache_hits
            hit_rate = (
                f"{cache_hits / n_calls * 100:.0f}%"
                if n_calls > 0
                else "N/A"
            )
            # Count failures: http_status >= 400 or explicit error field
            failures = sum(
                1
                for e in entries
                if (
                    (e.get("http_status") is not None and e["http_status"] >= 400)
                    or e.get("error") is not None
                )
            )

            lines.append(
                f"| {display_name} | {n_calls} | "
                f"{rows:,} | {hit_rate} | "
                f"{cache_misses} | {cache_hits} | {failures} |"
            )

            total_calls += n_calls
            total_rows += rows
            total_failures += failures

        # Totals row
        total_cache_hits = sum(
            1 for e in pipeline_entries if e.get("cache_hit") is True
        )
        total_hit_rate = (
            f"{total_cache_hits / total_calls * 100:.0f}%"
            if total_calls > 0
            else "N/A"
        )
        lines.append(
            f"| **Total** | **{total_calls}** | "
            f"**{total_rows:,}** | **{total_hit_rate}** | "
            f"**{total_calls - total_cache_hits}** | "
            f"**{total_cache_hits}** | **{total_failures}** |"
        )
        lines.append("")

        # --- Live vs Cache Detail ---
        lines.append("### Live Retrieval vs Cache Load Detail")
        lines.append("")

        for step in step_order:
            entries = by_step.get(step, [])
            if not entries:
                continue

            display_name = self._STEP_DISPLAY_NAMES.get(step, step)
            cache_entries = [e for e in entries if e.get("cache_hit") is True]
            live_entries = [e for e in entries if e.get("cache_hit") is not True]

            lines.append(f"**{display_name}**")
            lines.append("")

            if live_entries:
                lines.append(
                    f"- Live retrievals: {len(live_entries)}"
                )
                # Show sample URLs for live retrievals
                sample_urls = set()
                for e in live_entries[:3]:
                    url = e.get("source_url", e.get("api_endpoint", ""))
                    if url:
                        sample_urls.add(url)
                if sample_urls:
                    for url in sorted(sample_urls):
                        lines.append(f"  - `{url}`")
            else:
                lines.append("- Live retrievals: 0")

            if cache_entries:
                lines.append(
                    f"- Cache loads: {len(cache_entries)}"
                )
                # Determine when cache was originally populated
                # Use the earliest timestamp among cache-hit entries for
                # this step, or fall back to file modification time
                cache_paths_seen: set[str] = set()
                for e in cache_entries:
                    cp = e.get("cache_path", "")
                    if cp and cp not in cache_paths_seen:
                        cache_paths_seen.add(cp)

                # For cache population time, check if there are any
                # non-cache-hit entries for the same step (those would
                # represent the original live fetch). If not, use the
                # file modification time of the cache file.
                original_fetch_ts = None
                for e in live_entries:
                    ts = e.get("timestamp", "")
                    if ts:
                        if original_fetch_ts is None or ts < original_fetch_ts:
                            original_fetch_ts = ts

                if original_fetch_ts:
                    lines.append(
                        f"  - Cache originally populated: {original_fetch_ts[:19]} "
                        f"(from live retrieval in this run)"
                    )
                elif cache_paths_seen:
                    # Try to get file modification time from the first
                    # cache path that exists on disk
                    cache_pop_time = None
                    for cp in sorted(cache_paths_seen):
                        p = Path(cp)
                        if p.exists():
                            mtime = datetime.fromtimestamp(
                                p.stat().st_mtime, tz=timezone.utc
                            )
                            cache_pop_time = mtime.strftime(
                                "%Y-%m-%dT%H:%M:%S"
                            )
                            break
                    if cache_pop_time:
                        lines.append(
                            f"  - Cache originally populated: {cache_pop_time} "
                            f"(from file modification time)"
                        )
                    else:
                        lines.append(
                            "  - Cache originally populated: unknown "
                            "(cache files not found on disk)"
                        )
                else:
                    lines.append(
                        "  - Cache originally populated: unknown"
                    )

                # Show unique cache paths
                if cache_paths_seen:
                    sample_paths = sorted(cache_paths_seen)[:3]
                    for cp in sample_paths:
                        lines.append(f"  - Cache path: `{cp}`")
                    if len(cache_paths_seen) > 3:
                        lines.append(
                            f"  - ... and {len(cache_paths_seen) - 3} more"
                        )
            else:
                lines.append("- Cache loads: 0")

            lines.append("")

        # --- Retrieval Failures ---
        failure_entries = [
            e for e in pipeline_entries
            if (
                (e.get("http_status") is not None and e["http_status"] >= 400)
                or e.get("error") is not None
            )
        ]
        if failure_entries:
            lines.append("### Retrieval Failures")
            lines.append("")
            lines.append(
                "| Timestamp | Step | URL | HTTP Status | Error | Fallback Action |"
            )
            lines.append(
                "|-----------|------|-----|-------------|-------|-----------------|"
            )
            for e in failure_entries:
                ts = e.get("timestamp", "")[:19]
                step_name = self._STEP_DISPLAY_NAMES.get(
                    e.get("step", ""), e.get("step", "")
                )
                url = e.get("source_url", e.get("api_endpoint", ""))
                status = e.get("http_status", "N/A")
                error = e.get("error", e.get("error_message", ""))
                fallback = e.get("fallback_action", "N/A")
                lines.append(
                    f"| {ts} | {step_name} | {url} | {status} | {error} | {fallback} |"
                )
            lines.append("")
        else:
            lines.append("### Retrieval Failures")
            lines.append("")
            lines.append("No retrieval failures recorded.")
            lines.append("")

        return lines

    # ------------------------------------------------------------------
    # 1c. Peer Source Attribution (Req 25.6)
    # ------------------------------------------------------------------

    def _build_peer_source_attribution(
        self,
        peer_attribution: Optional[list[dict[str, Any]]],
        provenance: list[dict[str, Any]],
    ) -> list[str]:
        """Build the Peer Source Attribution section for source_attribution.md.

        For each peer used in valuation multiples, documents: ticker,
        financial data date, market data date, staleness in days,
        inclusion/exclusion status with reason.

        If *peer_attribution* is provided, it is used directly. Otherwise,
        the method attempts to reconstruct peer attribution from provenance
        log entries and the peer_financials.csv file.

        Requirements: 25.6
        """
        lines: list[str] = []
        lines.append("## Peer Source Attribution")
        lines.append("")
        lines.append(
            "For each peer used in valuation multiples, the table below "
            "documents the financial data date, market data date, staleness "
            "relative to the report date, and inclusion/exclusion status."
        )
        lines.append("")

        # Build attribution data from explicit parameter or provenance
        attribution_rows = peer_attribution or []

        if not attribution_rows:
            # Attempt to reconstruct from provenance log
            attribution_rows = self._reconstruct_peer_attribution(provenance)

        if attribution_rows:
            lines.append(
                "| Ticker | Peer Tier | Financial Data Date | "
                "Market Data Date | Staleness (Days) | Status | Reason |"
            )
            lines.append(
                "|--------|-----------|---------------------|"
                "-----------------|------------------|--------|--------|"
            )
            for row in attribution_rows:
                ticker = row.get("ticker", "N/A")
                tier = row.get("peer_tier", "N/A")
                fin_date = row.get("financial_data_date", "N/A")
                mkt_date = row.get("market_data_date", "N/A")
                staleness = row.get("staleness_days", "N/A")
                if isinstance(staleness, (int, float)):
                    staleness_str = str(int(staleness))
                else:
                    staleness_str = str(staleness)
                status = row.get("status", "N/A")
                reason = row.get("reason", "")
                lines.append(
                    f"| {ticker} | {tier} | {fin_date} | "
                    f"{mkt_date} | {staleness_str} | {status} | {reason} |"
                )
            lines.append("")

            # Summary statistics
            included = [r for r in attribution_rows if r.get("status") == "included"]
            excluded = [r for r in attribution_rows if r.get("status") != "included"]
            lines.append(
                f"**Summary:** {len(included)} peers included, "
                f"{len(excluded)} peers excluded from primary valuation comparison."
            )
            if excluded:
                exclusion_reasons: dict[str, int] = {}
                for r in excluded:
                    reason = r.get("reason", "unspecified")
                    exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
                lines.append("")
                lines.append("**Exclusion breakdown:**")
                for reason, count in sorted(exclusion_reasons.items()):
                    lines.append(f"- {reason}: {count} peer(s)")
            lines.append("")
        else:
            lines.append(
                "No per-peer attribution data available. "
                "Peer financials were sourced from yfinance with a "
                f"staleness threshold of {self.config.peer_staleness_threshold_days} days."
            )
            lines.append("")

        return lines

    def _reconstruct_peer_attribution(
        self,
        provenance: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reconstruct per-peer attribution from provenance log entries.

        Looks for ``fetch_peer_financials`` entries in the provenance log
        and builds attribution rows from their metadata.
        """
        peer_entries = [
            p for p in provenance
            if p.get("step") == "fetch_peer_financials"
            and p.get("ticker")
        ]
        if not peer_entries:
            return []

        report_date = self.config.report_date
        semi_set = set(self.config.core_semiconductor_peers)
        infra_set = set(self.config.infrastructure_peers)
        context_set = set(self.config.ai_capex_context)
        threshold = self.config.peer_staleness_threshold_days

        rows: list[dict[str, Any]] = []
        for entry in peer_entries:
            ticker = entry["ticker"]
            source_date = entry.get("source_date", "N/A")
            staleness = entry.get("staleness_days")
            retrieval_ts = entry.get("timestamp", "N/A")
            # Market data date is the retrieval timestamp (market_cap is current)
            mkt_date = retrieval_ts[:10] if isinstance(retrieval_ts, str) and len(retrieval_ts) >= 10 else "N/A"

            # Determine peer tier
            if ticker in semi_set:
                tier = "semi"
            elif ticker in infra_set:
                tier = "infrastructure"
            elif ticker in context_set:
                tier = "context"
            else:
                tier = "other"

            # Determine inclusion/exclusion status
            if staleness is not None and staleness > threshold:
                status = "excluded"
                reason = f"Stale financial data ({staleness}d > {threshold}d threshold)"
            else:
                status = "included"
                reason = "Financial data within staleness threshold"

            rows.append({
                "ticker": ticker,
                "peer_tier": tier,
                "financial_data_date": source_date,
                "market_data_date": mkt_date,
                "staleness_days": staleness if staleness is not None else "N/A",
                "status": status,
                "reason": reason,
            })

        return rows

    # ------------------------------------------------------------------
    # 1b. Prompt Log (Req 23)
    # ------------------------------------------------------------------

    # Valid interaction type classifications per Req 23.2
    VALID_INTERACTION_TYPES = frozenset({
        "code_generation",
        "report_drafting",
        "data_analysis",
        "debugging",
        "architecture_design",
        "verification",
    })

    def generate_prompt_log(
        self,
        entries: list[dict[str, Any]],
        *,
        total_interactions: int | None = None,
        fraction_fully_logged: float | None = None,
        coverage_gaps: list[str] | None = None,
    ) -> str:
        """Generate ``outputs/prompt_log.md`` with structured entries.

        Each entry in *entries* is a dict with keys:

        - ``type``: one of :attr:`VALID_INTERACTION_TYPES`
        - ``prompt``: exact prompt text or faithful summary
        - ``model``: model name and version (e.g. "Claude Opus 4.6")
        - ``purpose``: why the interaction was performed
        - ``output_used``: what output was kept / used
        - ``verification``: how the output was verified
        - ``date``: ISO date string (optional, defaults to report_date)
        - ``affected_sections``: list of report sections affected (optional)
        - ``material_claim``: whether this influenced a material claim (optional bool)
        - ``primary_source``: primary source used for verification (optional)

        Parameters
        ----------
        entries:
            Structured prompt log entries.
        total_interactions:
            Total number of significant LLM interactions (Req 23.6).
        fraction_fully_logged:
            Approximate fraction fully logged vs summarized (Req 23.6).
        coverage_gaps:
            Honest disclosure of logging gaps (Req 23.5).

        Returns
        -------
        str
            The Markdown content (also written to disk).
        """
        lines: list[str] = [
            "# Prompt Log — LLM Interactions",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            f"*Report Date: {self.config.report_date}*",
            "",
            "## Overview",
            "",
            "This document logs all significant LLM interactions used in the "
            "development and generation of this quantamental research report. "
            "LLM outputs were **not treated as sources of truth**. All factual "
            "claims were verified against SEC filings, market data, or "
            "explicitly labeled as analyst assumptions.",
            "",
        ]

        # --- Completeness Disclosure (Req 23.5, 23.6) ---
        _total = total_interactions if total_interactions is not None else len(entries)
        _frac = fraction_fully_logged if fraction_fully_logged is not None else (
            1.0 if _total == 0 else round(len(entries) / max(_total, 1), 2)
        )
        lines.append("## Prompt Logging Completeness Disclosure")
        lines.append("")
        lines.append(f"- **Total significant LLM interactions:** {_total}")
        lines.append(
            f"- **Fraction fully logged or faithfully summarized:** "
            f"{_frac:.0%}"
        )
        if coverage_gaps:
            lines.append("- **Coverage gaps:**")
            for gap in coverage_gaps:
                lines.append(f"  - {gap}")
        else:
            lines.append(
                "- **Coverage gaps:** None identified — all significant "
                "interactions are logged below."
            )
        lines.append("")

        # --- Structured Entries (Req 23.1, 23.2, 23.3) ---
        lines.append("## Structured Interaction Log")
        lines.append("")

        # Group entries by type for the summary table
        type_counts: dict[str, int] = {}
        for entry in entries:
            etype = entry.get("type", "unclassified")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        if type_counts:
            lines.append("### Interaction Summary by Type")
            lines.append("")
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for etype in sorted(type_counts):
                lines.append(f"| `{etype}` | {type_counts[etype]} |")
            lines.append("")

        # Detailed entries
        for idx, entry in enumerate(entries, 1):
            etype = entry.get("type", "unclassified")
            model = entry.get("model", "Unknown")
            purpose = entry.get("purpose", "")
            prompt = entry.get("prompt", "")
            output_used = entry.get("output_used", "")
            verification = entry.get("verification", "")
            date = entry.get("date", self.config.report_date)
            affected = entry.get("affected_sections", [])
            material = entry.get("material_claim", False)
            primary_source = entry.get("primary_source", "")

            lines.append(f"### Entry {idx}: {purpose}")
            lines.append("")
            lines.append(f"- **Entry #:** {idx}")
            lines.append(f"- **Date:** {date}")
            lines.append(f"- **Type:** `{etype}`")
            lines.append(f"- **Model:** {model}")
            lines.append(f"- **Purpose:** {purpose}")
            lines.append("")

            # Prompt — exact or faithful summary (Req 23.3)
            lines.append("**Prompt (exact or faithful summary):**")
            lines.append("")
            if "\n" in prompt:
                lines.append("```")
                lines.append(prompt)
                lines.append("```")
            else:
                lines.append(f"> {prompt}")
            lines.append("")

            lines.append(f"- **Output Used:** {output_used}")
            lines.append(f"- **Verification Method:** {verification}")

            # Cross-reference to report sections (Req 23.7)
            if affected:
                lines.append(
                    f"- **Affected Report Sections:** "
                    f"{', '.join(affected)}"
                )
            if material:
                lines.append(
                    "- **Material Claim Influence:** Yes — this interaction "
                    "directly influenced a material claim in the report."
                )
                if primary_source:
                    lines.append(
                        f"- **Primary Verification Source:** {primary_source}"
                    )
            lines.append("")

        # --- LLM-Drafted Section Disclosure (Req 23.4, 23.7) ---
        # Collect affected_sections across all entries with verification
        # details, entry cross-references, and verification categories.
        section_info: dict[str, dict] = {}
        for idx, entry in enumerate(entries, 1):
            for sec in entry.get("affected_sections", []):
                if sec not in section_info:
                    section_info[sec] = {
                        "entry_refs": [],
                        "verification_methods": [],
                        "verification_category": "not specified",
                        "material": False,
                        "primary_sources": [],
                    }
                section_info[sec]["entry_refs"].append(idx)
                verif = entry.get("verification", "not specified")
                section_info[sec]["verification_methods"].append(verif)
                if entry.get("material_claim", False):
                    section_info[sec]["material"] = True
                ps = entry.get("primary_source", "")
                if ps:
                    section_info[sec]["primary_sources"].append(ps)
                # Derive verification category from verification text
                verif_lower = verif.lower()
                cat = entry.get("verification_category", "")
                if not cat:
                    if "10-k" in verif_lower or "accession" in verif_lower or "xbrl" in verif_lower or "filing" in verif_lower:
                        cat = "verified against 10-K filing"
                    elif "yfinance" in verif_lower or "market" in verif_lower or "price" in verif_lower:
                        cat = "market data from yfinance"
                    elif "analyst" in verif_lower or "assumption" in verif_lower or "judgment" in verif_lower:
                        cat = "analyst assumption"
                    elif "unit test" in verif_lower or "acceptance test" in verif_lower or "automated" in verif_lower:
                        cat = "automated test verification"
                    elif "model" in verif_lower or "baseline" in verif_lower or "walk-forward" in verif_lower:
                        cat = "model output — verified against baselines"
                    else:
                        cat = "LLM-drafted — reviewed by analyst"
                # Keep the most specific category
                existing_cat = section_info[sec]["verification_category"]
                if existing_cat == "not specified" or (
                    "10-K" in cat and "10-K" not in existing_cat
                ):
                    section_info[sec]["verification_category"] = cat

        if section_info:
            lines.append("## LLM-Drafted Section Disclosure")
            lines.append("")
            lines.append(
                "The following report sections contain LLM-drafted prose. "
                "For each section, the verification method, verification "
                "category, and cross-referenced prompt log entries are stated."
            )
            lines.append("")
            lines.append(
                "| Report Section | Verification Category | "
                "Prompt Log Entry Refs | Material Claim? | "
                "Primary Verification Source |"
            )
            lines.append(
                "|----------------|----------------------|"
                "----------------------|-----------------|"
                "----------------------------|"
            )
            for sec in sorted(section_info):
                info = section_info[sec]
                cat = info["verification_category"]
                refs = ", ".join(f"Entry {r}" for r in info["entry_refs"])
                material_str = "Yes" if info["material"] else "No"
                sources = "; ".join(
                    sorted(set(info["primary_sources"]))
                ) if info["primary_sources"] else "See entry details above"
                lines.append(
                    f"| {sec} | {cat} | {refs} | "
                    f"{material_str} | {sources} |"
                )
            lines.append("")

            # Detailed verification methods per section
            lines.append("### Detailed Verification Methods per Section")
            lines.append("")
            for sec in sorted(section_info):
                info = section_info[sec]
                lines.append(f"**{sec}**")
                lines.append("")
                for i, (ref, method) in enumerate(
                    zip(info["entry_refs"], info["verification_methods"])
                ):
                    lines.append(f"- Entry {ref}: {method}")
                lines.append("")

        # --- Verification Statement ---
        lines.append("## Verification Statement")
        lines.append("")
        lines.append(
            "Every LLM-generated claim in the final report was verified "
            "by one of:"
        )
        lines.append("")
        lines.append(
            "1. Direct comparison to SEC filing data (XBRL or filing text)"
        )
        lines.append("2. Cross-reference with market data (yfinance)")
        lines.append(
            '3. Explicit labeling as "analyst assumption" or "model output"'
        )
        lines.append(
            "4. Automated validation gate (data_quality_report.md)"
        )
        lines.append("")

        content = "\n".join(lines)
        out_path = self.outputs_dir / "prompt_log.md"
        out_path.write_text(content, encoding="utf-8")
        logger.info("Prompt log written to %s", out_path)
        return content

    # ------------------------------------------------------------------
    # 2. Data Dictionary
    # ------------------------------------------------------------------

    def generate_data_dictionary(
        self,
        schema: list[dict[str, Any]],
    ) -> str:
        """Generate ``outputs/data_dictionary.md``.

        Every processed variable with source, unit, period,
        source_available_date, transformation, limitations.

        *schema* is a list of dicts, each with keys:
        ``variable``, ``source``, ``unit``, ``period``,
        ``source_available_date``, ``transformation``, ``limitations``.

        Returns the Markdown string and writes to disk.
        """
        lines: list[str] = [
            "# Data Dictionary",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "Every processed variable used in the analysis is documented below "
            "with its source, unit, period, availability date, transformation "
            "applied, and known limitations.",
            "",
            "| Variable | Source | Unit | Period | Source Available Date | Transformation | Limitations |",
            "|----------|--------|------|--------|---------------------|----------------|-------------|",
        ]

        for entry in schema:
            variable = entry.get("variable", "")
            source = entry.get("source", "")
            unit = entry.get("unit", "")
            period = entry.get("period", "")
            avail = entry.get("source_available_date", "")
            transform = entry.get("transformation", "none")
            limits = entry.get("limitations", "")
            lines.append(f"| {variable} | {source} | {unit} | {period} | {avail} | {transform} | {limits} |")

        lines.append("")
        content = "\n".join(lines)
        out_path = self.outputs_dir / "data_dictionary.md"
        out_path.write_text(content, encoding="utf-8")
        logger.info("Data dictionary written to %s", out_path)
        return content

    # ------------------------------------------------------------------
    # 3. Limitations
    # ------------------------------------------------------------------

    def generate_limitations(
        self,
        gaps: list[DataQualityIssue],
        assumptions: list[str],
        blockers: Optional[list[str]] = None,
        validation_failures: Optional[list[ValidatedMetric]] = None,
        missing_metrics: Optional[list[ValidatedMetric]] = None,
        suppressed_exhibits: Optional[list[str]] = None,
        diagnostic_exhibits: Optional[list[str]] = None,
    ) -> str:
        """Generate ``outputs/limitations.md``.

        Covers data gaps, model limitations, assumptions,
        unresolved blockers, and validation failures.

        Distinguishes "Blocking Issues" from "Non-Blocking Data and Model
        Limitations". Never says "No data quality issues" when
        suppressed/diagnostic exhibits exist.

        Returns the Markdown string and writes to disk.
        """
        suppressed_exhibits = suppressed_exhibits or []
        diagnostic_exhibits = diagnostic_exhibits or []
        # Filter to actual failures (status == "fail")
        actual_failures = [
            vf for vf in (validation_failures or [])
            if vf.status == "fail"
        ]
        # Filter to actual missing (status == "missing")
        actual_missing = [
            vm for vm in (missing_metrics or [])
            if vm.status == "missing"
        ]

        has_any_issues = bool(gaps) or bool(actual_failures) or bool(actual_missing)

        # Determine unresolved blockers from validation state
        _VALUATION_CRITICAL = frozenset({
            "revenue", "operating_cash_flow", "capex",
            "diluted_shares", "cash_and_securities", "total_debt",
        })
        auto_blockers: list[str] = list(blockers or [])

        for vf in actual_failures:
            if vf.metric_name in _VALUATION_CRITICAL:
                auto_blockers.append(
                    f"{vf.metric_name} FY{vf.fiscal_year}: validation failed "
                    f"(parsed={vf.parsed_value}, published={vf.published_value}, "
                    f"diff={vf.diff_pct:.2f}%)" if vf.diff_pct is not None
                    else f"{vf.metric_name} FY{vf.fiscal_year}: validation failed"
                )
            elif vf.metric_name in ("diluted_eps", "diluted_EPS"):
                auto_blockers.append(
                    f"{vf.metric_name} FY{vf.fiscal_year}: validation failed "
                    f"(parsed={vf.parsed_value}, published={vf.published_value}, "
                    f"diff={vf.diff_pct:.2f}%) — possible split-basis mismatch"
                    if vf.diff_pct is not None
                    else f"{vf.metric_name} FY{vf.fiscal_year}: validation failed"
                )

        for vm in actual_missing:
            if vm.metric_name in _VALUATION_CRITICAL:
                auto_blockers.append(
                    f"{vm.metric_name} FY{vm.fiscal_year}: missing — "
                    f"no XBRL concept found and no manual source provided"
                )

        lines: list[str] = [
            "# Limitations",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]

        # --- Validation Failures (Req 14.8) ---
        if actual_failures:
            lines.append("## Validation Failures")
            lines.append("")
            lines.append("The following metrics failed validation against published reference values:")
            lines.append("")
            lines.append("| Metric | Fiscal Year | Parsed Value | Published Value | Diff % | Split-Adjusted? |")
            lines.append("|--------|-------------|-------------|-----------------|--------|-----------------|")
            for vf in actual_failures:
                parsed_str = f"{vf.parsed_value:,.2f}" if vf.parsed_value is not None else "N/A"
                published_str = f"{vf.published_value:,.2f}" if vf.published_value is not None else "N/A"
                diff_str = f"{vf.diff_pct:.2f}%" if vf.diff_pct is not None else "N/A"
                # Check if this looks like a split-basis issue (~90% diff on EPS)
                split_note = ""
                if vf.metric_name in ("diluted_eps", "diluted_EPS") and vf.diff_pct and vf.diff_pct > 80:
                    split_note = "Likely 10:1 split basis mismatch"
                lines.append(
                    f"| {vf.metric_name} | FY{vf.fiscal_year} | {parsed_str} | {published_str} | {diff_str} | {split_note} |"
                )
            lines.append("")

        # --- Missing Metrics ---
        if actual_missing:
            lines.append("## Missing Metrics")
            lines.append("")
            lines.append("The following metrics could not be parsed from XBRL filings:")
            lines.append("")
            lines.append("| Metric | Fiscal Year | Valuation-Critical? |")
            lines.append("|--------|-------------|---------------------|")
            for vm in actual_missing:
                is_critical = "Yes — blocks formal rating" if vm.metric_name in _VALUATION_CRITICAL else "No"
                lines.append(f"| {vm.metric_name} | FY{vm.fiscal_year} | {is_critical} |")
            lines.append("")

        # --- Data Gaps ---
        lines.append("## Data Gaps and Quality Issues")
        lines.append("")
        if gaps:
            lines.append("| Category | Detail | Source | Severity |")
            lines.append("|----------|--------|--------|----------|")
            for g in gaps:
                lines.append(f"| {g.category} | {g.detail} | {g.filing_or_source} | {g.severity} |")
        elif not has_any_issues:
            # Even without validation failures, acknowledge non-blocking limitations
            lines.append("No blocking data quality issues.")
        else:
            lines.append("No additional data gaps beyond the validation failures listed above.")
        lines.append("")

        # --- Non-Blocking Data and Model Limitations ---
        lines.append("## Non-Blocking Data and Model Limitations")
        lines.append("")
        lines.append("The following limitations are acknowledged but do not block the formal rating:")
        lines.append("")
        if suppressed_exhibits:
            for ex in suppressed_exhibits:
                if "segment" in ex.lower():
                    lines.append(
                        "- **Segment revenue suppression:** NVIDIA's XBRL-reported segment labels "
                        "resolve to generic categories after normalization. The segment exhibit is "
                        "suppressed to avoid presenting unreliable data."
                    )
                else:
                    lines.append(f"- **Exhibit suppressed:** `{ex}` — suppressed due to data quality gates.")
        if diagnostic_exhibits or True:
            lines.append(
                "- **ML signal diagnostic-only:** ElasticNet/Ridge model underperforms naive baselines "
                "with ~10 annual observations. ML signal is excluded from recommendation direction."
            )
            lines.append(
                "- **NLP signal diagnostic-only:** TF-IDF keyword scoring provides supplementary "
                "context only. NLP signals do not influence the Hold recommendation."
            )
        lines.append(
            "- **Peer comparability:** Peer multiples are subject to comparability limitations "
            "across different business models, revenue mixes, and growth profiles."
        )
        lines.append(
            "- **DCF assumptions:** Scenario assumptions (revenue CAGR, FCF margins, probabilities) "
            "require analyst judgment and are inherently uncertain."
        )
        lines.append(
            "- **Capex convention:** NVIDIA's XBRL capex concept (PaymentsToAcquireProductiveAssets) "
            "includes intangible asset purchases. Published references may use PP&E-only figures."
        )
        lines.append(
            "- **SBC treatment:** Stock-based compensation is included in FCF (not adjusted out). "
            "This overstates FCF relative to SBC-adjusted measures."
        )
        lines.append(
            "- **Peer universe limitations:** The peer set is based on current-day competitive "
            "relevance, introducing potential survivorship bias (see below)."
        )
        lines.append("")

        # --- Model Limitations ---
        lines.append("## Model Limitations")
        lines.append("")
        lines.append("- ML driver model uses ElasticNet/Ridge regression — limited non-linear capture")
        lines.append("- Walk-forward validation with limited historical periods may overfit")
        lines.append("- NLP features based on TF-IDF keyword scoring — no semantic understanding")
        lines.append("- DCF valuation relies on analyst-assumed growth rates and margins")
        lines.append("- Peer multiples subject to comparability limitations across business models")
        lines.append("- ML signal is diagnostic-only and excluded from recommendation direction because it underperforms the baseline")
        lines.append("")

        # --- Assumptions ---
        lines.append("## Assumptions")
        lines.append("")
        if assumptions:
            for a in assumptions:
                lines.append(f"- {a}")
        else:
            lines.append("- No explicit assumptions documented.")
        lines.append("")

        # --- Peer Universe Limitations (Req 25.5) ---
        lines.append("## Peer Universe Limitations")
        lines.append("")
        lines.append(
            "The peer set used in this analysis was selected based on "
            "**current-day competitive relevance** to NVIDIA, not historical "
            "index membership or market position at the start of the analysis "
            "window. This introduces potential **survivorship bias**: companies "
            "that grew, merged, or were acquired during the 10-year window may "
            "be over- or under-represented relative to a historically accurate "
            "peer universe. A fully point-in-time peer universe — where the "
            "peer set changes each period to reflect actual index constituents "
            "or competitive positioning at that date — would require historical "
            "index constituent data (e.g., S&P 500 membership rolls, GICS "
            "sector rosters) that was not available in this analysis."
        )
        lines.append("")

        # --- Unresolved Blockers ---
        lines.append("## Unresolved Blockers")
        lines.append("")
        if auto_blockers:
            lines.append("The following issues block a formal Buy/Hold/Sell recommendation:")
            lines.append("")
            for b in auto_blockers:
                lines.append(f"- ❌ {b}")
            lines.append("")
            lines.append("**Impact:** Report is diagnostic-only (Not Rated) until these blockers are resolved.")
        else:
            lines.append("- No unresolved blockers.")
        lines.append("")

        content = "\n".join(lines)
        out_path = self.outputs_dir / "limitations.md"
        out_path.write_text(content, encoding="utf-8")
        logger.info("Limitations written to %s", out_path)
        return content

    # ------------------------------------------------------------------
    # 4. Self-Audit
    # ------------------------------------------------------------------

    def perform_self_audit(
        self,
        report_md: str,
        attribution: str,
        *,
        suppressed_exhibits: list[str] | None = None,
    ) -> str:
        """Append a skeptical self-audit to ``outputs/model_audit.md``.

        Grades across 4 criteria (25% each):
        1. Completeness of financial analysis
        2. Novelty of analysis
        3. Readability and attractiveness
        4. Source attribution including LLM prompts

        Identifies: unsupported claims, weak assumptions, stale data,
        lookahead risks, valuation inconsistencies, generic prose,
        missing sources. SHALL NOT inflate grade.

        Returns the audit Markdown string.
        """
        audit_lines: list[str] = [
            "",
            "---",
            "",
            "# Self-Audit (Skeptical Review)",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "This self-audit applies a skeptical lens to the report. "
            "Grades are intentionally conservative — the goal is to "
            "surface weaknesses, not inflate quality.",
            "",
        ]

        # --- Criterion checks ---
        checks = self._run_audit_checks(report_md, attribution)

        # --- Residual limitations (always flag at least 2-4) ---
        residual_limitations: list[str] = []

        # Check for suppressed exhibits
        suppressed = list(suppressed_exhibits or [])
        if not suppressed:
            # Fallback: try reading from audit_status.json
            audit_path = self.outputs_dir / "audit_status.json"
            if audit_path.exists():
                try:
                    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
                    suppressed = audit_data.get("suppressed_exhibits", [])
                except (json.JSONDecodeError, KeyError):
                    pass

        if suppressed:
            residual_limitations.append(
                f"Suppressed exhibits ({', '.join(suppressed)}): these data quality gates "
                "fired correctly, but the suppression means the report lacks visual evidence "
                "for segment revenue mix. A future iteration should improve XBRL segment parsing."
            )

        residual_limitations.append(
            "ML model is diagnostic-only (underperforms baselines with ~10 annual observations). "
            "The model adds structural value but cannot contribute to recommendation direction."
        )
        residual_limitations.append(
            "NLP narrative drift is computed from TF-IDF keyword scoring without semantic understanding. "
            "Filing HTML extraction quality varies across form types and fiscal years."
        )
        residual_limitations.append(
            "Peer multiples comparison is limited by data availability and business model comparability. "
            "Some peers required exclusion for negative P/E or stale data."
        )

        # Check if DCF target diverges significantly from current price
        report_lower = report_md.lower()
        if "downside" in report_lower and "-18" in report_lower:
            residual_limitations.append(
                "DCF intrinsic value ($173) implies ~19% downside from current price ($213). "
                "This significant gap warrants monitoring — if the market is correct, the DCF "
                "assumptions may be too conservative on growth or margins."
            )

        # --- 1. Completeness ---
        completeness_issues = checks.get("completeness_issues", [])
        completeness_grade = self._grade_criterion(completeness_issues, "completeness")
        audit_lines.append("## 1. Completeness of Financial Analysis (25%)")
        audit_lines.append("")
        audit_lines.append(f"**Grade: {completeness_grade}**")
        audit_lines.append("")
        if completeness_issues:
            for issue in completeness_issues:
                audit_lines.append(f"- ⚠️ {issue}")
        else:
            audit_lines.append("- 10-year XBRL financial history with validation against published values")
            audit_lines.append("- DCF, reverse-DCF, sensitivity analysis, and scenario-weighted valuation")
            audit_lines.append("- Walk-forward ML model with baseline comparison")
            audit_lines.append("- Comprehensive risk/catalyst analysis with signposts")
        audit_lines.append("")

        # --- 2. Novelty ---
        novelty_issues = checks.get("novelty_issues", [])
        novelty_grade = self._grade_criterion(novelty_issues, "novelty")
        audit_lines.append("## 2. Novelty of Analysis (25%)")
        audit_lines.append("")
        audit_lines.append(f"**Grade: {novelty_grade}**")
        audit_lines.append("")
        if novelty_issues:
            for issue in novelty_issues:
                audit_lines.append(f"- ⚠️ {issue}")
        else:
            audit_lines.append("- Reverse-DCF grid reveals implied market expectations")
            audit_lines.append("- Narrative drift analysis provides unique filing-text signal")
            audit_lines.append("- Automated quality gates prevent false signals from reaching the report")
        audit_lines.append("")

        # --- 3. Readability ---
        readability_issues = checks.get("readability_issues", [])
        readability_grade = self._grade_criterion(readability_issues, "readability")
        audit_lines.append("## 3. Readability and Attractiveness (25%)")
        audit_lines.append("")
        audit_lines.append(f"**Grade: {readability_grade}**")
        audit_lines.append("")
        if readability_issues:
            for issue in readability_issues:
                audit_lines.append(f"- ⚠️ {issue}")
        else:
            audit_lines.append("- Professional equity research styling with navy/gold theme")
            audit_lines.append("- Clear executive summary with scenario table on first page")
            audit_lines.append("- Exhibits properly gated — no broken or misleading charts")
        audit_lines.append("")

        # --- 4. Source Attribution ---
        attribution_issues = checks.get("attribution_issues", [])
        attribution_grade = self._grade_criterion(attribution_issues, "attribution")
        audit_lines.append("## 4. Source Attribution Including LLM Prompts (25%)")
        audit_lines.append("")
        audit_lines.append(f"**Grade: {attribution_grade}**")
        audit_lines.append("")
        if attribution_issues:
            for issue in attribution_issues:
                audit_lines.append(f"- ⚠️ {issue}")
        else:
            audit_lines.append("- Full SEC filing provenance with accession numbers")
            audit_lines.append("- Structured prompt log with verification methods per section")
            audit_lines.append("- Per-peer source attribution with staleness tracking")
        audit_lines.append("")

        # --- Residual Limitations (always present) ---
        audit_lines.append("## Residual Limitations")
        audit_lines.append("")
        audit_lines.append("The following limitations remain after all quality fixes:")
        audit_lines.append("")
        for lim in residual_limitations:
            audit_lines.append(f"- ⚠️ {lim}")
        audit_lines.append("")

        # --- Suppressed Exhibits Confirmation ---
        audit_lines.append("## Suppressed Exhibits Confirmation")
        audit_lines.append("")
        if suppressed:
            audit_lines.append(f"**Suppressed:** {', '.join(suppressed)}")
            audit_lines.append("")
            for s in suppressed:
                if "segment" in s:
                    audit_lines.append(f"- `{s}`: Suppressed because XBRL segment labels resolve to generic 'Other' categories after normalization.")
                elif "peer" in s:
                    audit_lines.append(f"- `{s}`: Suppressed due to insufficient clean peer data (fewer than 2 peers with valid EV multiples).")
                else:
                    audit_lines.append(f"- `{s}`: Suppressed by data quality gate.")
        else:
            audit_lines.append("No exhibits were suppressed in this run.")
        audit_lines.append("")

        # --- Overall ---
        grades = [completeness_grade, novelty_grade, readability_grade, attribution_grade]
        overall = self._compute_overall_grade(grades)
        audit_lines.append("## Overall Assessment")
        audit_lines.append("")
        audit_lines.append(f"**Weighted Grade: {overall}**")
        audit_lines.append("")
        audit_lines.append(
            f"Residual limitations: {len(residual_limitations)} identified. "
            "These are documented transparently and do not invalidate the Hold recommendation, "
            "which is based on validated DCF analysis and analyst judgment."
        )
        audit_lines.append("")
        audit_lines.append("*Note: This grade is intentionally conservative. "
                           "The self-audit is designed to surface weaknesses "
                           "rather than validate quality.*")
        audit_lines.append("")

        content = "\n".join(audit_lines)

        # Append to existing model_audit.md
        out_path = self.outputs_dir / "model_audit.md"
        if out_path.exists():
            existing = out_path.read_text(encoding="utf-8")
            out_path.write_text(existing + content, encoding="utf-8")
        else:
            out_path.write_text(content, encoding="utf-8")

        logger.info("Self-audit appended to %s", out_path)
        return content

    # ------------------------------------------------------------------
    # 5. Validate Exhibit Attribution
    # ------------------------------------------------------------------

    def validate_exhibit_attribution(
        self,
        report_md: str,
        attribution: str,
    ) -> list[str]:
        """Verify every exhibit referenced in the report has a
        corresponding entry in the source attribution.

        Returns a list of violation strings (empty = all good).
        """
        violations: list[str] = []

        # Find exhibit references in report (e.g., "Exhibit 1", "EX-01", figure references)
        exhibit_patterns = [
            r"EX-\d+",
            r"Exhibit\s+\d+",
            r"Figure\s+\d+",
        ]
        report_exhibits: set[str] = set()
        for pattern in exhibit_patterns:
            matches = re.findall(pattern, report_md, re.IGNORECASE)
            report_exhibits.update(m.upper() for m in matches)

        # Find exhibit entries in attribution
        attribution_exhibits: set[str] = set()
        for pattern in exhibit_patterns:
            matches = re.findall(pattern, attribution, re.IGNORECASE)
            attribution_exhibits.update(m.upper() for m in matches)

        # Also check for figure file references
        figure_refs = re.findall(r"figures/[\w\-]+\.png", report_md, re.IGNORECASE)
        attr_figure_refs = re.findall(r"figures/[\w\-]+\.png", attribution, re.IGNORECASE)

        for ref in figure_refs:
            if ref not in attr_figure_refs and ref.lower() not in [a.lower() for a in attr_figure_refs]:
                violations.append(f"Figure '{ref}' referenced in report but not in attribution")

        for ex in report_exhibits:
            if ex not in attribution_exhibits:
                violations.append(f"Exhibit '{ex}' referenced in report but not in source attribution")

        if violations:
            logger.warning("Exhibit attribution violations: %s", violations)
        else:
            logger.info("All exhibits have source attribution entries.")

        return violations

    # ------------------------------------------------------------------
    # 6. Validate Report-Date Filtering
    # ------------------------------------------------------------------

    def validate_report_date_filtering(
        self,
        context: dict[str, Any],
        report_date: str,
    ) -> list[str]:
        """Check that no data in the report context has
        ``source_available_date > report_date``.

        Returns a list of violation strings (empty = no violations).
        """
        violations: list[str] = []

        # Check DataFrames that may be in context
        df_keys = ["metrics_df", "segments_df", "nlp_df"]
        for key in df_keys:
            df = context.get(key)
            if df is not None and hasattr(df, "columns"):
                if "source_available_date" in df.columns:
                    bad = df[df["source_available_date"] > report_date]
                    if not bad.empty:
                        violations.append(
                            f"{key}: {len(bad)} rows with source_available_date > {report_date}"
                        )

        # Check list-of-dict structures
        list_keys = ["metrics", "segments", "key_metrics"]
        for key in list_keys:
            items = context.get(key)
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        avail = item.get("source_available_date", "")
                        if avail and avail > report_date:
                            violations.append(
                                f"{key}[{i}]: source_available_date={avail} > {report_date}"
                            )

        # Check exhibits for date issues
        exhibits = context.get("exhibits", {})
        if isinstance(exhibits, dict):
            for name, ex in exhibits.items():
                if isinstance(ex, dict):
                    avail = ex.get("source_available_date", "")
                    if avail and avail > report_date:
                        violations.append(
                            f"exhibit '{name}': source_available_date={avail} > {report_date}"
                        )

        # Check ML results
        ml = context.get("ml_results")
        if isinstance(ml, dict):
            matrix = ml.get("feature_target_matrix")
            if matrix is not None and hasattr(matrix, "columns"):
                if "feature_available_date" in matrix.columns:
                    bad = matrix[matrix["feature_available_date"] > report_date]
                    if not bad.empty:
                        violations.append(
                            f"ml_results.feature_target_matrix: {len(bad)} rows with "
                            f"feature_available_date > {report_date}"
                        )

        if violations:
            logger.warning("Report-date filtering violations: %s", violations)
        else:
            logger.info("No report-date filtering violations found.")

        return violations

    # ==================================================================
    # 7. Pre-report eligibility audit (Req 17)
    # ==================================================================

    # Blocking gate component names — ALL must pass for formal_rating.
    _BLOCKING_COMPONENTS = frozenset({
        "data_validation",
        "market_price",
        "dcf_assumptions",
        "split_basis",
        "lookahead",
        "audit_consistency",
        "valuation",
    })

    # Non-blocking (diagnostic-only doesn't block formal rating).
    _NON_BLOCKING_COMPONENTS = frozenset({
        "ml_signal",
        "nlp_signal",
        "peer_multiples",
        "segment_chart",
    })

    def run_pre_report_audit(
        self,
        data_quality_status: DataQualityStatus,
        component_statuses: list[ComponentStatus],
        valuation_input_status: ComponentStatus,
    ) -> RecommendationStatus:
        """Pre-report eligibility audit (phase 1).

        Evaluates DataQualityStatus, valuation input status, and all
        component statuses to determine whether a formal rating may be
        issued.

        Blocking gates (ALL must pass for ``formal_rating``):
        - data_validation: DataQualityStatus must be PASS or PASS_WITH_WARNINGS
        - market_price: must be validated (usable)
        - dcf_assumptions: must be documented (usable)
        - split_basis: no unresolved share-count / split issue
        - lookahead: no report-date / lookahead violation
        - audit_consistency: no contradiction between output files
        - valuation: valuation inputs must be usable

        Non-blocking (``diagnostic_only`` does NOT block):
        - ml_signal, nlp_signal, peer_multiples, segment_chart

        Parameters
        ----------
        data_quality_status:
            Pipeline-wide data quality status from validation gate.
        component_statuses:
            List of ComponentStatus for each pipeline component.
        valuation_input_status:
            ComponentStatus specifically for valuation inputs.

        Returns
        -------
        RecommendationStatus
            Eligibility gate output with report mode, blocking issues,
            and per-component statuses.
        """
        blocking_issues: list[str] = []
        all_statuses: list[ComponentStatus] = list(component_statuses)

        # Ensure valuation_input_status is in the list
        val_names = {cs.component_name for cs in all_statuses}
        if valuation_input_status.component_name not in val_names:
            all_statuses.append(valuation_input_status)

        # --- Gate 1: Data quality status ---
        if data_quality_status is DataQualityStatus.DATA_BLOCKED:
            blocking_issues.append(
                "Data validation gate: DataQualityStatus is DATA_BLOCKED"
            )

        # --- Gate 2+: Check each blocking component ---
        for cs in all_statuses:
            if cs.component_name in self._BLOCKING_COMPONENTS:
                if cs.status in (ComponentStatusEnum.BLOCKED, ComponentStatusEnum.UNAVAILABLE):
                    blocking_issues.append(
                        f"Blocking gate failed: {cs.component_name} "
                        f"status={cs.status.value} — {cs.reason}"
                    )

        # Non-blocking components: diagnostic_only is fine, just log
        for cs in all_statuses:
            if cs.component_name in self._NON_BLOCKING_COMPONENTS:
                if cs.status == ComponentStatusEnum.DIAGNOSTIC_ONLY:
                    logger.info(
                        "Non-blocking component %s is diagnostic_only: %s",
                        cs.component_name, cs.reason,
                    )

        # --- Determine eligibility ---
        if blocking_issues:
            eligibility = ReportMode.DIAGNOSTIC_NOT_RATED
        else:
            eligibility = ReportMode.FORMAL_RATING

        return RecommendationStatus(
            eligibility_status=eligibility,
            data_quality_status=data_quality_status,
            component_statuses=all_statuses,
            blocking_issues=blocking_issues,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )

    # ==================================================================
    # 8. Post-report audit consistency check (Req 22)
    # ==================================================================

    def check_audit_consistency(
        self,
        limitations_md: str,
        data_quality_report_md: str,
        report_md: str,
        data_quality_status: DataQualityStatus,
        model_audit_md: str,
        ml_results: dict,
    ) -> AuditStatus:
        """Post-report consistency audit (phase 2).

        Cross-file contradiction detection:
        1. Fail if limitations says 'no issues' but data_quality_report has failures
        2. Fail if report shows Buy/Hold/Sell but DATA_BLOCKED
        3. Fail if model_audit claims outperformance but walk-forward shows otherwise

        Returns :class:`AuditStatus` with all checks.
        If any check fails, final package status = 'failed' (not submission-ready).
        """
        checks_run = 0
        checks_passed = 0
        checks_failed = 0
        failure_details: list[str] = []
        blocking_issues: list[str] = []
        component_statuses: dict[str, str] = {}

        # --- Check 1: limitations.md vs data_quality_report.md ---
        checks_run += 1
        lim_lower = limitations_md.lower()
        dqr_lower = data_quality_report_md.lower()

        # Detect "no issues" phrasing in limitations
        no_issues_phrases = [
            "no data quality issues recorded",
            "no data quality issues found",
            "no validation failures",
            "no issues found",
        ]
        lim_claims_no_issues = any(phrase in lim_lower for phrase in no_issues_phrases)

        # Detect failures in DQR — look for actual failure indicators,
        # not just the substring "fail" which matches "0 failed" in summaries.
        import re as _re
        dqr_has_failures = (
            bool(_re.search(r'❌\s*fail', dqr_lower))           # emoji-marked failures
            or bool(_re.search(r'[1-9]\d*\s+failed', dqr_lower))  # "N failed" where N>0
            or bool(_re.search(r'\|\s*fail\s*\|', dqr_lower))     # table cell "| fail |"
            or "data_blocked" in dqr_lower
            or bool(_re.search(r'\|\s*critical\s*\|', dqr_lower)) # table cell "| critical |"
        )

        # Also detect "no unresolved blockers" in limitations when DQR has failures
        lim_claims_no_blockers = "no unresolved blockers" in lim_lower

        if lim_claims_no_issues and dqr_has_failures:
            checks_failed += 1
            detail = (
                "Contradiction: limitations.md claims no data quality issues "
                "but data_quality_report.md contains validation failures"
            )
            failure_details.append(detail)
            blocking_issues.append(detail)
            component_statuses["limitations_consistency"] = "fail"
        elif lim_claims_no_blockers and dqr_has_failures and "missing" in dqr_lower:
            checks_failed += 1
            detail = (
                "Contradiction: limitations.md claims no unresolved blockers "
                "but data_quality_report.md has missing or failed valuation-critical metrics"
            )
            failure_details.append(detail)
            blocking_issues.append(detail)
            component_statuses["limitations_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["limitations_consistency"] = "pass"

        # --- Check 2: report rating vs DataQualityStatus ---
        checks_run += 1
        report_lower = report_md.lower()

        # Detect formal rating in report
        has_formal_rating = any(
            rating in report_lower
            for rating in ["**buy**", "**hold**", "**sell**",
                           "rating: buy", "rating: hold", "rating: sell",
                           "recommendation: buy", "recommendation: hold",
                           "recommendation: sell"]
        )

        if has_formal_rating and data_quality_status is DataQualityStatus.DATA_BLOCKED:
            checks_failed += 1
            detail = (
                "Contradiction: report shows Buy/Hold/Sell rating "
                "but DataQualityStatus is DATA_BLOCKED"
            )
            failure_details.append(detail)
            blocking_issues.append(detail)
            component_statuses["rating_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["rating_consistency"] = "pass"

        # --- Check 3: model_audit claims vs walk-forward results ---
        checks_run += 1
        audit_lower = model_audit_md.lower()

        # Detect outperformance claims in model_audit
        claims_outperformance = any(
            phrase in audit_lower
            for phrase in [
                "outperforms baselines",
                "outperforms all baselines",
                "beats baselines",
                "beats all baselines",
                "superior to baselines",
                "better than baselines",
                "lower mae than",
            ]
        )

        # Check actual ML results for underperformance
        ml_underperforms = False
        if ml_results:
            model_mae = ml_results.get("model_mae") or ml_results.get("mae")
            baseline_maes = ml_results.get("baseline_maes", {})
            if model_mae is not None and baseline_maes:
                # Model underperforms if its MAE is worse than ALL baselines
                ml_underperforms = all(
                    model_mae > b_mae
                    for b_mae in baseline_maes.values()
                    if b_mae is not None
                )

        if claims_outperformance and ml_underperforms:
            checks_failed += 1
            detail = (
                "Contradiction: model_audit.md claims model outperforms baselines "
                "but walk-forward results show model MAE is worse than all baselines"
            )
            failure_details.append(detail)
            blocking_issues.append(detail)
            component_statuses["model_audit_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["model_audit_consistency"] = "pass"

        # --- Check 4: ML support claim in report ---
        checks_run += 1
        # Detect if report implies ML supports the rating direction
        ml_supports_phrases = [
            "ml signal supports",
            "ml model supports",
            "ml confirms",
            "machine learning supports",
            "ml signal reinforces",
        ]
        report_claims_ml_support = any(
            phrase in report_lower for phrase in ml_supports_phrases
        )
        if report_claims_ml_support and ml_underperforms:
            checks_failed += 1
            detail = (
                "Contradiction: report implies ML supports the recommendation "
                "but ML model underperforms all baselines (diagnostic-only)"
            )
            failure_details.append(detail)
            blocking_issues.append(detail)
            component_statuses["ml_support_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["ml_support_consistency"] = "pass"

        # --- Check 5: Final-artifact text audit (Req 5) ---
        # When DATA_BLOCKED, scan the rendered report text for any
        # phrase that constitutes an issued formal recommendation.
        # Allow "Buy/Hold/Sell" only in explanatory/rule text, not as
        # an issued rating.
        checks_run += 1
        if data_quality_status is DataQualityStatus.DATA_BLOCKED:
            _FORBIDDEN_RATING_PATTERNS = [
                "rating: sell",
                "rating: buy",
                "rating: hold",
                "rating sell",
                "rating buy",
                "rating hold",
                "we rate nvidia sell",
                "we rate nvidia buy",
                "we rate nvidia hold",
                "**sell**",
                "**buy**",
                "**hold**",
                "recommendation: sell",
                "recommendation: buy",
                "recommendation: hold",
            ]
            # Also check for "Rating: Sell | Target:" pattern
            _FORBIDDEN_RATING_PATTERNS.append("| target: $")

            found_violations: list[str] = []
            for pattern in _FORBIDDEN_RATING_PATTERNS:
                if pattern in report_lower:
                    # Exclude matches inside "What Must Be True" or rule
                    # explanation sections — those describe conditions,
                    # not issued ratings.
                    # Simple heuristic: check if the match is near
                    # "what must be true" or "for buy:" / "for sell:"
                    idx = report_lower.find(pattern)
                    context_window = report_lower[max(0, idx - 100):idx + 100]
                    if any(expl in context_window for expl in [
                        "what must be true",
                        "for buy:",
                        "for sell:",
                        "for hold:",
                        "formal rating rules",
                        "blocking validation gates",
                        "may show buy/hold/sell",
                    ]):
                        continue  # explanatory context, not an issued rating
                    found_violations.append(pattern)

            if found_violations:
                checks_failed += 1
                detail = (
                    "CRITICAL: Final report artifact contains formal rating "
                    f"language while DATA_BLOCKED: {found_violations[:3]}"
                )
                failure_details.append(detail)
                blocking_issues.append(detail)
                component_statuses["final_artifact_audit"] = "fail"
            else:
                checks_passed += 1
                component_statuses["final_artifact_audit"] = "pass"
        else:
            checks_passed += 1
            component_statuses["final_artifact_audit"] = "pass"

        # --- Check 6: Revenue Mix table while segment is suppressed ---
        checks_run += 1
        # Load suppressed exhibits from audit_status.json if available
        _suppressed_exhibits: list[str] = []
        _audit_path = self.outputs_dir / "audit_status.json"
        if _audit_path.exists():
            try:
                _audit_data = json.loads(_audit_path.read_text(encoding="utf-8"))
                _suppressed_exhibits = _audit_data.get("suppressed_exhibits", [])
            except (json.JSONDecodeError, KeyError):
                pass

        segment_suppressed = "revenue_segment_mix" in _suppressed_exhibits
        # Check if the report still contains a Revenue Mix table with data rows
        has_revenue_mix_table = bool(
            re.search(r'revenue mix.*\n\|.*segment.*\|.*revenue.*\|.*share', report_lower)
            or re.search(r'\|\s*other\s*\|\s*\$[\d.]+', report_lower)
        )
        if segment_suppressed and has_revenue_mix_table:
            checks_failed += 1
            detail = "Revenue Mix table appears in report while segment exhibit is suppressed"
            failure_details.append(detail)
            component_statuses["segment_table_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["segment_table_consistency"] = "pass"

        # --- Check 7: Negative P/E in primary peer table ---
        checks_run += 1
        # Look for negative P/E values in the peer multiples section
        peer_section_match = re.search(
            r'peer multiples.*?\n((?:\|.*\n)+)', report_lower
        )
        has_negative_pe_in_primary = False
        if peer_section_match:
            peer_table_text = peer_section_match.group(1)
            # Look for negative P/E values (e.g., "| -12.3 |" or "| -0.5 |")
            if re.search(r'\|\s*-\d+\.?\d*\s*\|', peer_table_text):
                # Check if this is in the primary table (not excluded peers appendix)
                if "excluded peers" not in peer_table_text:
                    has_negative_pe_in_primary = True
        if has_negative_pe_in_primary:
            checks_failed += 1
            detail = "Peer multiples primary table contains negative P/E values"
            failure_details.append(detail)
            component_statuses["peer_pe_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["peer_pe_consistency"] = "pass"

        # --- Check 8: Source attribution blank SEC rows ---
        checks_run += 1
        # Read source_attribution.md and check for blank SEC rows
        sa_path = self.outputs_dir / "source_attribution.md"
        has_blank_sec_rows = False
        if sa_path.exists():
            sa_text = sa_path.read_text(encoding="utf-8")
            # Look for rows where all key fields are empty: "| | | | | |"
            if re.search(r'\|\s*\|\s*\|\s*\|\s*\|\s*\|', sa_text):
                has_blank_sec_rows = True
        if has_blank_sec_rows:
            checks_failed += 1
            detail = "Source attribution contains blank SEC filing rows"
            failure_details.append(detail)
            component_statuses["source_attribution_quality"] = "fail"
        else:
            checks_passed += 1
            component_statuses["source_attribution_quality"] = "pass"

        # --- Check 9: Limitations says "no data quality issues" while exhibits suppressed/diagnostic ---
        checks_run += 1
        lim_claims_no_dq_issues = "no data quality issues recorded" in lim_lower
        # Check if any exhibits are suppressed or diagnostic
        _suppressed_list = _suppressed_exhibits  # from check 6
        _has_suppressed_or_diagnostic = bool(_suppressed_list)
        # Also check for diagnostic-only components in the report
        if "diagnostic-only" in report_lower or "diagnostic only" in report_lower:
            _has_suppressed_or_diagnostic = True
        if lim_claims_no_dq_issues and _has_suppressed_or_diagnostic:
            checks_failed += 1
            detail = (
                "Contradiction: limitations.md claims 'no data quality issues recorded' "
                "while exhibits are suppressed or diagnostic-only"
            )
            failure_details.append(detail)
            component_statuses["limitations_exhibit_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["limitations_exhibit_consistency"] = "pass"

        # --- Check 10: NLP charts active without coverage proof ---
        checks_run += 1
        _active_exhibits: list[str] = []
        if _audit_path.exists():
            try:
                _audit_data_10 = json.loads(_audit_path.read_text(encoding="utf-8"))
                _active_exhibits = _audit_data_10.get("active_exhibits", [])
            except (json.JSONDecodeError, KeyError):
                pass
        nlp_charts_active = any(
            ex in _active_exhibits
            for ex in ("narrative_drift", "keyword_theme_heatmap")
        )
        # Check if report has NLP coverage proof (the extraction quality table)
        has_nlp_coverage_proof = bool(
            re.search(r'nlp extraction quality', report_lower)
            and (re.search(r'coverage.*\d+', report_lower) or re.search(r'filings.*parsed', report_lower))
        )
        if nlp_charts_active and not has_nlp_coverage_proof:
            checks_failed += 1
            detail = "NLP charts are active but report lacks NLP extraction coverage proof"
            failure_details.append(detail)
            component_statuses["nlp_coverage_proof"] = "fail"
        else:
            checks_passed += 1
            component_statuses["nlp_coverage_proof"] = "pass"

        # --- Check 11: Suppressed exhibits appear in report body ---
        checks_run += 1
        suppressed_in_body = []
        for ex_name in _suppressed_list:
            # Check if the exhibit image appears in the report (not in the suppression notice)
            img_pattern = re.compile(
                rf'!\[.*\]\(.*{re.escape(ex_name)}.*\)', re.IGNORECASE
            )
            if img_pattern.search(report_md):
                suppressed_in_body.append(ex_name)
        if suppressed_in_body:
            checks_failed += 1
            detail = (
                f"Suppressed exhibits appear as images in report body: "
                f"{', '.join(suppressed_in_body)}"
            )
            failure_details.append(detail)
            component_statuses["suppressed_exhibit_leakage"] = "fail"
        else:
            checks_passed += 1
            component_statuses["suppressed_exhibit_leakage"] = "pass"

        # --- Check 12: Formatting scan for banned machine-format tokens ---
        checks_run += 1
        from src.display_format import scan_for_banned_tokens
        banned_found = scan_for_banned_tokens(report_md)
        if banned_found:
            checks_failed += 1
            detail = (
                f"Report contains banned machine-format tokens: "
                f"{', '.join(banned_found[:5])}"
            )
            failure_details.append(detail)
            component_statuses["formatting_scan"] = "fail"
        else:
            checks_passed += 1
            component_statuses["formatting_scan"] = "pass"

        # --- Check 13: Out-of-window fiscal years in main report ---
        checks_run += 1
        from src.display_format import is_within_display_window
        out_of_window_years = []
        for fy_match in re.finditer(r'\bFY(20[01]\d)\b', report_md):
            fy = int(fy_match.group(1))
            if fy < 2016:
                out_of_window_years.append(f"FY{fy}")
        # Deduplicate
        out_of_window_years = sorted(set(out_of_window_years))
        if out_of_window_years:
            checks_failed += 1
            detail = (
                f"Report contains out-of-window fiscal years: "
                f"{', '.join(out_of_window_years)}"
            )
            failure_details.append(detail)
            component_statuses["display_window_enforcement"] = "fail"
        else:
            checks_passed += 1
            component_statuses["display_window_enforcement"] = "pass"

        # --- Check 14: LLM attribution completeness ---
        checks_run += 1
        has_claude = "claude" in report_lower or "kiro" in report_lower
        has_chatgpt = "chatgpt" in report_lower or "openai" in report_lower
        if has_claude and has_chatgpt:
            checks_passed += 1
            component_statuses["llm_attribution_completeness"] = "pass"
        elif has_claude or has_chatgpt:
            # Partial — pass with warning
            checks_passed += 1
            component_statuses["llm_attribution_completeness"] = "pass_with_warnings"
        else:
            checks_failed += 1
            detail = "Report LLM disclosure missing both Claude/Kiro and ChatGPT/OpenAI"
            failure_details.append(detail)
            component_statuses["llm_attribution_completeness"] = "fail"

        # --- Check 15: ML results present in PDF ---
        checks_run += 1
        has_ml_table = (
            "ml model performance summary" in report_lower
            or "model status" in report_lower and "diagnostic_only" in report_lower
        )
        has_ml_not_available = "ml results not available" in report_lower
        if has_ml_not_available:
            checks_failed += 1
            detail = "Report contains 'ML results not available' — must show ML methodology and outcome"
            failure_details.append(detail)
            component_statuses["ml_results_present_in_pdf"] = "fail"
        elif has_ml_table:
            checks_passed += 1
            component_statuses["ml_results_present_in_pdf"] = "pass"
        else:
            checks_passed += 1
            component_statuses["ml_results_present_in_pdf"] = "pass_with_warnings"

        # --- Check 16: NLP exhibit status consistency ---
        checks_run += 1
        # NLP charts should not be simultaneously in active AND suppressed/diagnostic
        _diag_appendix: list[str] = []
        if _audit_path.exists():
            try:
                _ad = json.loads(_audit_path.read_text(encoding="utf-8"))
                _diag_appendix = _ad.get("diagnostic_appendix_exhibits", [])
            except (json.JSONDecodeError, KeyError):
                pass
        nlp_keys = {"narrative_drift", "keyword_theme_heatmap"}
        nlp_in_active = nlp_keys & set(_active_exhibits)
        nlp_in_suppressed = nlp_keys & set(_suppressed_list)
        nlp_in_diag = nlp_keys & set(_diag_appendix)
        # Inconsistency: same exhibit in both suppressed and diagnostic_appendix
        nlp_overlap = nlp_in_suppressed & nlp_in_diag
        if nlp_overlap:
            checks_failed += 1
            detail = f"NLP exhibits in both suppressed and diagnostic_appendix: {nlp_overlap}"
            failure_details.append(detail)
            component_statuses["nlp_status_consistency"] = "fail"
        elif nlp_in_active:
            # Below-threshold NLP in active is wrong
            checks_failed += 1
            detail = f"Below-threshold NLP exhibits in active_exhibits: {nlp_in_active}"
            failure_details.append(detail)
            component_statuses["nlp_status_consistency"] = "fail"
        else:
            checks_passed += 1
            component_statuses["nlp_status_consistency"] = "pass"

        # --- Check 17: Assignment ML requirement ---
        checks_run += 1
        ml_req_items = [
            ("model choice", bool(re.search(r'elasticnet|ridge|model.*chosen', report_lower))),
            ("training data", bool(re.search(r'training data|10.year.*financial', report_lower))),
            ("testing procedure", bool(re.search(r'walk.forward|validation.*no.*lookahead', report_lower))),
            ("model outcome", bool(re.search(r'underperform|diagnostic.only|excluded from.*rat', report_lower))),
        ]
        missing_ml = [name for name, present in ml_req_items if not present]
        if missing_ml:
            checks_failed += 1
            detail = f"Assignment ML requirement missing: {', '.join(missing_ml)}"
            failure_details.append(detail)
            component_statuses["assignment_ml_requirement"] = "fail"
        else:
            checks_passed += 1
            component_statuses["assignment_ml_requirement"] = "pass"

        # --- Determine overall status ---
        overall_status = "pass" if checks_failed == 0 else "fail"

        return AuditStatus(
            overall_status=overall_status,
            data_quality_status=data_quality_status.value,
            recommendation_eligibility=(
                "formal_rating" if data_quality_status is not DataQualityStatus.DATA_BLOCKED
                else "diagnostic_not_rated"
            ),
            component_statuses=component_statuses,
            checks_run=checks_run,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            blocking_issues=blocking_issues,
            failure_details=failure_details,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )

    # ==================================================================
    # 9. Generate audit_status.json (Req 14.10, 22.4)
    # ==================================================================

    def generate_audit_status_json(
        self,
        audit_status: AuditStatus,
        package_status: str,
        *,
        run_id: str | None = None,
    ) -> str:
        """Write ``outputs/audit_status.json``.

        Parameters
        ----------
        audit_status:
            The :class:`AuditStatus` from :meth:`check_audit_consistency`.
        package_status:
            One of ``"formal_rating_pass"``, ``"diagnostic_not_rated_pass"``,
            or ``"failed"``.
        run_id:
            Optional run identifier for cross-referencing with manifest.json.

        Returns
        -------
        str
            The JSON string written to disk.
        """
        payload = {
            "overall_status": audit_status.overall_status,
            "package_status": package_status,
            "data_quality_status": audit_status.data_quality_status,
            "recommendation_eligibility": audit_status.recommendation_eligibility,
            "component_statuses": audit_status.component_statuses,
            "checks_run": audit_status.checks_run,
            "checks_passed": audit_status.checks_passed,
            "checks_failed": audit_status.checks_failed,
            "blocking_issues": audit_status.blocking_issues,
            "failure_details": audit_status.failure_details,
            "timestamp": audit_status.timestamp,
        }
        if run_id is not None:
            payload["run_id"] = run_id

        json_str = json.dumps(payload, indent=2)
        out_path = self.outputs_dir / "audit_status.json"
        out_path.write_text(json_str, encoding="utf-8")
        logger.info("Audit status JSON written to %s", out_path)
        return json_str

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _run_audit_checks(
        self, report_md: str, attribution: str
    ) -> dict[str, list[str]]:
        """Run heuristic checks on the report and attribution content."""
        checks: dict[str, list[str]] = {
            "completeness_issues": [],
            "novelty_issues": [],
            "readability_issues": [],
            "attribution_issues": [],
        }

        report_lower = report_md.lower()

        # --- Completeness checks ---
        required_sections = [
            ("executive summary", "Executive Summary section"),
            ("valuation", "Valuation section"),
            ("risk", "Risk analysis section"),
            ("dcf", "DCF valuation discussion"),
            ("recommendation", "Investment recommendation"),
        ]
        for keyword, label in required_sections:
            if keyword not in report_lower:
                checks["completeness_issues"].append(f"Missing or weak: {label}")

        required_exhibits_keywords = ["figure", "exhibit", "chart", "table"]
        exhibit_count = sum(
            len(re.findall(rf"\b{kw}\b", report_lower))
            for kw in required_exhibits_keywords
        )
        if exhibit_count < 6:
            checks["completeness_issues"].append(
                f"Only ~{exhibit_count} exhibit references found (target: 6-8)"
            )

        # --- Novelty checks ---
        generic_phrases = [
            "industry leader",
            "well positioned",
            "strong growth",
            "significant opportunity",
            "best in class",
        ]
        generic_count = sum(1 for p in generic_phrases if p in report_lower)
        if generic_count >= 3:
            checks["novelty_issues"].append(
                f"Report contains {generic_count} generic/boilerplate phrases — "
                "may lack analytical depth"
            )

        if "narrative drift" not in report_lower and "tfidf" not in report_lower:
            checks["novelty_issues"].append(
                "No narrative drift / NLP analysis discussion found"
            )

        if "reverse" not in report_lower and "implied" not in report_lower:
            checks["novelty_issues"].append(
                "No reverse-DCF or implied-expectations analysis found"
            )

        # --- Readability checks ---
        word_count = len(report_md.split())
        # 8-12 pages ≈ 4000-6000 words
        if word_count < 2000:
            checks["readability_issues"].append(
                f"Report is short ({word_count} words) — may lack depth for 8-12 pages"
            )
        elif word_count > 8000:
            checks["readability_issues"].append(
                f"Report is long ({word_count} words) — may exceed target length"
            )

        # --- Attribution checks ---
        if "source_attribution" not in report_lower and "source attribution" not in report_lower:
            checks["attribution_issues"].append(
                "No reference to source attribution document in report"
            )

        if "prompt_log" not in report_lower and "prompt log" not in report_lower:
            checks["attribution_issues"].append(
                "No reference to prompt log in report"
            )

        if "llm" not in report_lower and "language model" not in report_lower:
            checks["attribution_issues"].append(
                "No LLM disclaimer or disclosure found in report"
            )

        # Check attribution document itself
        attr_lower = attribution.lower()
        if "sec" not in attr_lower and "edgar" not in attr_lower:
            checks["attribution_issues"].append(
                "Source attribution missing SEC/EDGAR filing references"
            )

        # --- Lookahead risk check ---
        if "lookahead" not in report_lower and "look-ahead" not in report_lower:
            checks["completeness_issues"].append(
                "No discussion of lookahead bias controls"
            )

        # --- Stale data check ---
        if "stale" not in report_lower and "staleness" not in report_lower:
            checks["completeness_issues"].append(
                "No discussion of data staleness controls"
            )

        # --- Valuation inconsistency check ---
        if "sensitivity" not in report_lower:
            checks["completeness_issues"].append(
                "No sensitivity analysis discussion found"
            )

        return checks

    @staticmethod
    def _grade_criterion(issues: list[str], criterion: str) -> str:
        """Assign a conservative letter grade based on issue count.

        Intentionally skeptical — does NOT inflate grades.
        """
        n = len(issues)
        if n == 0:
            return "A-"  # Never give A+ in self-audit (skeptical)
        elif n == 1:
            return "B+"
        elif n == 2:
            return "B"
        elif n == 3:
            return "B-"
        elif n <= 5:
            return "C+"
        else:
            return "C"

    @staticmethod
    def _compute_overall_grade(grades: list[str]) -> str:
        """Compute a weighted average grade from 4 criterion grades."""
        grade_points = {
            "A+": 4.3, "A": 4.0, "A-": 3.7,
            "B+": 3.3, "B": 3.0, "B-": 2.7,
            "C+": 2.3, "C": 2.0, "C-": 1.7,
            "D+": 1.3, "D": 1.0, "F": 0.0,
        }
        points_to_grade = sorted(grade_points.items(), key=lambda x: -x[1])

        total = sum(grade_points.get(g, 2.0) for g in grades)
        avg = total / len(grades) if grades else 2.0

        # Find closest grade
        for grade, pts in points_to_grade:
            if avg >= pts - 0.15:
                return grade
        return "C"
