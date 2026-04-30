"""
NVDA Quantamental Engine — XBRL Parser.

Parses SEC companyfacts JSON into structured financial tables,
validates against published values, and generates data-quality reports.

Requirements: 3.1–3.8, 15.2–15.4
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DataQualityStatus, EngineConfig, FinancialFact, ValidatedMetric
from src.split_adjuster import SplitAdjuster

logger = logging.getLogger(__name__)


class XBRLParser:
    """Parse companyfacts JSON into structured :class:`FinancialFact` rows."""

    # Priority-ordered XBRL concept names per metric.
    # The parser tries each concept in order and uses the first one found.
    CONCEPT_MAP: dict[str, list[str]] = {
        "revenue": [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],
        "cogs": [
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold",
            "CostOfRevenue",
            "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        ],
        "gross_profit": [
            "GrossProfit",
        ],
        "operating_income": [
            "OperatingIncomeLoss",
            "IncomeLossFromContinuingOperations",
        ],
        "net_income": [
            "NetIncomeLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "ProfitLoss",
        ],
        "diluted_eps": [
            "EarningsPerShareDiluted",
        ],
        "r_and_d": [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ],
        "sga": [
            "SellingGeneralAndAdministrativeExpense",
            "GeneralAndAdministrativeExpense",
        ],
        "operating_cash_flow": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "capex": [
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "CapitalExpenditureDiscontinuedOperations",
        ],
        "stock_based_compensation": [
            "ShareBasedCompensation",
            "AllocatedShareBasedCompensationExpense",
            "ShareBasedCompensationIncludingDiscontinuedOperations",
        ],
        "cash_and_securities": [
            "CashCashEquivalentsAndShortTermInvestments",
            "CashAndCashEquivalentsAtCarryingValue",
        ],
        "short_term_debt": [
            "ShortTermBorrowings",
            "ShortTermDebtCurrent",
            "DebtCurrent",
        ],
        "long_term_debt": [
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermDebtAndCapitalLeaseObligations",
        ],
        "total_debt": [
            "LongTermDebt",
            "DebtAndCapitalLeaseObligations",
            "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
            "Debt",
        ],
        "total_assets": [
            "Assets",
        ],
        "shareholders_equity": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "current_assets": [
            "AssetsCurrent",
        ],
        "current_liabilities": [
            "LiabilitiesCurrent",
        ],
        "diluted_shares": [
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "CommonStockSharesOutstanding",
            "WeightedAverageNumberDilutedSharesOutstandingAdjustment",
        ],
        "segment_revenue": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
        ],
    }

    # Form-type priority for deduplication (lower = preferred).
    _FORM_PRIORITY = {"10-K": 0, "10-K/A": 1, "10-Q": 2, "10-Q/A": 3}

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self._fallbacks_used: list[dict] = []
        self._missing_tags: list[dict] = []

    # Tolerance (days) for matching a fact's period_end to a fiscal year end.
    _FY_END_TOLERANCE_DAYS = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Columns returned by parse_companyfacts (base + period metadata).
    _BASE_COLUMNS = [
        "ticker", "fiscal_period", "fiscal_year", "filing_date",
        "source_available_date", "accession_number", "metric_name",
        "value", "unit", "form_type",
    ]
    _PERIOD_COLUMNS = [
        "fiscal_period_type", "period_start", "period_end",
        "duration_days", "frame", "xbrl_concept",
        "selection_rank", "selection_reason",
    ]
    _SPLIT_COLUMNS = [
        "raw_value", "adjusted_value", "adjustment_factor",
        "adjustment_basis", "validation_basis",
    ]
    OUTPUT_COLUMNS = _BASE_COLUMNS + _PERIOD_COLUMNS + _SPLIT_COLUMNS

    def parse_companyfacts(self, facts_json: dict) -> pd.DataFrame:
        """Parse companyfacts JSON into a DataFrame matching FinancialFact schema.

        Parameters
        ----------
        facts_json:
            Raw JSON from the SEC companyfacts endpoint.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, fiscal_period, fiscal_year, filing_date,
            source_available_date, accession_number, metric_name, value,
            unit, form_type, fiscal_period_type, period_start, period_end,
            duration_days, frame, xbrl_concept.
        """
        rows: list[dict] = []
        self._fallbacks_used = []
        self._missing_tags = []

        for metric_name in self.CONCEPT_MAP:
            concept, entries = self.resolve_concept(facts_json, metric_name)
            if concept is None or entries is None:
                self._missing_tags.append({"metric": metric_name})
                logger.warning("No XBRL concept found for metric '%s'", metric_name)
                continue

            # Track fallback usage
            primary = self.CONCEPT_MAP[metric_name][0]
            if concept != primary:
                self._fallbacks_used.append({
                    "metric": metric_name,
                    "primary": primary,
                    "used": concept,
                })

            for entry in entries:
                period_start = entry.get("start")  # None for instant facts
                period_end = entry.get("end")
                duration_days = self._compute_duration_days(period_start, period_end)
                fiscal_period_type = self._classify_period_type(
                    period_start, period_end, duration_days,
                )

                rows.append({
                    "ticker": self.config.ticker,
                    "fiscal_period": entry.get("fp", ""),
                    "fiscal_year": entry.get("fy", 0),
                    "filing_date": entry.get("filed", ""),
                    "source_available_date": entry.get("filed", ""),
                    "accession_number": entry.get("accn", ""),
                    "metric_name": metric_name,
                    "value": entry.get("val"),
                    "unit": self._infer_unit(metric_name, concept, facts_json),
                    "form_type": entry.get("form", ""),
                    "fiscal_period_type": fiscal_period_type,
                    "period_start": period_start,
                    "period_end": period_end,
                    "duration_days": duration_days,
                    "frame": entry.get("frame"),
                    "xbrl_concept": entry.get("_concept", concept),
                    "selection_rank": 0,
                    "selection_reason": "",
                })

        if not rows:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        df = pd.DataFrame(rows)

        # Build fiscal-year-end lookup from known_values or from the data
        fy_end_dates = self._build_fy_end_dates(df)

        # Apply annual fact selection per metric + fiscal year
        df = self._select_annual_facts(df, fy_end_dates)

        # Apply standard deduplication for non-annual periods
        df = self.deduplicate_facts(df)

        # Fill missing capex from filing HTML when XBRL data is absent
        filings_dir = self.config.raw_dir / "filings"
        df = self._fill_capex_from_html_fallback(df, filings_dir)

        # Apply split adjustments to per-share metrics (Reqs 3.1, 3.2)
        adjuster = SplitAdjuster(self.config)
        df = adjuster.apply_split_adjustments(df)

        return df

    def resolve_concept(
        self, facts_json: dict, metric_name: str
    ) -> tuple[Optional[str], Optional[list[dict]]]:
        """Merge entries from all matching concepts in priority order.

        Earlier concepts in the CONCEPT_MAP list are preferred.  Each
        entry is tagged with ``_concept`` so the caller knows which XBRL
        concept it came from.  Entries from higher-priority concepts take
        precedence when the same (fy, fp, start, end) tuple appears in
        multiple concepts.

        Returns
        -------
        tuple[str | None, list[dict] | None]
            (primary_concept_name, merged_entries) or (None, None).
        """
        concepts = self.CONCEPT_MAP.get(metric_name, [])
        taxonomies = facts_json.get("facts", {})

        primary_concept: Optional[str] = None
        seen_keys: set[tuple] = set()
        merged: list[dict] = []

        for concept in concepts:
            for taxonomy_name, taxonomy_data in taxonomies.items():
                if concept in taxonomy_data:
                    concept_data = taxonomy_data[concept]
                    units = concept_data.get("units", {})
                    # Try USD first, then USD/shares, then shares
                    entries: Optional[list[dict]] = None
                    for unit_key in ("USD", "USD/shares", "shares"):
                        if unit_key in units and units[unit_key]:
                            entries = units[unit_key]
                            break
                    if entries is None:
                        for unit_key, unit_entries in units.items():
                            if unit_entries:
                                entries = unit_entries
                                break
                    if not entries:
                        continue

                    if primary_concept is None:
                        primary_concept = concept

                    for entry in entries:
                        key = (
                            entry.get("fy"),
                            entry.get("fp"),
                            entry.get("start"),
                            entry.get("end"),
                        )
                        if key not in seen_keys:
                            seen_keys.add(key)
                            tagged = dict(entry)
                            tagged["_concept"] = concept
                            merged.append(tagged)

        if not merged:
            return None, None
        return primary_concept, merged

    def deduplicate_facts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prefer 10-K over 10-Q for same period; latest filing over amendments.

        Deduplication key: (metric_name, fiscal_year, fiscal_period).

        For instant/balance-sheet facts (fiscal_period_type == "instant"),
        when multiple entries exist for the same (metric, fiscal_year,
        fiscal_period), prefer the entry with the latest period_end date.
        This ensures we select the current-year balance rather than the
        comparative prior-year balance that 10-K filings also report.
        """
        if df.empty:
            return df

        df = df.copy()
        df["_form_priority"] = df["form_type"].map(self._FORM_PRIORITY).fillna(99)
        df["_filing_date_sort"] = pd.to_datetime(df["filing_date"], errors="coerce")
        # For instant facts, prefer the latest period_end (current-year
        # balance over comparative prior-year balance).
        df["_period_end_sort"] = pd.to_datetime(df["period_end"], errors="coerce")

        df = df.sort_values(
            ["metric_name", "fiscal_year", "fiscal_period",
             "_form_priority", "_period_end_sort", "_filing_date_sort"],
            ascending=[True, True, True, True, False, False],
        )
        df = df.drop_duplicates(
            subset=["metric_name", "fiscal_year", "fiscal_period"],
            keep="first",
        )
        df = df.drop(columns=["_form_priority", "_filing_date_sort", "_period_end_sort"])
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Fiscal-year selection (Reqs 15.2–15.6)
    # ------------------------------------------------------------------

    def _build_fy_end_dates(self, df: pd.DataFrame) -> dict[int, date]:
        """Build a mapping of fiscal_year → fiscal_year_end_date.

        Uses the period_end of annual-duration facts from 10-K filings
        to infer the fiscal year end date for each year.  Falls back to
        the ``known_values`` fixture when available via config.
        """
        fy_ends: dict[int, date] = {}

        # Try to infer from annual-duration 10-K facts in the data
        annual_10k = df[
            (df["fiscal_period_type"] == "annual")
            & (df["form_type"].isin(["10-K", "10-K/A"]))
            & (df["period_end"].notna())
        ]
        for _, row in annual_10k.iterrows():
            fy = int(row["fiscal_year"])
            try:
                end_dt = date.fromisoformat(row["period_end"])
            except (ValueError, TypeError):
                continue
            # For a given FY, keep the latest period_end we see
            if fy not in fy_ends or end_dt > fy_ends[fy]:
                fy_ends[fy] = end_dt

        # Also try to load from the published values fixture
        try:
            published_path = self.config.fixtures_dir / "nvda_published_values.json"
            if published_path.exists():
                import json
                with open(published_path) as f:
                    pub = json.load(f)
                for fy_label, fy_data in pub.get("values", {}).items():
                    try:
                        fy_num = int(fy_label.replace("FY", ""))
                    except (ValueError, AttributeError):
                        continue
                    fy_end_str = fy_data.get("fiscal_year_end")
                    if fy_end_str:
                        try:
                            fy_ends.setdefault(fy_num, date.fromisoformat(fy_end_str))
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass  # Non-critical; we can still work with inferred dates

        return fy_ends

    def _select_annual_facts(
        self, df: pd.DataFrame, fy_end_dates: dict[int, date],
    ) -> pd.DataFrame:
        """Apply annual fact selection across all metrics and fiscal years.

        For each (metric, fiscal_year) where we have a known fiscal year end
        date, select the best annual fact using ``_select_annual_fact``.
        Non-annual rows (quarterly, YTD, instant) are passed through unchanged.

        Returns the full DataFrame with annual rows replaced by the selected
        best candidates, and ``selection_rank`` / ``selection_reason`` populated.
        """
        if df.empty or not fy_end_dates:
            return df

        # Separate annual FY rows from everything else
        annual_fy_mask = (df["fiscal_period"] == "FY") & (df["fiscal_period_type"] == "annual")
        non_annual = df[~annual_fy_mask].copy()
        annual_candidates = df[annual_fy_mask].copy()

        if annual_candidates.empty:
            return df

        selected_rows: list[pd.Series] = []

        for metric in annual_candidates["metric_name"].unique():
            metric_candidates = annual_candidates[annual_candidates["metric_name"] == metric]

            # Group by the fiscal years we know about
            for fy, fy_end in fy_end_dates.items():
                result = self._select_annual_fact(metric_candidates, metric, fy, fy_end)
                if result is not None:
                    selected_rows.append(result)

        if selected_rows:
            selected_df = pd.DataFrame(selected_rows)
            result = pd.concat([non_annual, selected_df], ignore_index=True)
        else:
            result = non_annual.copy()

        return result.reset_index(drop=True)

    def _select_annual_fact(
        self,
        candidates: pd.DataFrame,
        metric: str,
        fy: int,
        fy_end: date,
    ) -> pd.Series | None:
        """Select the best annual fact for a given metric and fiscal year.

        The algorithm matches facts to the correct fiscal year using the
        actual ``period_end`` date (not the ``fy`` field from companyfacts),
        because SEC EDGAR tags ALL facts from a 10-K filing with the
        filing's fiscal year — including comparative periods from prior years.

        Selection ranking (Reqs 15.2, 15.5):
            1. 10-K annual-duration fact whose period_end matches the target
               fiscal year end (within tolerance) — rank 1
            2. Frame-tagged annual fact matching the target FY end — rank 2
            3. Latest amendment for the target FY end — rank 3

        Rejection rules (Reqs 15.3, 15.4):
            - Quarterly facts (duration < 100 days) are never selected
            - YTD facts (duration 100–340 days) are never selected

        Parameters
        ----------
        candidates:
            All annual-duration FY-period facts for this metric (across all
            actual fiscal years — they may share the same ``fy`` tag).
        metric:
            The metric name (for logging).
        fy:
            The target fiscal year number.
        fy_end:
            The known fiscal year end date for this FY.

        Returns
        -------
        pd.Series | None
            The selected fact row with selection_rank and selection_reason
            populated, or None if no suitable candidate exists.
        """
        if candidates.empty:
            return None

        tolerance = timedelta(days=self._FY_END_TOLERANCE_DAYS)

        # Step 1: Filter to candidates whose period_end matches the target FY end
        matched: list[tuple[int, pd.Series, str]] = []  # (rank, row, reason)

        for idx, row in candidates.iterrows():
            period_end_str = row.get("period_end")
            if not period_end_str:
                continue

            try:
                period_end_dt = date.fromisoformat(period_end_str)
            except (ValueError, TypeError):
                continue

            # Check if this fact's period_end matches the target FY end
            if abs((period_end_dt - fy_end).days) > tolerance.days:
                continue

            # Reject quarterly and YTD facts (should already be filtered
            # by caller, but enforce here as a safety net)
            duration = row.get("duration_days")
            if duration is not None:
                if duration < 100:
                    logger.debug(
                        "Rejecting quarterly fact for %s FY%d: duration=%d days",
                        metric, fy, duration,
                    )
                    continue
                if 100 <= duration < 340:
                    logger.debug(
                        "Rejecting YTD fact for %s FY%d: duration=%d days",
                        metric, fy, duration,
                    )
                    continue

            # Determine rank based on form type and frame
            form = row.get("form_type", "")
            frame = row.get("frame")
            is_10k = form in ("10-K", "10-K/A")

            if is_10k and frame:
                rank = 1
                reason = "10-K annual-duration frame-tagged"
            elif is_10k:
                rank = 1
                reason = "10-K annual-duration"
            elif frame:
                rank = 2
                reason = "frame-tagged annual"
            else:
                rank = 3
                reason = "annual-duration fallback"

            matched.append((rank, row.copy(), reason))

        if not matched:
            logger.debug(
                "No annual fact found for %s FY%d (fy_end=%s)",
                metric, fy, fy_end.isoformat(),
            )
            return None

        # Step 2: Handle amendments — among same-rank candidates, latest
        # filing_date wins (Req 15.6)
        matched = self._handle_amendments(matched)

        # Step 3: Pick the best (lowest rank, then latest filing_date)
        matched.sort(key=lambda x: (x[0], self._filing_date_sort_key(x[1])))
        best_rank, best_row, best_reason = matched[0]

        best_row = best_row.copy()
        best_row["selection_rank"] = best_rank
        best_row["selection_reason"] = best_reason
        # Ensure the fiscal_year reflects the target FY (not the fy tag from EDGAR)
        best_row["fiscal_year"] = fy

        return best_row

    def _handle_amendments(
        self,
        ranked_candidates: list[tuple[int, pd.Series, str]],
    ) -> list[tuple[int, pd.Series, str]]:
        """Deterministic amendment handling: latest filing_date wins.

        Among candidates with the same selection rank, keep only the one
        with the latest filing_date. This ensures amendments (10-K/A)
        supersede original filings when they have the same or better rank.

        Req 15.6: The parser SHALL handle fiscal-year amendments
        deterministically: the latest amendment for a given fiscal year
        supersedes prior filings.
        """
        if len(ranked_candidates) <= 1:
            return ranked_candidates

        # Group by rank
        by_rank: dict[int, list[tuple[int, pd.Series, str]]] = {}
        for item in ranked_candidates:
            rank = item[0]
            by_rank.setdefault(rank, []).append(item)

        result: list[tuple[int, pd.Series, str]] = []
        for rank, items in sorted(by_rank.items()):
            if len(items) == 1:
                result.append(items[0])
            else:
                # Latest filing_date wins
                items.sort(key=lambda x: self._filing_date_sort_key(x[1]), reverse=False)
                # reverse=False because _filing_date_sort_key returns negated
                # timestamp for descending sort — actually let's just sort
                # by filing_date descending directly
                best = max(items, key=lambda x: x[1].get("filing_date", ""))
                # Update reason to note amendment handling
                _, row, reason = best
                result.append((rank, row, f"{reason} (latest amendment)"))

        return result

    @staticmethod
    def _filing_date_sort_key(row: pd.Series) -> str:
        """Return filing_date as a string for sorting (descending = latest first)."""
        return row.get("filing_date", "")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_against_published(
        self,
        parsed_df: pd.DataFrame,
        known_values: dict,
        tolerance_pct: float = 2.0,
    ) -> pd.DataFrame:
        """Compare parsed values against known published values.

        Parameters
        ----------
        parsed_df:
            Output of :meth:`parse_companyfacts`.
        known_values:
            Dict in the format of ``tests/fixtures/known_validation_values.json``.
        tolerance_pct:
            Acceptable percentage difference (default 2%).

        Returns
        -------
        pd.DataFrame
            Columns: metric, fiscal_year, parsed_value, published_value,
            diff_pct, status (pass / fail / missing).
        """
        tolerance = known_values.get("tolerance_pct", tolerance_pct)
        values_by_fy = known_values.get("values", {})

        # Map from known_values keys to our metric names
        known_to_metric = {
            "revenue": "revenue",
            "net_income": "net_income",
            "operating_cash_flow": "operating_cash_flow",
            "capex": "capex",
            "diluted_eps": "diluted_eps",
            "total_debt": "total_debt",
            "r_and_d": "r_and_d",
            "diluted_shares": "diluted_shares",
        }

        results: list[dict] = []

        for fy_label, fy_data in values_by_fy.items():
            # Extract fiscal year number from label like "FY2024"
            try:
                fy_num = int(fy_label.replace("FY", ""))
            except (ValueError, AttributeError):
                continue

            for known_key, metric_name in known_to_metric.items():
                published_value = fy_data.get(known_key)
                if published_value is None:
                    continue

                # Find parsed value for this metric + FY (annual)
                mask = (
                    (parsed_df["metric_name"] == metric_name)
                    & (parsed_df["fiscal_year"] == fy_num)
                    & (parsed_df["fiscal_period"] == "FY")
                )
                matched = parsed_df.loc[mask]

                if matched.empty:
                    results.append({
                        "metric": metric_name,
                        "fiscal_year": fy_num,
                        "parsed_value": None,
                        "published_value": published_value,
                        "diff_pct": None,
                        "status": "missing",
                    })
                    continue

                parsed_value = matched.iloc[0]["value"]
                if parsed_value is None or published_value == 0:
                    results.append({
                        "metric": metric_name,
                        "fiscal_year": fy_num,
                        "parsed_value": parsed_value,
                        "published_value": published_value,
                        "diff_pct": None,
                        "status": "missing" if parsed_value is None else "fail",
                    })
                    continue

                diff_pct = abs(parsed_value - published_value) / abs(published_value) * 100
                status = "pass" if diff_pct <= tolerance else "fail"

                results.append({
                    "metric": metric_name,
                    "fiscal_year": fy_num,
                    "parsed_value": parsed_value,
                    "published_value": published_value,
                    "diff_pct": round(diff_pct, 4),
                    "status": status,
                })

        return pd.DataFrame(results)

    def validate_with_gate(
        self,
        parsed_df: pd.DataFrame,
        published_values: dict,
    ) -> tuple[DataQualityStatus, list[ValidatedMetric]]:
        """Validate parsed data using the :class:`DataValidationGate`.

        Delegates to :meth:`DataValidationGate.validate_parsed_data` for
        enhanced validation with severity, blocker status, and
        :class:`DataQualityStatus`.

        Parameters
        ----------
        parsed_df:
            DataFrame output from :meth:`parse_companyfacts`.
        published_values:
            Dict matching the ``tests/fixtures/nvda_published_values.json``
            schema.

        Returns
        -------
        tuple[DataQualityStatus, list[ValidatedMetric]]
            Overall pipeline status and per-metric validation results.

        Reqs: 14.9, 3.6
        """
        from src.data_validation import DataValidationGate

        gate = DataValidationGate(self.config)
        return gate.validate_parsed_data(parsed_df, published_values)

    def compute_data_quality_status(
        self,
        validations: list[ValidatedMetric],
    ) -> DataQualityStatus:
        """Compute :class:`DataQualityStatus` from a list of validated metrics.

        Convenience method that applies the same rules as
        :class:`DataValidationGate`:

        - ``DATA_BLOCKED`` if any :class:`ValidatedMetric` has
          ``severity="critical"`` AND ``blocker=True`` AND
          ``status`` in ``("fail", "missing")``.
        - ``PASS_WITH_WARNINGS`` if any non-critical failures or
          non-blocker missing values exist.
        - ``PASS`` otherwise.

        Parameters
        ----------
        validations:
            List of :class:`ValidatedMetric` from validation.

        Returns
        -------
        DataQualityStatus

        Reqs: 14.9, 3.6
        """
        # Check for critical blockers first
        for m in validations:
            if (
                m.severity == "critical"
                and m.blocker
                and m.status in ("fail", "missing")
            ):
                return DataQualityStatus.DATA_BLOCKED

        # Check for non-critical warnings
        has_warnings = any(
            m.status == "fail" and m.severity != "critical"
            for m in validations
        )
        if has_warnings:
            return DataQualityStatus.PASS_WITH_WARNINGS

        # Check for non-blocker missing metrics
        has_non_blocker_missing = any(
            m.status == "missing" and not m.blocker
            for m in validations
        )
        if has_non_blocker_missing:
            return DataQualityStatus.PASS_WITH_WARNINGS

        return DataQualityStatus.PASS

    def generate_data_quality_report(
        self,
        parsed: pd.DataFrame,
        validation: pd.DataFrame | list[ValidatedMetric],
        data_quality_status: DataQualityStatus | None = None,
    ) -> str:
        """Generate markdown for ``outputs/data_quality_report.md``.

        Sections: data quality status (if provided), missing tags, fallback
        tags used, coverage %, validation results.

        Parameters
        ----------
        parsed:
            DataFrame from :meth:`parse_companyfacts`.
        validation:
            Either a :class:`pd.DataFrame` (legacy format from
            :meth:`validate_against_published`) or a ``list`` of
            :class:`ValidatedMetric` (from :meth:`validate_with_gate`).
        data_quality_status:
            Optional :class:`DataQualityStatus` to include in the report
            header.  When provided, a "Data Quality Status" section is
            rendered at the top showing the overall status and any
            blocking issues.

        Reqs: 14.9, 3.8
        """
        # Normalise validation input to a list of dicts for uniform rendering
        is_validated_metric_list = (
            isinstance(validation, list)
            and len(validation) > 0
            and isinstance(validation[0], ValidatedMetric)
        )
        if is_validated_metric_list:
            val_rows: list[dict] = []
            for vm in validation:
                val_rows.append({
                    "metric": vm.metric_name,
                    "fiscal_year": vm.fiscal_year,
                    "parsed_value": vm.parsed_value,
                    "published_value": vm.published_value,
                    "diff_pct": vm.diff_pct,
                    "status": vm.status,
                    "severity": vm.severity,
                    "blocker": vm.blocker,
                })
            val_df = pd.DataFrame(val_rows) if val_rows else pd.DataFrame()
            has_severity = True
        elif isinstance(validation, list) and len(validation) == 0:
            val_df = pd.DataFrame()
            has_severity = False
        else:
            val_df = validation
            has_severity = False

        lines: list[str] = [
            "# Data Quality Report",
            "",
            f"**Ticker:** {self.config.ticker}",
            f"**Report Date:** {self.config.report_date}",
            "",
        ]

        # --- Data Quality Status (new, Req 14.9) ---
        if data_quality_status is not None:
            status_label = {
                DataQualityStatus.PASS: "✅ PASS",
                DataQualityStatus.PASS_WITH_WARNINGS: "⚠️ PASS_WITH_WARNINGS",
                DataQualityStatus.DATA_BLOCKED: "🚫 DATA_BLOCKED",
            }.get(data_quality_status, str(data_quality_status.value))

            lines.append("## Data Quality Status")
            lines.append("")
            lines.append(f"**Overall Status:** {status_label}")
            lines.append("")

            if data_quality_status is DataQualityStatus.DATA_BLOCKED:
                lines.append(
                    "> **⛔ Formal recommendation is BLOCKED.** "
                    "One or more critical metrics failed validation. "
                    "All downstream outputs are diagnostic only."
                )
                lines.append("")

                # List blocking issues
                if is_validated_metric_list:
                    blockers = [
                        vm for vm in validation
                        if vm.blocker and vm.status in ("fail", "missing")
                    ]
                    if blockers:
                        lines.append("### Blocking Issues")
                        lines.append("")
                        for b in blockers:
                            if b.status == "fail" and b.diff_pct is not None:
                                lines.append(
                                    f"- **{b.metric_name}** FY{b.fiscal_year}: "
                                    f"parsed={b.parsed_value:,.0f}, "
                                    f"published={b.published_value:,.0f}, "
                                    f"diff={b.diff_pct:.2f}% "
                                    f"(tolerance={b.tolerance}%)"
                                )
                            else:
                                lines.append(
                                    f"- **{b.metric_name}** FY{b.fiscal_year}: "
                                    f"**{b.status}**"
                                )
                        lines.append("")

            elif data_quality_status is DataQualityStatus.PASS_WITH_WARNINGS:
                lines.append(
                    "> **⚠️ Non-critical warnings detected.** "
                    "Formal recommendation may proceed, but review "
                    "warnings below."
                )
                lines.append("")

            else:
                lines.append(
                    "> All core metrics validated within tolerance. "
                    "Pipeline may proceed with formal recommendation."
                )
                lines.append("")

        # --- Missing Tags ---
        lines.append("## Missing XBRL Tags")
        lines.append("")
        if self._missing_tags:
            for item in self._missing_tags:
                lines.append(f"- **{item['metric']}**: no matching XBRL concept found")
        else:
            lines.append("No missing tags.")
        lines.append("")

        # --- Fallback Tags ---
        lines.append("## Fallback Tags Used")
        lines.append("")
        if self._fallbacks_used:
            lines.append("| Metric | Primary Concept | Fallback Used |")
            lines.append("|--------|----------------|---------------|")
            for fb in self._fallbacks_used:
                lines.append(f"| {fb['metric']} | {fb['primary']} | {fb['used']} |")
        else:
            lines.append("No fallback tags were needed.")
        lines.append("")

        # --- Coverage ---
        lines.append("## Coverage")
        lines.append("")
        total_metrics = len(self.CONCEPT_MAP)
        found = total_metrics - len(self._missing_tags)
        pct = (found / total_metrics * 100) if total_metrics else 0
        lines.append(f"- Metrics defined: {total_metrics}")
        lines.append(f"- Metrics found: {found}")
        lines.append(f"- Coverage: {pct:.1f}%")
        lines.append("")

        if not parsed.empty:
            fy_range = sorted(int(y) for y in parsed["fiscal_year"].unique())
            lines.append(f"- Fiscal years covered: {fy_range}")
            lines.append(f"- Total parsed facts: {len(parsed)}")
        lines.append("")

        # --- Validation Results ---
        lines.append("## Validation Against Published Values")
        lines.append("")
        if val_df.empty:
            lines.append("No validation data available.")
        else:
            # Build header based on whether severity info is available
            if has_severity:
                lines.append(
                    "| Metric | FY | Parsed | Published | Diff % "
                    "| Severity | Blocker | Status |"
                )
                lines.append(
                    "|--------|----|--------|-----------|--------"
                    "|----------|---------|--------|"
                )
            else:
                lines.append("| Metric | FY | Parsed | Published | Diff % | Status |")
                lines.append("|--------|----|--------|-----------|--------|--------|")

            for _, row in val_df.iterrows():
                parsed_val = (
                    f"{row['parsed_value']:,.0f}"
                    if row["parsed_value"] is not None and pd.notna(row["parsed_value"])
                    else "N/A"
                )
                pub_val = (
                    f"{row['published_value']:,.0f}"
                    if row["published_value"] is not None and pd.notna(row["published_value"])
                    else "N/A"
                )
                diff = (
                    f"{row['diff_pct']:.2f}%"
                    if row["diff_pct"] is not None and pd.notna(row["diff_pct"])
                    else "N/A"
                )
                status_icon = {"pass": "✅", "fail": "❌", "missing": "⚠️"}.get(
                    row["status"], row["status"]
                )

                if has_severity:
                    severity = row.get("severity", "")
                    blocker = "Yes" if row.get("blocker") else "No"
                    lines.append(
                        f"| {row['metric']} | {row['fiscal_year']} "
                        f"| {parsed_val} | {pub_val} | {diff} "
                        f"| {severity} | {blocker} | {status_icon} {row['status']} |"
                    )
                else:
                    lines.append(
                        f"| {row['metric']} | {row['fiscal_year']} "
                        f"| {parsed_val} | {pub_val} | {diff} | {status_icon} {row['status']} |"
                    )

            # Summary
            total = len(val_df)
            passed = (val_df["status"] == "pass").sum()
            failed = (val_df["status"] == "fail").sum()
            missing = (val_df["status"] == "missing").sum()
            lines.append("")
            lines.append(f"**Summary:** {passed}/{total} passed, {failed} failed, {missing} missing")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_capex_from_html_fallback(
        self, df: pd.DataFrame, filings_dir: Path,
    ) -> pd.DataFrame:
        """Fill missing capex values by extracting from filing HTML cash flow statements.

        Some NVIDIA quarterly filings (FY2023-Q1/Q2/Q3, FY2024-Q1/Q2) don't
        include ``PaymentsToAcquireProductiveAssets`` or
        ``PaymentsToAcquirePropertyPlantAndEquipment`` in the SEC XBRL
        companyfacts JSON.  However, the values ARE present in the filing
        HTML under "Purchases related to property and equipment and
        intangible assets".

        This method identifies fiscal periods that have ``operating_cash_flow``
        but are missing ``capex``, then attempts to extract capex from the
        corresponding filing HTML files.

        Parameters
        ----------
        df:
            DataFrame from the main parsing pipeline (before split adjustment).
        filings_dir:
            Path to the directory containing filing HTML files
            (e.g. ``data/raw/filings/``).

        Returns
        -------
        pd.DataFrame
            The input DataFrame with any extracted capex rows appended.
        """
        if df.empty:
            return df

        # Identify periods with OCF but no capex
        ocf_periods = df[df["metric_name"] == "operating_cash_flow"][
            ["fiscal_year", "fiscal_period", "accession_number", "filing_date",
             "source_available_date", "form_type", "period_start", "period_end",
             "duration_days", "fiscal_period_type", "frame"]
        ].drop_duplicates(subset=["fiscal_year", "fiscal_period"])

        capex_periods = df[df["metric_name"] == "capex"][
            ["fiscal_year", "fiscal_period"]
        ].drop_duplicates()

        # Find OCF periods missing capex
        merged = ocf_periods.merge(
            capex_periods,
            on=["fiscal_year", "fiscal_period"],
            how="left",
            indicator=True,
        )
        missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        if missing.empty:
            return df

        new_rows: list[dict] = []

        for _, row in missing.iterrows():
            accn = row["accession_number"]
            fy = int(row["fiscal_year"])
            fp = row["fiscal_period"]

            # Build the HTML file path
            html_path = filings_dir / f"{accn}.html"
            if not html_path.exists():
                logger.warning(
                    "HTML file not found for capex fallback: %s (FY%d-%s)",
                    html_path, fy, fp,
                )
                continue

            # Extract capex from HTML
            capex_value = self._extract_capex_from_html(html_path)
            if capex_value is None:
                logger.warning(
                    "Could not extract capex from HTML for FY%d-%s (accn=%s)",
                    fy, fp, accn,
                )
                continue

            # Values in the HTML are in millions
            capex_usd = capex_value * 1_000_000

            logger.warning(
                "HTML fallback capex: FY%d-%s = $%.0f (from %s)",
                fy, fp, capex_usd, accn,
            )

            new_rows.append({
                "ticker": self.config.ticker,
                "fiscal_period": fp,
                "fiscal_year": fy,
                "filing_date": row["filing_date"],
                "source_available_date": row["source_available_date"],
                "accession_number": accn,
                "metric_name": "capex",
                "value": capex_usd,
                "unit": "USD",
                "form_type": row["form_type"],
                "fiscal_period_type": row["fiscal_period_type"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "duration_days": row["duration_days"],
                "frame": row["frame"],
                "xbrl_concept": "html_fallback_capex",
                "selection_rank": 0,
                "selection_reason": "html_cash_flow_extraction",
            })

            self._fallbacks_used.append({
                "metric": "capex",
                "primary": "PaymentsToAcquireProductiveAssets",
                "used": f"html_fallback_capex ({accn})",
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            logger.info(
                "Filled %d missing capex values from HTML fallback", len(new_rows),
            )

        return df

    @staticmethod
    def _extract_capex_from_html(html_path: Path) -> Optional[float]:
        """Extract capex value from a filing HTML cash flow statement.

        Searches for "Purchases related to property and equipment and
        intangible assets" followed by a parenthesized number (negative
        cash flow convention).

        Parameters
        ----------
        html_path:
            Path to the filing HTML file.

        Returns
        -------
        float | None
            The capex value in millions (as a positive number), or None
            if the pattern is not found.
        """
        try:
            html_content = html_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read HTML file %s: %s", html_path, exc)
            return None

        # Strip HTML tags and clean entities
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator=" ")
        except Exception as exc:
            logger.warning("Failed to parse HTML %s: %s", html_path, exc)
            return None

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Search for the capex line item with parenthesized value.
        # NVIDIA uses two phrasings across filing vintages:
        #   - "Purchases related to property and equipment and intangible assets"
        #   - "Purchases of property and equipment and intangible assets"
        patterns = [
            (
                r"[Pp]urchases\s+related\s+to\s+property\s+and\s+equipment"
                r"\s+and\s+intangible\s+assets?\s*"
                r"\(\s*([\d,\s]+)\s*\)"
            ),
            (
                r"[Pp]urchases\s+of\s+property\s+and\s+equipment"
                r"\s+and\s+intangible\s+assets?\s*"
                r"\(\s*([\d,\s]+)\s*\)"
            ),
        ]
        match = None
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                break
        if not match:
            return None

        # Clean the matched value: remove commas and whitespace
        raw_value = match.group(1).replace(",", "").replace(" ", "").strip()
        try:
            return float(raw_value)
        except ValueError:
            logger.warning(
                "Could not parse capex value '%s' from %s", match.group(1), html_path,
            )
            return None

    @staticmethod
    def _compute_duration_days(
        period_start: Optional[str], period_end: Optional[str],
    ) -> Optional[int]:
        """Compute the number of days between *period_start* and *period_end*.

        Returns ``None`` when either date is missing (instant / balance-sheet
        items typically have no start date).
        """
        if not period_start or not period_end:
            return None
        try:
            start = date.fromisoformat(period_start)
            end = date.fromisoformat(period_end)
            return (end - start).days
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _classify_period_type(
        period_start: Optional[str],
        period_end: Optional[str],
        duration_days: Optional[int],
    ) -> str:
        """Classify a fact's reporting period.

        Returns one of:
        - ``"instant"``   — no start date (balance-sheet / point-in-time items)
        - ``"annual"``    — 350 ≤ duration_days ≤ 380
        - ``"quarterly"`` — duration_days < 100
        - ``"ytd"``       — 100 ≤ duration_days < 340  (but not annual)
        - ``"other"``     — anything else (edge cases, e.g. 340-349 days)

        Reqs: 15.2, 15.3, 15.4
        """
        if period_start is None:
            return "instant"
        if duration_days is None:
            return "other"
        if 350 <= duration_days <= 380:
            return "annual"
        if duration_days < 100:
            return "quarterly"
        if 100 <= duration_days < 340:
            return "ytd"
        return "other"

    def _infer_unit(
        self, metric_name: str, concept: str, facts_json: dict
    ) -> str:
        """Determine the unit string for a resolved concept."""
        taxonomies = facts_json.get("facts", {})
        for taxonomy_data in taxonomies.values():
            if concept in taxonomy_data:
                units = taxonomy_data[concept].get("units", {})
                if "USD" in units:
                    return "USD"
                if "USD/shares" in units:
                    return "USD/shares"
                if "shares" in units:
                    return "shares"
                # Return first available
                for key in units:
                    return key
        # Fallback based on metric name
        if metric_name in ("diluted_eps",):
            return "USD/shares"
        if metric_name in ("diluted_shares",):
            return "shares"
        return "USD"
