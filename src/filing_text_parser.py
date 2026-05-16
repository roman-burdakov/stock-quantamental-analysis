"""
NVDA Quantamental Engine — Filing Text Parser.

Extracts narrative sections from SEC filing HTML using form-specific
Item-number regex patterns (primary) with section-title fallback.
Produces TextSectionRecord instances and coverage reports.

Reqs: 5.1–5.5
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.config import ComponentStatus, ComponentStatusEnum, EngineConfig, TextSectionRecord


class FilingTextParser:
    """Parse SEC filing HTML into structured narrative sections."""

    # 10-K sections (Req 5.1)
    SECTIONS_10K: dict[str, str] = {
        "business": "Item 1",
        "risk_factors": "Item 1A",
        "mda": "Item 7",
        "quant": "Item 7A",
    }

    # 10-Q sections (Req 5.2)
    SECTIONS_10Q: dict[str, str] = {
        "risk_factors": "Part II, Item 1A",
        "mda": "Part I, Item 2",
    }

    # Ordered item labels for boundary detection in 10-K filings
    _ORDERED_10K = [
        "Item 1", "Item 1A", "Item 1B", "Item 1C",
        "Item 2", "Item 3", "Item 4",
        "Item 5", "Item 6", "Item 7", "Item 7A",
        "Item 8", "Item 9", "Item 9A", "Item 9B",
        "Item 10", "Item 11", "Item 12", "Item 13", "Item 14", "Item 15",
    ]

    # Ordered item labels for boundary detection in 10-Q filings
    _ORDERED_10Q = [
        "Part I, Item 1", "Part I, Item 2", "Part I, Item 3", "Part I, Item 4",
        "Part II, Item 1", "Part II, Item 1A", "Part II, Item 2",
        "Part II, Item 3", "Part II, Item 4", "Part II, Item 5", "Part II, Item 6",
    ]

    # Title-based fallback patterns
    _TITLE_MAP: dict[str, list[str]] = {
        "Item 1": ["Business"],
        "Item 1A": ["Risk Factors"],
        "Item 7": [
            "Management's Discussion and Analysis",
            "Management\u2019s Discussion and Analysis",
        ],
        "Item 7A": [
            "Quantitative and Qualitative Disclosures About Market Risk",
        ],
        "Part II, Item 1A": ["Risk Factors"],
        "Part I, Item 2": [
            "Management's Discussion and Analysis",
            "Management\u2019s Discussion and Analysis",
        ],
    }

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_filing(
        self,
        html: str,
        accession: str,
        form_type: str,
        filing_date: str,
        source_available_date: str,
    ) -> list[TextSectionRecord]:
        """Extract narrative sections from a filing's HTML.

        Strategy:
          1. Strip HTML to plain text.
          2. For each target section, try Item-number regex (primary).
          3. Fall back to section-title matching if regex misses.
          4. If both fail, fall back to mda_proxy: first ~10,000 chars of
             cleaned filing text after the header (Req 3.2).
          5. Build TextSectionRecord per section.
          6. Save extracted text to data/interim/.

        Each record carries an ``extraction_tier`` attribute:
          - ``full_extraction``: Item-number regex succeeded
          - ``partial_extraction``: title fallback succeeded
          - ``mda_proxy_fallback``: proxy text returned (only for mda)
          - ``failed``: nothing usable found

        Reqs: 3.1–3.6, 14 (extraction quality).
        """
        plain_text = self._strip_html(html)
        sections = self._sections_for_form(form_type)

        results: list[TextSectionRecord] = []
        # Pre-compute proxy text once per filing so all sections can share it.
        proxy_text = self._build_mda_proxy(plain_text)

        for section_name, item_label in sections.items():
            text, status, tier = self._extract_section(
                plain_text, item_label, form_type
            )
            # Apply mda_proxy_fallback to the MD&A section when:
            #  - extraction is missing entirely, OR
            #  - extraction is suspiciously short (< 500 chars) which
            #    typically indicates the regex/title matched a TOC entry
            #    rather than the full narrative section.
            mda_min_chars = self.config.min_nlp_char_mda
            if section_name == "mda" and proxy_text is not None:
                if status == "missing" or len(text) < mda_min_chars:
                    text = proxy_text
                    status = "fallback"
                    tier = "mda_proxy_fallback"

            char_count = len(text)
            record = TextSectionRecord(
                accession_number=accession,
                form_type=form_type,
                section_name=section_name,
                text=text,
                char_count=char_count,
                filing_date=filing_date,
                source_available_date=source_available_date,
                parse_status=status,
            )
            # extraction_tier is a v2 addition. Set as a dynamic attribute
            # to avoid changing the v1 dataclass schema (which would break
            # existing tests). Downstream readers can use getattr(rec,
            # "extraction_tier", None) and gracefully handle the v1 case.
            record.extraction_tier = tier  # type: ignore[attr-defined]
            results.append(record)

        self._save_sections(results, accession)
        return results

    # ------------------------------------------------------------------
    # MD&A proxy fallback (Req 3.2)
    # ------------------------------------------------------------------

    def _build_mda_proxy(self, plain_text: str) -> str | None:
        """Return the first ~10,000 chars of cleaned filing text as an
        mda_proxy fallback (Req 3.2).

        Heuristic, in order of preference:

        1. Find anchor candidates (``PART I``, ``Item 1.``, ``Item 2.``).
        2. Reject candidates that appear inside a table of contents.
           A TOC entry is identified by the presence of dot-leader
           patterns (``........``) or trailing page numbers in the
           same line. Body headings are followed by paragraph text.
        3. Pick the FIRST surviving (non-TOC) candidate. This is
           typically the actual narrative-section heading.
        4. If no non-TOC candidate exists, fall back to the LAST
           candidate (which is at minimum past the TOC).
        5. The candidate must produce ≥500 chars after stripping.

        Quality is gated separately by ``check_proxy_quality()`` against
        a list of MD&A indicator terms (Req 14.1).
        """
        if not plain_text:
            return None
        anchor_pattern = re.compile(
            r"\n\s*(?:PART\s+I|Item\s+1\.|Item\s+2\.)",
            re.IGNORECASE,
        )
        anchors = list(anchor_pattern.finditer(plain_text))
        if not anchors:
            return None

        # TOC detector: an anchor is "in TOC" if the line containing it
        # has a dot-leader run (>=4 dots) OR a trailing page-number.
        toc_line_pattern = re.compile(
            r"\.{4,}|\b\d{1,4}\s*$",
        )

        non_toc = []
        for m in anchors:
            # Extract the line the anchor is on.
            line_start = plain_text.rfind("\n", 0, m.start()) + 1
            line_end = plain_text.find("\n", m.end())
            line = plain_text[line_start : (line_end if line_end != -1 else len(plain_text))]
            if not toc_line_pattern.search(line):
                non_toc.append(m)

        if non_toc:
            anchor = non_toc[0]
        else:
            # All matches are in TOC; fall back to the last one and hope
            # the body section follows immediately.
            anchor = anchors[-1]

        start = anchor.start()
        candidate = plain_text[start : start + 10_000].strip()
        if len(candidate) < 500:
            return None
        return candidate

    @staticmethod
    def check_proxy_quality(text: str) -> tuple[bool, dict[str, int]]:
        """Heuristic quality check on a fallback proxy extraction.

        Returns ``(is_relevant, term_counts)`` where ``is_relevant`` is
        True iff at least one MD&A indicator term appears in the text.
        ``term_counts`` is a dict of indicator → count for transparency.

        Indicator terms (Req 14.1): ``revenue``, ``results of operations``,
        ``net income``, ``gross margin``, ``compared to``, ``quarter``,
        ``fiscal year``.

        This is the automated, deterministic half of Req 14. The manual
        QA workflow (Req 14.2/14.3) writes 10 sample fallback extractions
        for human review and only allows NLP to be active when at least
        8/10 are labeled relevant.
        """
        if not text:
            return False, {}
        indicators = [
            "revenue",
            "results of operations",
            "net income",
            "gross margin",
            "compared to",
            "quarter",
            "fiscal year",
        ]
        lowered = text.lower()
        counts = {t: lowered.count(t) for t in indicators}
        is_relevant = any(c > 0 for c in counts.values())
        return is_relevant, counts

    def check_extraction_quality(
        self,
        records: list[TextSectionRecord],
        config: EngineConfig | None = None,
    ) -> ComponentStatus:
        """Assess text extraction quality for NLP downstream use.

        Returns ``ComponentStatus`` with ``diagnostic_only`` when more than
        50% of filings have below-threshold character counts for required
        sections (MD&A and Risk Factors).

        Thresholds (from config):
            - MD&A: ``min_nlp_char_mda`` (default 500)
            - Risk Factors: ``min_nlp_char_risk`` (default 300)

        Parameters
        ----------
        records:
            All ``TextSectionRecord`` instances across filings.
        config:
            Optional engine config for threshold overrides.

        Returns
        -------
        ComponentStatus
        """
        cfg = config or self.config
        min_mda = cfg.min_nlp_char_mda
        min_risk = cfg.min_nlp_char_risk

        thresholds = {"mda": min_mda, "risk_factors": min_risk}

        # Group records by accession (filing)
        filings: dict[str, dict[str, TextSectionRecord]] = {}
        for r in records:
            if r.section_name not in thresholds:
                continue
            filings.setdefault(r.accession_number, {})[r.section_name] = r

        if not filings:
            return ComponentStatus(
                component_name="text_extraction",
                status=ComponentStatusEnum.UNAVAILABLE,
                reason="No MD&A or Risk Factors sections found in any filing.",
            )

        poor_count = 0
        total_count = len(filings)

        for accession, sections in filings.items():
            filing_poor = False
            for section_name, threshold in thresholds.items():
                rec = sections.get(section_name)
                if rec is None or rec.char_count < threshold:
                    filing_poor = True
                    break
            if filing_poor:
                poor_count += 1

        poor_pct = poor_count / total_count if total_count > 0 else 0.0

        if poor_pct > 0.50:
            return ComponentStatus(
                component_name="text_extraction",
                status=ComponentStatusEnum.DIAGNOSTIC_ONLY,
                reason=(
                    f"{poor_count}/{total_count} filings ({poor_pct:.0%}) have "
                    f"below-threshold text extraction (MD&A < {min_mda} chars "
                    f"or Risk Factors < {min_risk} chars)."
                ),
                details={
                    "poor_count": poor_count,
                    "total_count": total_count,
                    "poor_pct": round(poor_pct * 100, 1),
                },
            )

        return ComponentStatus(
            component_name="text_extraction",
            status=ComponentStatusEnum.USABLE,
            reason="Text extraction quality is adequate for NLP analysis.",
            details={
                "poor_count": poor_count,
                "total_count": total_count,
                "poor_pct": round(poor_pct * 100, 1),
            },
        )

    def generate_coverage_report(
        self, results: list[TextSectionRecord]
    ) -> pd.DataFrame:
        """Build a coverage DataFrame and append to data_quality_report.md.

        Columns: accession, form_type, section, char_count, parse_status, warning
        """
        rows = []
        for r in results:
            warning = ""
            if r.parse_status == "missing":
                warning = "Section not found"
            elif r.parse_status == "fallback":
                warning = "Extracted via title fallback"
            elif r.char_count < 500:
                warning = "Unusually short section"
            rows.append(
                {
                    "accession": r.accession_number,
                    "form_type": r.form_type,
                    "section": r.section_name,
                    "char_count": r.char_count,
                    "parse_status": r.parse_status,
                    "warning": warning,
                }
            )

        df = pd.DataFrame(rows)
        self._append_to_quality_report(df)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sections_for_form(self, form_type: str) -> dict[str, str]:
        ft = form_type.upper().replace("/A", "")
        if "10-K" in ft:
            return self.SECTIONS_10K
        if "10-Q" in ft:
            return self.SECTIONS_10Q
        return self.SECTIONS_10K

    def _strip_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_section(
        self, text: str, item_label: str, form_type: str
    ) -> tuple[str, str, str]:
        """Try Item-number regex, then title fallback.

        Returns ``(text, status, tier)`` where:
        - ``status`` ∈ {"success", "fallback", "missing"} — preserved for
          v1 API compatibility (parse_status field)
        - ``tier`` ∈ {"full_extraction", "partial_extraction", "failed"} —
          v2 extraction-quality tier (Req 3.3)
        """
        # Primary: Item-number regex (Req 5.3)
        extracted = self._extract_by_item_number(text, item_label, form_type)
        if extracted and len(extracted.strip()) > 50:
            return extracted.strip(), "success", "full_extraction"

        # Fallback: section-title matching
        extracted = self._extract_by_title_fallback(text, item_label)
        if extracted and len(extracted.strip()) > 50:
            return extracted.strip(), "fallback", "partial_extraction"

        return "", "missing", "failed"

    # ------------------------------------------------------------------
    # Primary extraction: Item-number regex
    # ------------------------------------------------------------------

    def _extract_by_item_number(
        self, text: str, item_label: str, form_type: str
    ) -> str | None:
        """Find the Item heading in the body (not TOC) and extract until next Item.

        SEC filings typically have a TOC with short Item references, then the
        actual sections with full content. We find ALL occurrences of the
        heading and pick the one followed by the most content.
        """
        heading_re = self._build_heading_regex(item_label)
        matches = list(heading_re.finditer(text))
        if not matches:
            return None

        stop_re = self._build_stop_regex(item_label, form_type)

        best_text = ""
        for m in matches:
            content_start = m.end()
            if stop_re:
                stop_match = stop_re.search(text, content_start)
                if stop_match:
                    candidate = text[content_start : stop_match.start()]
                else:
                    candidate = text[content_start : content_start + 200_000]
            else:
                candidate = text[content_start : content_start + 200_000]

            if len(candidate) > len(best_text):
                best_text = candidate

        return best_text if best_text else None

    def _build_heading_regex(self, item_label: str) -> re.Pattern:
        """Build a regex that matches an Item heading line.

        Handles both 10-K style ("Item 7.") and 10-Q style where
        "Part I" and "Item 2" may appear on separate lines.
        """
        if "," in item_label:
            # 10-Q style: "Part II, Item 1A" — the Part and Item may be
            # on separate lines or the same line in the plain text.
            part, item = [s.strip() for s in item_label.split(",", 1)]
            part_esc = re.escape(part).replace(r"\ ", r"\s+")
            item_esc = re.escape(item).replace(r"\ ", r"\s+")
            # Match: Part ... (optional stuff) ... Item N
            pattern = (
                r"(?:^|\n)\s*"
                + part_esc
                + r"[\s\-—]*(?:[\w\s]*\n\s*)?"
                + item_esc
                + r"\.?"
                + r"[\.\s]"
            )
        else:
            # 10-K style: "Item 7" — need word boundary after number
            # to avoid "Item 1" matching "Item 1A"
            escaped = re.escape(item_label).replace(r"\ ", r"\s+")
            # For items without a letter suffix (Item 1, Item 7), require
            # the match NOT to be followed by a letter (to avoid Item 1 matching Item 1A)
            if re.match(r"Item \d+$", item_label):
                pattern = r"(?:^|\n)\s*" + escaped + r"(?![A-Za-z])" + r"\.?\s"
            else:
                pattern = r"(?:^|\n)\s*" + escaped + r"\.?\s"

        return re.compile(pattern, re.IGNORECASE)

    def _build_stop_regex(
        self, item_label: str, form_type: str
    ) -> re.Pattern | None:
        """Build a regex matching the next section boundary after item_label."""
        stop_labels = self._get_stop_labels(item_label, form_type)
        if not stop_labels:
            return None

        parts = []
        for lbl in stop_labels:
            if "," in lbl:
                # 10-Q: "Part II, Item 1A"
                part, item = [s.strip() for s in lbl.split(",", 1)]
                part_esc = re.escape(part).replace(r"\ ", r"\s+")
                item_esc = re.escape(item).replace(r"\ ", r"\s+")
                parts.append(
                    part_esc + r"[\s\-—]*(?:[\w\s]*\n\s*)?" + item_esc + r"\.?"
                )
            else:
                esc = re.escape(lbl).replace(r"\ ", r"\s+")
                if re.match(r"Item \d+$", lbl):
                    parts.append(esc + r"(?![A-Za-z])\.?")
                else:
                    parts.append(esc + r"\.?")

        # Also stop at PART boundaries
        parts.append(r"PART\s+[IV]+")

        combined = r"(?:^|\n)\s*(?:" + "|".join(parts) + r")\s"
        return re.compile(combined, re.IGNORECASE)

    def _get_stop_labels(self, current_label: str, form_type: str) -> list[str]:
        """Return ordered Item labels that follow current_label."""
        ft = form_type.upper().replace("/A", "")
        if "10-Q" in ft or current_label.startswith("Part"):
            ordered = self._ORDERED_10Q
        else:
            ordered = self._ORDERED_10K

        norm = current_label.strip().lower()
        stops: list[str] = []
        found = False
        for lbl in ordered:
            if lbl.strip().lower() == norm:
                found = True
                continue
            if found:
                stops.append(lbl)

        return stops

    # ------------------------------------------------------------------
    # Fallback extraction: section title
    # ------------------------------------------------------------------

    def _extract_by_title_fallback(
        self, text: str, item_label: str
    ) -> str | None:
        """Search for the section title text rather than Item number."""
        titles = self._TITLE_MAP.get(item_label, [])
        for title in titles:
            pattern = re.compile(
                r"(?:^|\n)\s*" + re.escape(title),
                re.IGNORECASE,
            )
            # Find all matches and pick the one with the most content after it
            matches = list(pattern.finditer(text))
            best = ""
            for m in matches:
                start = m.end()
                # Find next major heading
                next_heading = re.search(
                    r"\n\s*(?:Item\s+\d|Part\s+[IV]+|PART\s+[IV]+)",
                    text[start:],
                    re.IGNORECASE,
                )
                if next_heading:
                    candidate = text[start : start + next_heading.start()]
                else:
                    candidate = text[start : start + 200_000]
                if len(candidate) > len(best):
                    best = candidate

            if best and len(best.strip()) > 50:
                return best

        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_sections(
        self, records: list[TextSectionRecord], accession: str
    ) -> None:
        """Persist extracted sections to data/interim/ as JSON."""
        interim_dir = Path(self.config.interim_dir)
        interim_dir.mkdir(parents=True, exist_ok=True)

        safe_accession = accession.replace("-", "_").replace("/", "_")
        out_path = interim_dir / f"sections_{safe_accession}.json"

        payload = [
            {
                "accession_number": r.accession_number,
                "form_type": r.form_type,
                "section_name": r.section_name,
                "char_count": r.char_count,
                "filing_date": r.filing_date,
                "source_available_date": r.source_available_date,
                "parse_status": r.parse_status,
                "extraction_tier": getattr(r, "extraction_tier", None),
                "text": r.text,
            }
            for r in records
        ]

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown table without requiring tabulate."""
        try:
            return df.to_markdown(index=False)
        except ImportError:
            # Manual markdown table if tabulate is not installed
            cols = list(df.columns)
            lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
            lines.append("| " + " | ".join("---" for _ in cols) + " |")
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
            return "\n".join(lines)

    def _append_to_quality_report(self, df: pd.DataFrame) -> None:
        """Append coverage table to outputs/data_quality_report.md."""
        report_path = Path(self.config.outputs_dir) / "data_quality_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        header = "\n\n## Filing Text Extraction Coverage\n\n"
        table = self._df_to_markdown(df)

        if report_path.exists():
            existing = report_path.read_text(encoding="utf-8")
            marker = "## Filing Text Extraction Coverage"
            if marker in existing:
                before = existing[: existing.index(marker)]
                report_path.write_text(
                    before.rstrip() + header + table + "\n",
                    encoding="utf-8",
                )
            else:
                report_path.write_text(
                    existing.rstrip() + header + table + "\n",
                    encoding="utf-8",
                )
        else:
            report_path.write_text(
                "# Data Quality Report\n" + header + table + "\n",
                encoding="utf-8",
            )
