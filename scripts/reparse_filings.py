"""
Task 2.1 — Re-parse all cached filings with the v2 parser improvements.

For each accession with a section JSON in ``data/interim/``:
  1. Locate the corresponding HTML in ``data/raw/filings/``.
  2. If HTML is small (<100KB), it's likely an SEC archive directory
     page (not the actual filing); attempt to re-fetch via the
     EdgarFetcher's standard resolution.
  3. Re-parse with the updated FilingTextParser.

Emits a summary of extraction-tier distribution before and after.

Run as:
    python scripts/reparse_filings.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from src.config import EngineConfig
from src.edgar_fetch import EdgarFetcher
from src.filing_text_parser import FilingTextParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


SMALL_HTML_THRESHOLD = 100_000  # bytes — anything below this is probably an index page


def _accession_from_section_json(path: Path) -> str:
    """sections_0001045810_15_000097.json → 0001045810-15-000097"""
    name = path.stem.replace("sections_", "")
    parts = name.split("_")
    if len(parts) == 3:
        return "-".join(parts)
    return name


def _form_type_from_meta(accession: str, raw_dir: Path) -> str | None:
    """Read form_type from the cached <accession>_meta.json if present."""
    meta_path = raw_dir / "filings" / f"{accession}_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("form_type") or meta.get("form")
    except (json.JSONDecodeError, OSError):
        return None


def _filing_date_from_meta(accession: str, raw_dir: Path) -> str | None:
    meta_path = raw_dir / "filings" / f"{accession}_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("filing_date") or meta.get("filingDate")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> int:
    cfg = EngineConfig()
    raw_dir = cfg.raw_dir
    interim_dir = cfg.interim_dir
    parser = FilingTextParser(cfg)
    fetcher = EdgarFetcher(cfg)

    # Inventory existing section JSONs
    section_files = sorted(interim_dir.glob(f"sections_{cfg.cik}_*.json"))
    if not section_files:
        logger.warning("No cached section JSONs found in %s", interim_dir)
        return 0

    logger.info("Found %d cached section JSONs", len(section_files))

    pre_tiers: Counter[str] = Counter()
    post_tiers: Counter[str] = Counter()
    refetched: list[str] = []
    failed: list[str] = []
    parsed_ok: list[str] = []

    for sf in section_files:
        accession = _accession_from_section_json(sf)
        try:
            existing = json.loads(sf.read_text())
            for entry in existing:
                tier = entry.get("extraction_tier")
                if tier:
                    pre_tiers[tier] += 1
                else:
                    pre_tiers[entry.get("parse_status", "unknown")] += 1
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", sf, e)
            continue

        # Locate HTML
        html_path = raw_dir / "filings" / f"{accession}.html"
        html_size = html_path.stat().st_size if html_path.exists() else 0

        # Re-fetch if HTML is missing or small (likely an index page).
        # Delete small HTMLs first so the EdgarFetcher cache check fails
        # and a real network fetch is triggered.
        if not html_path.exists() or html_size < SMALL_HTML_THRESHOLD:
            if html_path.exists():
                logger.info(
                    "Deleting small cached HTML for %s (size: %d bytes) "
                    "to force fresh fetch",
                    accession,
                    html_size,
                )
                try:
                    html_path.unlink()
                    # Also clear meta so it gets re-fetched cleanly
                    meta_path = raw_dir / "filings" / f"{accession}_meta.json"
                    if meta_path.exists():
                        meta_path.unlink()
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", html_path, e)

            logger.info("Re-fetching %s", accession)
            try:
                html_text = fetcher.fetch_filing_document(
                    accession=accession, cik=cfg.cik
                )
                if not html_text or len(html_text) < SMALL_HTML_THRESHOLD:
                    logger.warning(
                        "Re-fetch for %s still small (%d bytes); marking as failed",
                        accession,
                        len(html_text) if html_text else 0,
                    )
                    failed.append(accession)
                    continue
                refetched.append(accession)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Re-fetch failed for %s: %s", accession, exc)
                failed.append(accession)
                continue
        else:
            try:
                html_text = html_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("Failed to read %s: %s", html_path, e)
                failed.append(accession)
                continue

        # Determine form type and filing date
        form_type = _form_type_from_meta(accession, raw_dir)
        if form_type is None:
            # Inspect existing JSON for form_type
            try:
                existing = json.loads(sf.read_text())
                if existing and "form_type" in existing[0]:
                    form_type = existing[0]["form_type"]
            except Exception:  # noqa: BLE001
                pass
        if form_type is None:
            logger.warning("No form_type for %s; defaulting to 10-K", accession)
            form_type = "10-K"

        filing_date = _filing_date_from_meta(accession, raw_dir)
        if filing_date is None:
            try:
                existing = json.loads(sf.read_text())
                if existing and "filing_date" in existing[0]:
                    filing_date = existing[0]["filing_date"]
            except Exception:  # noqa: BLE001
                pass
        if filing_date is None:
            filing_date = "2000-01-01"  # placeholder

        # Re-parse
        try:
            records = parser.parse_filing(
                html=html_text,
                accession=accession,
                form_type=form_type,
                filing_date=filing_date,
                source_available_date=filing_date,
            )
            parsed_ok.append(accession)
            for r in records:
                tier = getattr(r, "extraction_tier", None) or r.parse_status
                post_tiers[tier] += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Parse failed for %s: %s", accession, exc)
            failed.append(accession)

    # Summary
    logger.info("=" * 60)
    logger.info(
        "Re-parsed %d / %d filings (refetched %d, failed %d)",
        len(parsed_ok),
        len(section_files),
        len(refetched),
        len(failed),
    )
    logger.info("Pre-tiers: %s", dict(pre_tiers))
    logger.info("Post-tiers: %s", dict(post_tiers))

    if refetched:
        logger.info("Refetched: %s", ", ".join(refetched))
    if failed:
        logger.warning("Failed: %s", ", ".join(failed))

    # Coverage calculation: success + fallback + proxy = "active"
    active_post = (
        post_tiers.get("full_extraction", 0)
        + post_tiers.get("partial_extraction", 0)
        + post_tiers.get("mda_proxy_fallback", 0)
    )
    total_post = sum(post_tiers.values())
    coverage_pct = 100.0 * active_post / total_post if total_post else 0.0
    logger.info(
        "Active extractions: %d / %d (%.1f%%)",
        active_post,
        total_post,
        coverage_pct,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
