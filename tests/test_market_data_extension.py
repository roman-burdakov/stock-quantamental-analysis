"""
Tests for v2 market data extension — Task 1.1 (Req 4.1).

Verifies that:
- ``EngineConfig.index_tickers`` defaults to ``['^SOX', '^GSPC']``.
- The pipeline's market-price fetch includes them in ``all_tickers``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import EngineConfig


PIPELINE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"


def test_engine_config_has_index_tickers_default() -> None:
    cfg = EngineConfig()
    assert hasattr(cfg, "index_tickers")
    assert "^SOX" in cfg.index_tickers
    assert "^GSPC" in cfg.index_tickers


def test_index_tickers_are_independent_per_instance() -> None:
    """Mutating one EngineConfig must not affect another (dataclass
    field default_factory contract)."""
    cfg1 = EngineConfig()
    cfg2 = EngineConfig()
    cfg1.index_tickers.append("^DJI")
    assert "^DJI" not in cfg2.index_tickers


def test_run_pipeline_passes_index_tickers_to_fetch_market_prices() -> None:
    """The pipeline call site must concatenate config.index_tickers into
    the all_tickers list. Static-source check — avoids importing the
    pipeline module which has heavy dependencies."""
    text = PIPELINE_PATH.read_text(encoding="utf-8")
    # Must reference config.index_tickers in the call assembly.
    assert "config.index_tickers" in text, (
        "scripts/run_pipeline.py must include config.index_tickers in the "
        "all_tickers list passed to fetch_market_prices()."
    )

    # And the inclusion must be in the all_tickers assembly block, not
    # somewhere unrelated.
    block_match = re.search(
        r"all_tickers\s*=\s*\(([^)]+)\)",
        text,
        re.DOTALL,
    )
    assert block_match is not None, "all_tickers assembly block not found"
    block_body = block_match.group(1)
    assert "config.index_tickers" in block_body, (
        "config.index_tickers must be inside the all_tickers tuple "
        f"(found in pipeline but outside block):\n{block_body}"
    )
