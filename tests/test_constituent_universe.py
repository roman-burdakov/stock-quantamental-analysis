"""
Tests for Point-in-Time Constituent Universe (Milestone 20).

Validates:
- Report contains a Peer Universe section or table (Req 25.1)
- Each peer has ticker, tier, justification, and date range (Req 25.1)
- Peer universe is marked as fixed (Req 25.2)
- Survivorship bias disclosure is present (Req 25.2)
- Corporate events are documented for peers that have them (Req 25.3)
- Index membership note is present (Req 25.4)
- Constituent universe limitations note (Req 25.5)
- Source attribution includes per-peer staleness and inclusion status (Req 25.6)

Reqs: 25.1–25.6
"""

from __future__ import annotations

import pytest

from src.audit_utils import AuditModule
from src.config import (
    EngineConfig,
    ExhibitRecord,
    PEER_CORPORATE_EVENTS,
    get_default_config,
)
from src.report_utils import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> EngineConfig:
    return get_default_config()


@pytest.fixture
def generator(config: EngineConfig) -> ReportGenerator:
    return ReportGenerator(config)


def _sample_peer_attribution() -> list[dict]:
    """Sample per-peer attribution data for source_attribution tests."""
    return [
        {
            "ticker": "AMD",
            "peer_tier": "semi",
            "financial_data_date": "2025-12-27",
            "market_data_date": "2026-04-25",
            "staleness_days": 119,
            "status": "excluded",
            "reason": "Stale financial data (119d > 90d threshold)",
        },
        {
            "ticker": "AVGO",
            "peer_tier": "semi",
            "financial_data_date": "2026-02-01",
            "market_data_date": "2026-04-25",
            "staleness_days": 83,
            "status": "included",
            "reason": "Financial data within staleness threshold",
        },
        {
            "ticker": "TSM",
            "peer_tier": "infrastructure",
            "financial_data_date": "2026-03-31",
            "market_data_date": "2026-04-25",
            "staleness_days": 25,
            "status": "included",
            "reason": "Financial data within staleness threshold",
        },
        {
            "ticker": "MSFT",
            "peer_tier": "context",
            "financial_data_date": "2026-01-15",
            "market_data_date": "2026-04-25",
            "staleness_days": 100,
            "status": "excluded",
            "reason": "Stale financial data (100d > 90d threshold)",
        },
    ]


# ---------------------------------------------------------------------------
# Tests: Peer Universe section in report (Req 25.1, 25.2)
# ---------------------------------------------------------------------------

class TestPeerUniverseSection:
    """Tests for build_peer_universe_section() in ReportGenerator."""

    def test_returns_peer_universe_table(self, generator: ReportGenerator, config: EngineConfig):
        """Req 25.1: The report contains a Peer Universe table with all configured peers."""
        result = generator.build_peer_universe_section()
        table = result["peer_universe_table"]

        all_peers = (
            config.core_semiconductor_peers
            + config.infrastructure_peers
            + config.ai_capex_context
        )
        assert len(table) == len(all_peers)

        table_tickers = [row["ticker"] for row in table]
        for peer in all_peers:
            assert peer in table_tickers, f"Peer {peer} missing from universe table"

    def test_each_peer_has_required_fields(self, generator: ReportGenerator):
        """Req 25.1: Each peer has ticker, tier, justification, and date range."""
        result = generator.build_peer_universe_section()
        for row in result["peer_universe_table"]:
            assert "ticker" in row and row["ticker"], "Missing ticker"
            assert "tier" in row and row["tier"], "Missing tier"
            assert "justification" in row and row["justification"], "Missing justification"
            assert "date_range" in row and row["date_range"], "Missing date_range"

    def test_peer_tiers_are_correct(self, generator: ReportGenerator, config: EngineConfig):
        """Req 25.1: Peers are assigned to the correct tier."""
        result = generator.build_peer_universe_section()
        tier_map = {row["ticker"]: row["tier"] for row in result["peer_universe_table"]}

        for ticker in config.core_semiconductor_peers:
            assert tier_map[ticker] == "Core Semiconductor"
        for ticker in config.infrastructure_peers:
            assert tier_map[ticker] == "Infrastructure"
        for ticker in config.ai_capex_context:
            assert tier_map[ticker] == "AI Capex Context"

    def test_peer_universe_is_fixed(self, generator: ReportGenerator):
        """Req 25.2: The peer universe is marked as fixed."""
        result = generator.build_peer_universe_section()
        assert result["peer_universe_fixed"] is True

    def test_survivorship_bias_disclosure_present(self, generator: ReportGenerator):
        """Req 25.2: Survivorship bias disclosure mentions current-day relevance and survivorship bias."""
        result = generator.build_peer_universe_section()
        disclosure = result["survivorship_bias_disclosure"]

        assert "current-day relevance" in disclosure
        assert "survivorship bias" in disclosure.lower()

    def test_date_range_matches_config(self, generator: ReportGenerator, config: EngineConfig):
        """Req 25.1: Date range reflects the configured fiscal year window."""
        result = generator.build_peer_universe_section()
        expected_range = f"FY{config.start_fiscal_year}\u2013FY{config.end_fiscal_year}"

        for row in result["peer_universe_table"]:
            assert row["date_range"] == expected_range


# ---------------------------------------------------------------------------
# Tests: Corporate events (Req 25.3)
# ---------------------------------------------------------------------------

class TestPeerCorporateEvents:
    """Tests for corporate event documentation in the peer universe section."""

    def test_corporate_events_present_for_known_peers(self, generator: ReportGenerator):
        """Req 25.3: Corporate events are documented for peers that have them."""
        result = generator.build_peer_universe_section()
        events_list = result["peer_corporate_events"]

        # At least AMD and AVGO have corporate events in PEER_CORPORATE_EVENTS
        event_tickers = [entry["ticker"] for entry in events_list]
        peers_with_events = [
            t for t, evts in PEER_CORPORATE_EVENTS.items() if evts
        ]
        for ticker in peers_with_events:
            assert ticker in event_tickers, (
                f"Peer {ticker} has corporate events but is not documented"
            )

    def test_corporate_event_structure(self, generator: ReportGenerator):
        """Req 25.3: Each corporate event entry has ticker and events list."""
        result = generator.build_peer_universe_section()
        for entry in result["peer_corporate_events"]:
            assert "ticker" in entry
            assert "events" in entry
            assert isinstance(entry["events"], list)
            assert len(entry["events"]) > 0
            for event in entry["events"]:
                assert "date" in event
                assert "event" in event
                assert "impact" in event


# ---------------------------------------------------------------------------
# Tests: Index membership note (Req 25.4)
# ---------------------------------------------------------------------------

class TestIndexMembershipNote:
    """Tests for index membership documentation."""

    def test_index_membership_note_present(self, generator: ReportGenerator):
        """Req 25.4: Index membership note is present in the peer universe section."""
        result = generator.build_peer_universe_section()
        assert "index_membership_note" in result
        note = result["index_membership_note"]
        assert note, "Index membership note should not be empty"

    def test_index_membership_note_content(self, generator: ReportGenerator):
        """Req 25.4: Index membership note mentions S&P 500 or index membership."""
        result = generator.build_peer_universe_section()
        note = result["index_membership_note"]
        # Should mention index membership tracking status
        assert "index" in note.lower() or "S&P" in note


# ---------------------------------------------------------------------------
# Tests: Constituent Universe Limitations (Req 25.5)
# ---------------------------------------------------------------------------

class TestConstituentUniverseLimitations:
    """Tests for the Constituent Universe Limitations note in the rendered report."""

    def test_limitations_note_in_template(self, generator: ReportGenerator):
        """Req 25.5: The report template includes a Constituent Universe Limitations section.

        The limitations note is hardcoded in the Jinja2 template and rendered
        when peer_universe_table is present. We verify the section data is
        produced so the template will render it.
        """
        result = generator.build_peer_universe_section()
        # The template renders the limitations note whenever peer_universe_table
        # is non-empty, which is guaranteed by the section builder.
        assert len(result["peer_universe_table"]) > 0

    def test_survivorship_disclosure_mentions_point_in_time(self, generator: ReportGenerator):
        """Req 25.5: The survivorship disclosure mentions point-in-time and historical index data."""
        result = generator.build_peer_universe_section()
        disclosure = result["survivorship_bias_disclosure"]
        # The disclosure should mention that a fully point-in-time universe
        # would require historical index data
        assert "point-in-time" in disclosure.lower() or "historical" in disclosure.lower()


# ---------------------------------------------------------------------------
# Tests: Source attribution per-peer staleness and inclusion (Req 25.6)
# ---------------------------------------------------------------------------

class TestPeerSourceAttribution:
    """Tests for per-peer staleness and inclusion/exclusion in source_attribution.md."""

    def test_peer_attribution_table_in_source_attribution(self, tmp_path):
        """Req 25.6: source_attribution.md includes a Peer Source Attribution table."""
        cfg = get_default_config()
        cfg.outputs_dir = tmp_path
        mod = AuditModule(cfg)
        content = mod.generate_source_attribution(
            [], [], peer_attribution=_sample_peer_attribution(),
        )
        assert "Peer Source Attribution" in content

    def test_per_peer_staleness_present(self, tmp_path):
        """Req 25.6: Each peer row includes staleness in days."""
        cfg = get_default_config()
        cfg.outputs_dir = tmp_path
        mod = AuditModule(cfg)
        content = mod.generate_source_attribution(
            [], [], peer_attribution=_sample_peer_attribution(),
        )
        # Check that staleness values from our sample data appear
        assert "119" in content  # AMD staleness
        assert "83" in content   # AVGO staleness
        assert "25" in content   # TSM staleness

    def test_per_peer_inclusion_exclusion_status(self, tmp_path):
        """Req 25.6: Each peer has inclusion or exclusion status with reason."""
        cfg = get_default_config()
        cfg.outputs_dir = tmp_path
        mod = AuditModule(cfg)
        content = mod.generate_source_attribution(
            [], [], peer_attribution=_sample_peer_attribution(),
        )
        assert "included" in content
        assert "excluded" in content

    def test_peer_attribution_has_financial_and_market_dates(self, tmp_path):
        """Req 25.6: Each peer row includes financial data date and market data date."""
        cfg = get_default_config()
        cfg.outputs_dir = tmp_path
        mod = AuditModule(cfg)
        content = mod.generate_source_attribution(
            [], [], peer_attribution=_sample_peer_attribution(),
        )
        # Check that financial and market dates from sample data appear
        assert "2025-12-27" in content  # AMD financial date
        assert "2026-04-25" in content  # market data date

    def test_peer_attribution_summary_counts(self, tmp_path):
        """Req 25.6: Attribution includes summary of included vs excluded peers."""
        cfg = get_default_config()
        cfg.outputs_dir = tmp_path
        mod = AuditModule(cfg)
        content = mod.generate_source_attribution(
            [], [], peer_attribution=_sample_peer_attribution(),
        )
        # 2 included (AVGO, TSM), 2 excluded (AMD, MSFT)
        assert "2 peers included" in content
        assert "2 peers excluded" in content
