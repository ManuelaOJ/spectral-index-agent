"""
Unit tests for Step 5 — Conversational Memory & Agent Upgrade.

Tests cover:
  1. SessionState — artefact registration, lookups, summary
  2. SessionStore — CRUD, isolation between sessions
  3. agent_graph — graph construction, MemorySaver wiring
  4. spectral_agent — backwards-compatible create_spectral_agent with memory
"""

from __future__ import annotations

import pytest

from spectral_agent.memory.session_store import SessionState, SessionStore

# ═════════════════════════════════════════════════════════════════════════════
# SessionState
# ═════════════════════════════════════════════════════════════════════════════


class TestSessionState:
    def test_default_session_id_generated(self):
        s = SessionState()
        assert len(s.session_id) == 12
        # Two instances should have distinct ids
        assert SessionState().session_id != s.session_id

    def test_explicit_session_id(self):
        s = SessionState(session_id="test-123")
        assert s.session_id == "test-123"

    # ── Downloads ────────────────────────────────────────────────────────

    def test_register_and_has_scene(self):
        s = SessionState()
        assert s.has_scene("LC09_SCENE") is False
        s.register_download("LC09_SCENE", "landsat", "/data/raw/LC09.tar")
        assert s.has_scene("LC09_SCENE") is True
        assert s.downloaded_scenes["LC09_SCENE"]["satellite"] == "landsat"
        assert s.downloaded_scenes["LC09_SCENE"]["local_path"] == "/data/raw/LC09.tar"

    def test_register_download_extra_kwargs(self):
        s = SessionState()
        s.register_download("S2_SCENE", "sentinel", "/data/sentinel", cloud=5.2)
        assert s.downloaded_scenes["S2_SCENE"]["cloud"] == 5.2

    # ── Indices ──────────────────────────────────────────────────────────

    def test_register_and_has_index(self):
        s = SessionState()
        assert s.has_index("LC09", "NDVI") is False
        s.register_index("LC09", "NDVI", "/data/processed/NDVI.tif")
        assert s.has_index("LC09", "NDVI") is True
        assert s.get_index_path("LC09", "NDVI") == "/data/processed/NDVI.tif"

    def test_get_index_path_missing(self):
        s = SessionState()
        assert s.get_index_path("LC09", "EVI") is None

    # ── Maps ─────────────────────────────────────────────────────────────

    def test_register_map(self):
        s = SessionState()
        s.register_map("/raster.tif", {"static_png": "/map.png", "interactive_html": "/map.html"})
        assert "/raster.tif" in s.generated_maps
        assert s.generated_maps["/raster.tif"]["static_png"] == "/map.png"

    # ── Bbox / date ──────────────────────────────────────────────────────

    def test_set_bbox(self):
        s = SessionState()
        assert s.active_bbox is None
        s.set_bbox({"west": -75, "south": 6, "east": -74, "north": 7})
        assert s.active_bbox["west"] == -75

    def test_set_date_range(self):
        s = SessionState()
        s.set_date_range("2024-01-01", "2024-12-31")
        assert s.active_date_range == {"start": "2024-01-01", "end": "2024-12-31"}

    # ── Summary ──────────────────────────────────────────────────────────

    def test_summary_empty(self):
        s = SessionState(session_id="s1")
        summary = s.summary()
        assert summary["session_id"] == "s1"
        assert summary["downloaded_scenes"] == []
        assert summary["computed_indices"] == []
        assert summary["generated_maps"] == 0
        assert summary["active_bbox"] is None

    def test_summary_populated(self):
        s = SessionState(session_id="s2")
        s.register_download("LC09", "landsat", "/p")
        s.register_index("LC09", "NDVI", "/idx")
        s.register_map("/idx", {"static_png": "/m.png"})
        summary = s.summary()
        assert summary["downloaded_scenes"] == ["LC09"]
        assert "LC09::NDVI" in summary["computed_indices"]
        assert summary["generated_maps"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# SessionStore
# ═════════════════════════════════════════════════════════════════════════════


class TestSessionStore:
    def test_get_or_create_new(self):
        store = SessionStore()
        state = store.get_or_create("sess-1")
        assert state.session_id == "sess-1"
        assert len(store) == 1

    def test_get_or_create_returns_same(self):
        store = SessionStore()
        s1 = store.get_or_create("x")
        s2 = store.get_or_create("x")
        assert s1 is s2

    def test_get_or_create_auto_id(self):
        store = SessionStore()
        state = store.get_or_create()
        assert len(state.session_id) == 12
        assert state.session_id in store

    def test_get_existing(self):
        store = SessionStore()
        store.get_or_create("abc")
        assert store.get("abc") is not None

    def test_get_missing(self):
        store = SessionStore()
        assert store.get("nope") is None

    def test_delete(self):
        store = SessionStore()
        store.get_or_create("del-me")
        assert store.delete("del-me") is True
        assert store.get("del-me") is None
        assert store.delete("del-me") is False

    def test_list_sessions(self):
        store = SessionStore()
        store.get_or_create("a")
        store.get_or_create("b")
        assert sorted(store.list_sessions()) == ["a", "b"]

    def test_clear(self):
        store = SessionStore()
        store.get_or_create("a")
        store.get_or_create("b")
        store.clear()
        assert len(store) == 0

    def test_contains(self):
        store = SessionStore()
        store.get_or_create("in")
        assert "in" in store
        assert "out" not in store

    def test_isolation(self):
        """Changes in one session must not leak to another."""
        store = SessionStore()
        s1 = store.get_or_create("s1")
        s2 = store.get_or_create("s2")
        s1.register_download("SCENE_A", "landsat", "/a")
        assert s2.has_scene("SCENE_A") is False


# ═════════════════════════════════════════════════════════════════════════════
# agent_graph — build_agent & make_thread_config
# ═════════════════════════════════════════════════════════════════════════════


class TestAgentGraph:
    @pytest.fixture()
    def _fake_settings(self, monkeypatch):
        """Patch get_settings so we don't need real API keys."""
        from spectral_agent.config.settings import Settings

        fake = Settings(
            openai_api_key="sk-test-fake-key",
            data_dir="data",
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            cache_dir="data/cache",
        )
        monkeypatch.setattr(
            "spectral_agent.graphs.agent_graph.get_settings",
            lambda: fake,
        )

    def test_build_agent_returns_compiled_graph(self, _fake_settings):
        from spectral_agent.graphs.agent_graph import AgentDeps, build_agent

        deps = AgentDeps(provider="openai", model="gpt-4o")
        agent = build_agent(deps)

        from langgraph.graph.graph import CompiledGraph

        assert isinstance(agent, CompiledGraph)

    def test_build_agent_default_checkpointer(self, _fake_settings):
        from spectral_agent.graphs.agent_graph import AgentDeps, build_agent

        deps = AgentDeps(provider="openai", model="gpt-4o")
        agent = build_agent(deps)
        # The compiled graph should have a checkpointer attached
        assert agent.checkpointer is not None

    def test_build_agent_custom_checkpointer(self, _fake_settings):
        from langgraph.checkpoint.memory import MemorySaver

        from spectral_agent.graphs.agent_graph import AgentDeps, build_agent

        custom_cp = MemorySaver()
        deps = AgentDeps(provider="openai", model="gpt-4o", checkpointer=custom_cp)
        agent = build_agent(deps)
        assert agent.checkpointer is custom_cp

    def test_make_thread_config(self):
        from spectral_agent.graphs.agent_graph import make_thread_config

        cfg = make_thread_config("abc-123")
        assert cfg == {"configurable": {"thread_id": "abc-123"}}

    def test_agent_deps_from_settings(self, _fake_settings):
        from spectral_agent.graphs.agent_graph import AgentDeps

        deps = AgentDeps.from_settings(temperature=0.5)
        assert deps.temperature == 0.5
        assert deps.provider == "openai"  # default from settings


# ═════════════════════════════════════════════════════════════════════════════
# spectral_agent — backwards-compatible upgrades
# ═════════════════════════════════════════════════════════════════════════════


class TestSpectralAgentUpgrade:
    @pytest.fixture()
    def _fake_settings(self, monkeypatch):
        from spectral_agent.config.settings import Settings

        fake = Settings(
            openai_api_key="sk-test-fake-key",
            data_dir="data",
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            cache_dir="data/cache",
        )
        monkeypatch.setattr(
            "spectral_agent.agents.spectral_agent.get_settings",
            lambda: fake,
        )

    def test_create_agent_has_checkpointer_by_default(self, _fake_settings):
        from spectral_agent.agents.spectral_agent import (
            SpectralAgentConfig,
            create_spectral_agent,
        )

        cfg = SpectralAgentConfig(provider="openai", model="gpt-4o")
        agent = create_spectral_agent(cfg)
        assert agent.checkpointer is not None

    def test_create_agent_no_memory(self, _fake_settings):
        from spectral_agent.agents.spectral_agent import (
            SpectralAgentConfig,
            create_spectral_agent,
        )

        cfg = SpectralAgentConfig(provider="openai", model="gpt-4o", use_memory=False)
        agent = create_spectral_agent(cfg)
        assert agent.checkpointer is None

    def test_create_agent_custom_checkpointer(self, _fake_settings):
        from langgraph.checkpoint.memory import MemorySaver

        from spectral_agent.agents.spectral_agent import (
            SpectralAgentConfig,
            create_spectral_agent,
        )

        cp = MemorySaver()
        cfg = SpectralAgentConfig(provider="openai", model="gpt-4o")
        agent = create_spectral_agent(cfg, checkpointer=cp)
        assert agent.checkpointer is cp

    def test_config_use_memory_default_true(self):
        from spectral_agent.agents.spectral_agent import SpectralAgentConfig

        assert SpectralAgentConfig().use_memory is True
