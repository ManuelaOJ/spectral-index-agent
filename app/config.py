"""
Streamlit session configuration and state initialisation helpers.

This module centralises every ``st.session_state`` key used by the app so
that components can share state safely without key collisions.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE = "🛰️ AI Spectral Index Agent"
APP_ICON = "🛰️"
LAYOUT = "wide"

# Available indices (duplicated here so the sidebar doesn't need a heavy import)
AVAILABLE_INDICES: list[dict[str, str]] = [
    {"key": "NDVI", "label": "NDVI · Índice de vegetación"},
    {"key": "EVI", "label": "EVI · Índice de vegetación mejorado"},
    {"key": "SAVI", "label": "SAVI · Vegetación ajustado al suelo"},
    {"key": "NDWI", "label": "NDWI · Índice de agua"},
    {"key": "NBR", "label": "NBR · Índice de área quemada"},
    {"key": "NDBI", "label": "NDBI · Índice de área construida"},
]

# Supported geometry uploads
# SHP requires companion files (.shx, .dbf, .prj, .cpg) uploaded together
GEOMETRY_EXTENSIONS = ["kml", "shp", "shx", "dbf", "prj", "cpg", "geojson", "json"]


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    # ── API keys (never persisted to disk) ─────────────────────────────
    "usgs_username": "",
    "usgs_token": "",
    "copernicus_client_id": "",
    "copernicus_client_secret": "",
    "openai_api_key": "",
    "anthropic_api_key": "",
    "google_api_key": "",
    # ── LLM settings ───────────────────────────────────────────────────
    "llm_provider": "openai",
    "llm_model": "gpt-4o",
    # ── Chat ───────────────────────────────────────────────────────────
    "messages": [],  # list[dict] — {"role": "user"|"assistant", "content": str}
    "thread_id": None,  # filled on first user message
    "agent": None,  # CompiledGraph — built lazily
    # ── Selected indices ───────────────────────────────────────────────
    "selected_indices": [],
    # ── Map artefacts ──────────────────────────────────────────────────
    "last_static_png": None,  # Path | None
    "last_interactive_html": None,  # Path | None
    "last_index_raster": None,  # Path | None
    # ── Geometry upload ────────────────────────────────────────────────
    "uploaded_geometry_path": None,
    # ── Processing flags ───────────────────────────────────────────────
    "is_processing": False,
}


def _load_dev_keys() -> dict[str, str]:
    """When running in *development* mode, pre-fill API keys from ``.env``.

    Returns an empty dict in production so that keys must be entered
    manually via the sidebar.
    """
    try:
        from spectral_agent.config.settings import get_settings

        cfg = get_settings()
        if cfg.environment != "development":
            return {}

        return {
            "usgs_username": cfg.usgs_username,
            "usgs_token": cfg.usgs_token,
            "copernicus_client_id": cfg.copernicus_client_id,
            "copernicus_client_secret": cfg.copernicus_client_secret,
            "openai_api_key": cfg.openai_api_key,
            "anthropic_api_key": cfg.anthropic_api_key,
            "google_api_key": cfg.google_api_key,
            "llm_provider": cfg.default_llm_provider,
            "llm_model": cfg.default_llm_model,
        }
    except Exception:
        return {}


def init_session_state() -> None:
    """Populate ``st.session_state`` with sensible defaults (idempotent).

    In **development** mode the API keys are pre-filled from ``.env`` so
    you don't have to paste them into the sidebar on every reload.
    """
    import uuid

    dev_keys = _load_dev_keys()

    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = dev_keys.get(key, default)

    # Ensure a thread_id is always available for the status bar.
    if not st.session_state.get("thread_id"):
        st.session_state["thread_id"] = uuid.uuid4().hex[:12]


def reset_chat() -> None:
    """Clear chat history and map artefacts.  Keeps API keys."""
    import uuid

    st.session_state["messages"] = []
    st.session_state["thread_id"] = uuid.uuid4().hex[:12]
    st.session_state["agent"] = None
    st.session_state["last_static_png"] = None
    st.session_state["last_interactive_html"] = None
    st.session_state["last_index_raster"] = None
    st.session_state["is_processing"] = False
