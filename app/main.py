"""
AI Spectral Index Agent — Streamlit entry point.

Launch with::

    streamlit run app/main.py

The page follows a two-column layout:
  * **Left column** — Map display area + download buttons.
  * **Right column** — Chat interface with streaming agent responses.

The sidebar (rendered by ``components/sidebar.py``) holds credentials,
LLM settings, the geometry uploader, and live pipeline metrics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``from app.…`` works
# regardless of the directory Streamlit is launched from.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# El paquete vive en src/ (layout "src").  Se agrega tambien al path para
# que la app arranque desde un clon limpio sin `pip install -e .`.
_SRC_ROOT = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# ── Fix: PostgreSQL 18 sets SSL_CERT_FILE to a non-existent path ────────
# Override with certifi's trusted CA bundle so that ``requests`` (and
# every other TLS client) can verify certificates correctly.
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import logging

# ── File logging — captures all agent/tool errors to data/logs/agent.log ──
_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_LOG_DIR / "agent.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(_file_handler)
logging.getLogger().setLevel(logging.DEBUG)
# Quiet noisy libraries
for _lib in ("httpx", "httpcore", "urllib3", "openai", "anthropic", "markdown_it"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

import streamlit as st

# ── Streamlit page config (MUST be the first st.* call) ─────────────────
st.set_page_config(
    page_title="Spectral Index Agent",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── App imports (after set_page_config) ──────────────────────────────────
from app.components.chat import render_chat, render_chat_input
from app.components.map_display import render_map_display
from app.components.sidebar import render_sidebar
from app.config import init_session_state
from app.styles.theme import CUSTOM_CSS

# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>Spectral Index Agent</h1>
        <p>Busque, descargue y analice imágenes satelitales con lenguaje natural.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Status bar — compact inline context
# ─────────────────────────────────────────────────────────────────────────────

_provider = st.session_state.get("llm_provider", "openai")
_model = st.session_state.get("llm_model", "gpt-4o")
_thread = st.session_state.get("thread_id")

_status_parts = [
    f'<span class="status-item"><span class="label">Modelo</span>'
    f' <span class="value">{_provider}/{_model}</span></span>',
    '<span class="divider"></span>',
]

_geo = st.session_state.get("uploaded_geometry_path")
if _geo:
    _geo_name = Path(_geo).stem
    _status_parts.append(
        f'<span class="status-item"><span class="label">Área</span>'
        f' <span class="value">{_geo_name}</span></span>'
    )
    _status_parts.append('<span class="divider"></span>')

_status_parts.append(
    f'<span class="status-item"><span class="label">Sesión</span>'
    f' <span class="value">{_thread or "—"}</span></span>'
)

st.markdown(
    '<div class="status-bar">' + "".join(_status_parts) + "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main content — two-column layout  (map : chat = 5 : 6)
# ─────────────────────────────────────────────────────────────────────────────

map_col, chat_col = st.columns([5, 6], gap="medium")

with map_col:
    render_map_display()

with chat_col:
    render_chat()

# ── Chat input — page-level so Streamlit pins it to viewport bottom ──────
render_chat_input()
