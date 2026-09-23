"""
Per-session application state for the Spectral Agent.

LangGraph's ``MemorySaver`` handles **chat-level** persistence (the message
list kept across turns).  ``SessionStore`` adds an **application-level** layer
that tracks *what work has already been done* so the agent (and tools) can
avoid redundant downloads, crops, and computations.

Design
------
* One ``SessionState`` per ``session_id``.
* ``SessionStore`` is an in-memory dict of states (cheap, no DB).
* The store is injected into the agent graph; tools can query it via the
  ``RunnableConfig`` to check what's cached.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SessionState:
    """
    Application-level state for one user / conversation session.

    This is *not* the LangGraph message state — it lives alongside it and
    stores artefact metadata so that tools can make smart decisions (e.g.
    skip downloading a scene that was already fetched).

    Attributes
    ----------
    session_id : str
        Unique identifier for the session (doubles as ``thread_id``).
    downloaded_scenes : dict[str, dict]
        ``{scene_id: {"satellite": ..., "tar_path": ..., "download_time": ...}}``
    computed_indices : dict[tuple[str, str], str]
        ``{(scene_id, index_name): output_raster_path}``
    generated_maps : dict[str, dict]
        ``{raster_path: {"static_png": ..., "interactive_html": ...}}``
    active_bbox : dict | None
        Last bounding-box used (WGS-84).  ``{"west", "south", "east", "north"}``
    active_date_range : dict | None
        ``{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}``
    metadata : dict
        Free-form bag for anything the agent wants to remember.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Artefact tracking
    downloaded_scenes: dict[str, dict[str, Any]] = field(default_factory=dict)
    computed_indices: dict[str, str] = field(default_factory=dict)
    generated_maps: dict[str, dict[str, str]] = field(default_factory=dict)

    # Context carried across turns
    active_bbox: dict[str, float] | None = None
    active_date_range: dict[str, str] | None = None

    # Free-form
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Helpers ──────────────────────────────────────────────────────────

    def register_download(
        self,
        scene_id: str,
        satellite: str,
        local_path: str | Path,
        **extra: Any,
    ) -> None:
        """Record a newly downloaded scene."""
        self.downloaded_scenes[scene_id] = {
            "satellite": satellite,
            "local_path": str(local_path),
            **extra,
        }
        logger.debug("Session %s: registered download %s", self.session_id, scene_id)

    def has_scene(self, scene_id: str) -> bool:
        """Check whether *scene_id* was already downloaded this session."""
        return scene_id in self.downloaded_scenes

    def register_index(
        self,
        scene_id: str,
        index_name: str,
        output_path: str | Path,
    ) -> None:
        """Record a computed index raster."""
        key = f"{scene_id}::{index_name}"
        self.computed_indices[key] = str(output_path)
        logger.debug("Session %s: registered index %s", self.session_id, key)

    def has_index(self, scene_id: str, index_name: str) -> bool:
        """Check if an index has already been computed for a scene."""
        return f"{scene_id}::{index_name}" in self.computed_indices

    def get_index_path(self, scene_id: str, index_name: str) -> str | None:
        """Return the output path for a previously-computed index, or None."""
        return self.computed_indices.get(f"{scene_id}::{index_name}")

    def register_map(
        self,
        raster_path: str,
        outputs: dict[str, str],
    ) -> None:
        """Record generated map artefacts."""
        self.generated_maps[raster_path] = outputs

    def set_bbox(self, bbox: dict[str, float]) -> None:
        """Update the active bounding box."""
        self.active_bbox = bbox
        logger.debug("Session %s: bbox updated → %s", self.session_id, bbox)

    def set_date_range(self, start: str, end: str) -> None:
        """Update the active date range."""
        self.active_date_range = {"start": start, "end": end}

    def summary(self) -> dict[str, Any]:
        """Serialisable summary of this session's state."""
        return {
            "session_id": self.session_id,
            "downloaded_scenes": list(self.downloaded_scenes.keys()),
            "computed_indices": list(self.computed_indices.keys()),
            "generated_maps": len(self.generated_maps),
            "active_bbox": self.active_bbox,
            "active_date_range": self.active_date_range,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Session Store (in-memory registry of sessions)
# ─────────────────────────────────────────────────────────────────────────────


class SessionStore:
    """
    Thread-safe in-memory store that maps ``session_id`` → ``SessionState``.

    Usage::

        store = SessionStore()
        state = store.get_or_create("abc123")
        state.register_download("LC09_...", "landsat", "/data/raw/LC09_...")
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    # ── CRUD ─────────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str | None = None) -> SessionState:
        """
        Return existing session or create a fresh one.

        Parameters
        ----------
        session_id : str | None
            If *None*, a new random id is generated.
        """
        if session_id is None:
            state = SessionState()
            self._sessions[state.session_id] = state
            logger.info("SessionStore: created new session %s", state.session_id)
            return state

        if session_id not in self._sessions:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
            logger.info("SessionStore: created session %s", session_id)

        return self._sessions[session_id]

    def get(self, session_id: str) -> SessionState | None:
        """Return session if it exists, else ``None``."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Remove a session.  Returns ``True`` if it existed."""
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        """Return all active session ids."""
        return list(self._sessions.keys())

    def clear(self) -> None:
        """Drop every session."""
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions
