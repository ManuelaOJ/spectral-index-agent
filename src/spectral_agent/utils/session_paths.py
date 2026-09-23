"""
Session-scoped path helpers.

Every satellite pipeline run is isolated under its session directory::

    data/raw/<session_id>/landsat/...
    data/processed/<session_id>/landsat/...

The ``session_id`` is the LangGraph ``thread_id`` pulled from the
``RunnableConfig`` that the framework injects into every tool call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectral_agent.config import get_settings

# Fallback when no session is found (e.g. tests, notebooks)
_DEFAULT_SESSION = "default"


def get_session_id(config: dict[str, Any] | None) -> str:
    """Extract ``thread_id`` from a LangGraph ``RunnableConfig``.

    Falls back to ``"default"`` when *config* is ``None`` or has no
    ``thread_id``.
    """
    if config is None:
        return _DEFAULT_SESSION
    configurable = config.get("configurable", {})
    return configurable.get("thread_id") or _DEFAULT_SESSION


def session_raw_dir(session_id: str, satellite: str) -> Path:
    """Return ``data/raw/<session_id>/<satellite>/``, creating it if needed."""
    settings = get_settings()
    path = settings.raw_data_dir / session_id / satellite
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_processed_dir(session_id: str, satellite: str) -> Path:
    """Return ``data/processed/<session_id>/<satellite>/``, creating it if needed."""
    settings = get_settings()
    path = settings.processed_data_dir / session_id / satellite
    path.mkdir(parents=True, exist_ok=True)
    return path
