"""
Per-session debug logger for spectral index computations.

Captures and persists detailed debugging information for each chat
question / tool invocation, including:

* Satellite & sensor used
* Formula applied (symbolic + description)
* Bands selected (with Landsat band names or Sentinel-2 band codes)
* Value-range statistics
* Output file paths

Records are written to a JSONL file so they can be reviewed post-hoc
and are also surfaced in the chat reply during testing.

Usage
-----
::

    from spectral_agent.tracking.debug_log import get_debug_logger

    dbg = get_debug_logger()
    dbg.log_index_computation(
        session_id="abc123",
        satellite="landsat",
        sensor="Landsat 9",
        index_name="NDVI",
        formula="(NIR - RED) / (NIR + RED)",
        bands_used={"NIR": "SR_B5", "RED": "SR_B4"},
        output_path="/data/processed/landsat/scene/NDVI.tif",
        stats={"min": -0.12, "max": 0.87, "mean": 0.42},
    )
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Record schema
# ─────────────────────────────────────────────────────────────────────────────


class DebugRecord(BaseModel):
    """A single debug entry for an index computation."""

    timestamp: str = Field(description="ISO-8601 UTC timestamp")
    session_id: str = Field(default="", description="Chat thread / session id")
    satellite: str = Field(description="'landsat' or 'sentinel'")
    sensor: str = Field(default="", description="e.g. 'Landsat 9', 'Sentinel-2A'")
    index_name: str = Field(description="Spectral index, e.g. 'NDVI'")
    formula: str = Field(description="Human-readable formula string")
    bands_used: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of spectral role → band file/name",
    )
    evalscript: str = Field(
        default="",
        description="Sentinel Hub evalscript (Sentinel only)",
    )
    output_path: str = Field(default="", description="Path to output GeoTIFF")
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Value statistics (min, max, mean, nodata_pct, …)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata",
    )

    def to_json_line(self) -> str:
        return self.model_dump_json()


# ─────────────────────────────────────────────────────────────────────────────
# Formula descriptions (human-readable)
# ─────────────────────────────────────────────────────────────────────────────

FORMULA_DESCRIPTIONS: dict[str, str] = {
    "NDVI": "(NIR - RED) / (NIR + RED)",
    "EVI": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
    "SAVI": "((NIR - RED) / (NIR + RED + L)) * (1 + L), L=0.5",
    "NDWI": "(GREEN - NIR) / (GREEN + NIR)",
    "NBR": "(NIR - SWIR2) / (NIR + SWIR2)",
    "NDBI": "(SWIR1 - NIR) / (SWIR1 + NIR)",
}

SENTINEL_FORMULA_DESCRIPTIONS: dict[str, str] = {
    "NDVI": "(B08 - B04) / (B08 + B04)",
    "EVI": "2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)",
    "SAVI": "((B08 - B04) / (B08 + B04 + L)) * (1 + L), L=0.5",
    "NDWI": "(B03 - B08) / (B03 + B08)",
    "NBR": "(B08 - B12) / (B08 + B12)",
    "NDBI": "(B11 - B08) / (B11 + B08)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Debug logger singleton
# ─────────────────────────────────────────────────────────────────────────────


class DebugLogger:
    """Thread-safe debug logger that persists records to a JSONL file."""

    def __init__(self, log_path: Path | str | None = None) -> None:
        self._records: list[DebugRecord] = []
        self._lock = threading.Lock()
        self._active_session_id: str = ""

        if log_path is not None:
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._log_path = None

    # ── Session management ───────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        """Set the active session ID used for all subsequent log entries."""
        self._active_session_id = session_id

    @property
    def active_session_id(self) -> str:
        return self._active_session_id

    # ── Core logging method ──────────────────────────────────────────────

    def log_index_computation(
        self,
        *,
        session_id: str = "",
        satellite: str,
        sensor: str = "",
        index_name: str,
        formula: str = "",
        bands_used: dict[str, str] | None = None,
        evalscript: str = "",
        output_path: str = "",
        stats: dict[str, Any] | None = None,
        **extra: Any,
    ) -> DebugRecord:
        """Record a single index computation for debugging.

        Returns the created :class:`DebugRecord`.
        """
        name = index_name.upper()

        # Auto-resolve formula if not provided
        if not formula:
            if satellite == "sentinel":
                formula = SENTINEL_FORMULA_DESCRIPTIONS.get(name, "")
            else:
                formula = FORMULA_DESCRIPTIONS.get(name, "")

        # Use explicit session_id if provided, otherwise fall back to active
        resolved_session_id = session_id or self._active_session_id

        rec = DebugRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=resolved_session_id,
            satellite=satellite,
            sensor=sensor,
            index_name=name,
            formula=formula,
            bands_used=bands_used or {},
            evalscript=evalscript,
            output_path=output_path,
            stats=stats or {},
            extra=extra,
        )

        with self._lock:
            self._records.append(rec)

        # Persist to JSONL
        if self._log_path is not None:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(rec.to_json_line() + "\n")
            except OSError as exc:
                logger.warning("Failed to write debug log: %s", exc)

        logger.info(
            "DEBUG LOG | %s | %s | %s | formula=%s | bands=%s | path=%s",
            satellite,
            sensor,
            name,
            formula,
            bands_used or "-",
            output_path or "-",
        )

        return rec

    # ── Queries ──────────────────────────────────────────────────────────

    @property
    def records(self) -> list[DebugRecord]:
        with self._lock:
            return list(self._records)

    def last_n(self, n: int = 5) -> list[DebugRecord]:
        with self._lock:
            return list(self._records[-n:])

    def for_session(self, session_id: str) -> list[DebugRecord]:
        with self._lock:
            return [r for r in self._records if r.session_id == session_id]

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    def format_last_summary(self, n: int = 5) -> str:
        """Return a human-readable summary of the last *n* debug records."""
        recs = self.last_n(n)
        if not recs:
            return "No debug records yet."

        lines: list[str] = ["**Debug Info (last computations):**"]
        for r in recs:
            bands_str = (
                ", ".join(f"{role}={band}" for role, band in r.bands_used.items())
                if r.bands_used
                else "server-side"
            )
            stats_str = (
                ", ".join(f"{k}={v}" for k, v in r.stats.items()) if r.stats else "-"
            )

            lines.append(
                f"- **{r.index_name}** ({r.satellite}/{r.sensor})\n"
                f"  Formula: `{r.formula}`\n"
                f"  Bands: {bands_str}\n"
                f"  Stats: {stats_str}\n"
                f"  Output: `{r.output_path}`"
            )

        if self._log_path:
            lines.append(f"\nFull debug log: `{self._log_path}`")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: DebugLogger | None = None
_init_lock = threading.Lock()


def get_debug_logger(
    log_path: Path | str | None = "data/logs/debug_computations.jsonl",
) -> DebugLogger:
    """Return the module-level :class:`DebugLogger` singleton.

    On the first call the *log_path* is used; subsequent calls ignore it.
    """
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = DebugLogger(log_path=log_path)
    return _instance
