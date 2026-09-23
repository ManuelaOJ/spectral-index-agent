"""
Pipeline observability — per-step timing, success/failure, and metadata.

Every tool invocation in the agent pipeline is wrapped with
:func:`track_step` (or the :class:`StepTimer` context manager) so that
we capture wall-clock duration, outcome, satellite provider, spectral
index, and any error message.

Records are kept in-memory (thread-safe) and optionally flushed to a
JSONL file.  The companion Streamlit sidebar widget can surface a live
summary to the user.

Usage
-----
::

    from spectral_agent.tracking.metrics import get_metrics, track_step

    metrics = get_metrics()

    with track_step("scene_search", satellite="landsat") as step:
        results = heavy_search(...)
        step.set_metadata(scenes_found=len(results))
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step record
# ─────────────────────────────────────────────────────────────────────────────


class StepRecord(BaseModel):
    """Single pipeline step measurement."""

    timestamp: str = Field(description="ISO-8601 UTC timestamp (step start)")
    step_name: str = Field(description="Logical step name, e.g. scene_search")
    duration_s: float = Field(ge=0.0, description="Wall-clock seconds")
    status: str = Field(description="'success' or 'failure'")
    error: str | None = Field(default=None, description="Error message on failure")
    satellite: str | None = Field(default=None, description="landsat / sentinel / None")
    index_name: str | None = Field(default=None, description="Spectral index key")
    session_id: str = Field(default="", description="Thread / session identifier")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json_line(self) -> str:
        return self.model_dump_json()


# ─────────────────────────────────────────────────────────────────────────────
# Step timer context manager helper
# ─────────────────────────────────────────────────────────────────────────────


class StepTimer:
    """Accumulates timing & metadata for a single pipeline step.

    Normally used via :func:`track_step`; not instantiated directly.
    """

    def __init__(
        self,
        step_name: str,
        *,
        satellite: str | None = None,
        index_name: str | None = None,
        session_id: str = "",
    ) -> None:
        self.step_name = step_name
        self.satellite = satellite
        self.index_name = index_name
        self.session_id = session_id
        self._extra: dict[str, Any] = {}
        self._start: float = 0.0
        self._end: float = 0.0
        self._failed: bool = False
        self._error_msg: str | None = None

    # Allow callers to attach arbitrary k/v after the step runs
    def set_metadata(self, **kwargs: Any) -> None:
        self._extra.update(kwargs)

    def mark_failure(self, error: str) -> None:
        """Explicitly mark this step as failed (for caught exceptions)."""
        self._failed = True
        self._error_msg = error
        self._extra["error"] = error


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline metrics store
# ─────────────────────────────────────────────────────────────────────────────


class PipelineMetrics:
    """Thread-safe store for :class:`StepRecord` instances.

    Works similarly to :class:`TokenTracker` — records are kept in memory
    and optionally persisted to a JSONL file.
    """

    def __init__(self, log_path: Path | str | None = None) -> None:
        self._records: list[StepRecord] = []
        self._lock = threading.Lock()

        if log_path is not None:
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._log_path = None

    # ── record ────────────────────────────────────────────────────────────

    def record(self, rec: StepRecord) -> None:
        """Append a :class:`StepRecord` and optionally persist it."""
        with self._lock:
            self._records.append(rec)

        if self._log_path is not None:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(rec.to_json_line() + "\n")
            except OSError as exc:
                logger.warning("Failed to write metrics log: %s", exc)

        log_fn = logger.info if rec.status == "success" else logger.warning
        log_fn(
            "STEP %s | %s | %.3fs | satellite=%s index=%s | %s",
            rec.step_name,
            rec.status,
            rec.duration_s,
            rec.satellite or "-",
            rec.index_name or "-",
            rec.error or "OK",
        )

    # ── queries ───────────────────────────────────────────────────────────

    @property
    def records(self) -> list[StepRecord]:
        with self._lock:
            return list(self._records)

    def last_n(self, n: int = 10) -> list[StepRecord]:
        with self._lock:
            return list(self._records[-n:])

    # ── aggregated summary ────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return an aggregated summary dict."""
        recs = self.records
        if not recs:
            return {
                "total_steps": 0,
                "successes": 0,
                "failures": 0,
                "total_duration_s": 0.0,
                "by_step": {},
            }

        by_step: dict[str, dict[str, Any]] = {}
        successes = failures = 0
        total_dur = 0.0

        for r in recs:
            total_dur += r.duration_s
            if r.status == "success":
                successes += 1
            else:
                failures += 1

            bucket = by_step.setdefault(
                r.step_name,
                {
                    "calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_s": 0.0,
                    "avg_s": 0.0,
                    "satellites": set(),
                    "indices": set(),
                },
            )
            bucket["calls"] += 1
            bucket["total_s"] = round(bucket["total_s"] + r.duration_s, 4)
            if r.status == "success":
                bucket["successes"] += 1
            else:
                bucket["failures"] += 1
            if r.satellite:
                bucket["satellites"].add(r.satellite)
            if r.index_name:
                bucket["indices"].add(r.index_name)

        # Convert sets → sorted lists for JSON serialisation
        for bucket in by_step.values():
            bucket["avg_s"] = (
                round(bucket["total_s"] / bucket["calls"], 4)
                if bucket["calls"]
                else 0.0
            )
            bucket["satellites"] = sorted(bucket["satellites"])
            bucket["indices"] = sorted(bucket["indices"])

        return {
            "total_steps": len(recs),
            "successes": successes,
            "failures": failures,
            "total_duration_s": round(total_dur, 3),
            "by_step": by_step,
        }

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_metrics: PipelineMetrics | None = None


def get_metrics(
    log_path: Path | str | None = "data/logs/pipeline_metrics.jsonl",
) -> PipelineMetrics:
    """Get or create the global :class:`PipelineMetrics` singleton."""
    global _metrics
    if _metrics is None:
        _metrics = PipelineMetrics(log_path=log_path)
    return _metrics


# ─────────────────────────────────────────────────────────────────────────────
# Context manager
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def track_step(
    step_name: str,
    *,
    satellite: str | None = None,
    index_name: str | None = None,
    session_id: str = "",
    metrics: PipelineMetrics | None = None,
) -> Generator[StepTimer, None, None]:
    """Context manager that times a pipeline step and records the result.

    Usage::

        with track_step("scene_search", satellite="landsat") as step:
            results = do_search()
            step.set_metadata(scenes_found=len(results))
    """
    pm = metrics or get_metrics()
    timer = StepTimer(
        step_name,
        satellite=satellite,
        index_name=index_name,
        session_id=session_id,
    )
    timer._start = time.perf_counter()
    ts = datetime.now(timezone.utc).isoformat()

    status = "success"
    error: str | None = None
    try:
        yield timer
    except Exception as exc:
        status = "failure"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timer._end = time.perf_counter()
        duration = timer._end - timer._start

        # Also honour explicit mark_failure() calls from caught exceptions
        if timer._failed:
            status = "failure"
            error = error or timer._error_msg

        rec = StepRecord(
            timestamp=ts,
            step_name=timer.step_name,
            duration_s=round(duration, 4),
            status=status,
            error=error,
            satellite=timer.satellite or satellite,
            index_name=timer.index_name or index_name,
            session_id=timer.session_id or session_id,
            metadata=timer._extra,
        )
        pm.record(rec)
