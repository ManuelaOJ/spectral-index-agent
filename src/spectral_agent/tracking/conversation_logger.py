"""
Conversation logging for thesis reproducibility.

Captures the full cycle of each user–agent interaction: the original
user prompt, the enriched prompt (with geometry/context), the agent's
final response, intermediate tool calls, token usage, cost, and
wall-clock duration.

Records are persisted as JSON Lines for post-hoc qualitative and
quantitative analysis.

Usage
-----
::

    from spectral_agent.tracking.conversation_logger import get_conversation_logger

    conv_logger = get_conversation_logger()
    conv_logger.set_session_id("thread_abc123")

    conv_logger.log_turn(
        user_prompt="Calculate NDVI for Bogotá in January 2024",
        enriched_prompt="Calculate NDVI for Bogotá ...\n[Uploaded geometry ...]",
        agent_response="I found 3 Landsat scenes ...",
        tool_calls=[
            {"tool": "search_landsat_tool", "args": {...}, "result_summary": "3 scenes"},
        ],
        model="gpt-4o",
        provider="openai",
        input_tokens=1200,
        output_tokens=450,
        total_cost_usd=0.0075,
        duration_s=12.34,
    )
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation record
# ─────────────────────────────────────────────────────────────────────────────


class ConversationRecord(BaseModel):
    """Single user–agent interaction turn."""

    timestamp: str = Field(description="ISO-8601 UTC timestamp (turn start)")
    session_id: str = Field(default="", description="Thread / session identifier")
    turn_number: int = Field(ge=1, description="Sequential turn number within session")

    # ── Prompts ───────────────────────────────────────────────────────────
    user_prompt: str = Field(description="Original user input text")
    enriched_prompt: str = Field(description="Prompt after context injection (geometry, indices)")

    # ── Response ──────────────────────────────────────────────────────────
    agent_response: str = Field(description="Final agent text response")

    # ── Tool usage ────────────────────────────────────────────────────────
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool invocations: [{tool, args, result_summary}]",
    )

    # ── LLM usage ─────────────────────────────────────────────────────────
    model: str = Field(default="", description="LLM model name")
    provider: str = Field(default="", description="LLM provider (openai / anthropic)")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)

    # ── Timing ────────────────────────────────────────────────────────────
    duration_s: float = Field(default=0.0, ge=0.0, description="Wall-clock seconds")

    # ── Outcome ───────────────────────────────────────────────────────────
    status: str = Field(default="success", description="'success' or 'error'")
    error: str | None = Field(default=None, description="Error message if failed")

    def to_json_line(self) -> str:
        return self.model_dump_json()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation logger
# ─────────────────────────────────────────────────────────────────────────────


class ConversationLogger:
    """Thread-safe logger for full conversation turns.

    Follows the same singleton + JSONL pattern as :class:`TokenTracker`
    and :class:`PipelineMetrics`.
    """

    def __init__(self, log_path: Path | str | None = None) -> None:
        self._records: list[ConversationRecord] = []
        self._lock = threading.Lock()
        self._active_session_id: str = ""
        self._turn_counters: dict[str, int] = {}

        self._log_path: Path | None
        if log_path is not None:
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._log_path = None

    # ── session management ────────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        """Set the active session ID for subsequent calls."""
        self._active_session_id = session_id

    # ── record ────────────────────────────────────────────────────────────

    def log_turn(
        self,
        *,
        user_prompt: str,
        enriched_prompt: str,
        agent_response: str,
        tool_calls: list[dict[str, Any]] | None = None,
        model: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_cost_usd: float = 0.0,
        duration_s: float = 0.0,
        status: str = "success",
        error: str | None = None,
        session_id: str | None = None,
    ) -> ConversationRecord:
        """Log a complete user→agent turn.

        Parameters
        ----------
        user_prompt
            The raw text the user typed.
        enriched_prompt
            The prompt after ``_build_context_message()`` enrichment.
        agent_response
            The final text the agent returned.
        tool_calls
            List of ``{"tool": ..., "args": ..., "result_summary": ...}``
            dicts describing each tool invocation in the turn.
        model, provider
            LLM model and provider used for this turn.
        input_tokens, output_tokens, total_cost_usd
            Aggregated token/cost for the turn (from TokenTracker snapshot).
        duration_s
            Wall-clock time for the full ``_invoke_agent`` call.
        status
            ``"success"`` or ``"error"``.
        error
            Error message string if status is ``"error"``.
        session_id
            Override session ID; defaults to ``active_session_id``.
        """
        sid = session_id or self._active_session_id

        with self._lock:
            self._turn_counters[sid] = self._turn_counters.get(sid, 0) + 1
            turn_number = self._turn_counters[sid]

        rec = ConversationRecord(
            timestamp=datetime.now(UTC).isoformat(),
            session_id=sid,
            turn_number=turn_number,
            user_prompt=user_prompt,
            enriched_prompt=enriched_prompt,
            agent_response=agent_response,
            tool_calls=tool_calls or [],
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost_usd,
            duration_s=round(duration_s, 4),
            status=status,
            error=error,
        )

        with self._lock:
            self._records.append(rec)

        # Persist to JSONL
        if self._log_path is not None:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(rec.to_json_line() + "\n")
            except OSError as exc:
                logger.warning("Failed to write conversation log: %s", exc)

        logger.info(
            "CONVERSATION turn=%d | %s | model=%s | tools=%d | %.1fs | $%.6f",
            turn_number,
            status,
            model or "-",
            len(rec.tool_calls),
            duration_s,
            total_cost_usd,
        )
        return rec

    # ── queries ───────────────────────────────────────────────────────────

    @property
    def records(self) -> list[ConversationRecord]:
        with self._lock:
            return list(self._records)

    def for_session(self, session_id: str) -> list[ConversationRecord]:
        """Return records for a specific session."""
        with self._lock:
            return [r for r in self._records if r.session_id == session_id]

    def last_n(self, n: int = 10) -> list[ConversationRecord]:
        with self._lock:
            return list(self._records[-n:])

    def reset(self) -> None:
        """Clear in-memory records (does not delete the log file)."""
        with self._lock:
            self._records.clear()
            self._turn_counters.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_conversation_logger: ConversationLogger | None = None


def get_conversation_logger(
    log_path: Path | str | None = "data/logs/conversations.jsonl",
) -> ConversationLogger:
    """Get or create the global :class:`ConversationLogger` singleton."""
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = ConversationLogger(log_path=log_path)
    return _conversation_logger
