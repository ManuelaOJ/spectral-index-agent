"""
Token usage tracking and cost estimation.

Logs every LLM call with model, tokens, cost, timestamp, and metadata.
Stores records in JSON Lines format for easy analysis.

Pricing sources (verified 2026-03-05):
  OpenAI  — https://developers.openai.com/api/docs/pricing
  Anthropic — https://platform.claude.com/docs/en/about-claude/pricing
  Google  — https://ai.google.dev/gemini-api/docs/pricing

  Google (USD / 1M tokens, verified 2026-04-14):
    gemini-2.5-flash    input  $0.30  output  $2.50  (output price includes thinking tokens)

  OpenAI (USD / 1M tokens, standard tier):
    gpt-4o              input $2.50   output $10.00
    gpt-4o-2024-05-13   input $5.00   output $15.00
    gpt-4o-mini         input $0.15   output  $0.60
    gpt-4.1             input $2.00   output  $8.00
    gpt-4.1-mini        input $0.40   output  $1.60
    gpt-4.1-nano        input $0.10   output  $0.40
    gpt-4-turbo         input $10.00  output $30.00
    gpt-3.5-turbo       input $0.50   output  $1.50
    o3                  input $2.00   output  $8.00
    o3-mini             input $1.10   output  $4.40
    o4-mini             input $1.10   output  $4.40
    o1                  input $15.00  output $60.00
    o1-mini             input $1.10   output  $4.40

  Anthropic (USD / 1M tokens):
    claude-opus-4       input $15.00  output $75.00
    claude-sonnet-4     input  $3.00  output $15.00
    claude-haiku-4-5    input  $1.00  output  $5.00
    claude-3.5-sonnet   input  $3.00  output $15.00
    claude-3.5-haiku    input  $0.80  output  $4.00
    claude-3-opus       input $15.00  output $75.00
    claude-3-haiku      input  $0.25  output  $1.25
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
# Pricing table (USD per 1 M tokens)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PRICING: dict[str, dict[str, float]] = {
    # ── OpenAI (https://developers.openai.com/api/docs/pricing, 2026-03-05) ──
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    # ── Anthropic (https://platform.claude.com/docs/en/about-claude/pricing, 2026-03-05) ──
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-4": {"input": 1.00, "output": 5.00},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # ── Google (https://ai.google.dev/gemini-api/docs/pricing) ──
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}

# Keys sorted longest-first so "gpt-4o-mini" matches before "gpt-4o"
_PRICING_KEYS_SORTED = sorted(MODEL_PRICING, key=len, reverse=True)


def _resolve_pricing(model: str) -> dict[str, float]:
    """Resolve pricing for *model*, matching versioned names by prefix.

    E.g. ``gpt-4o-2024-08-06`` → pricing for ``gpt-4o``.
    """
    # 1. Exact match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # 2. Prefix / substring match (longest key first to avoid false positives)
    model_lower = model.lower()
    for key in _PRICING_KEYS_SORTED:
        if model_lower.startswith(key):
            return MODEL_PRICING[key]
    return {"input": 0.0, "output": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Usage Record
# ─────────────────────────────────────────────────────────────────────────────


class UsageRecord(BaseModel):
    """Single LLM usage record."""

    timestamp: str = Field(description="ISO-8601 UTC timestamp")
    user: str = Field(default="system", description="User or session identifier")
    session_id: str = Field(default="", description="Session / thread identifier")
    model: str = Field(description="LLM model name")
    provider: str = Field(description="LLM provider (openai / anthropic)")
    operation: str = Field(
        description="Operation type (extraction / normalization / rules / etc.)"
    )
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    input_cost_usd: float = Field(ge=0.0)
    output_cost_usd: float = Field(ge=0.0)
    total_cost_usd: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json_line(self) -> str:
        return self.model_dump_json()


# ─────────────────────────────────────────────────────────────────────────────
# Token Tracker
# ─────────────────────────────────────────────────────────────────────────────


class TokenTracker:
    """
    Thread-safe token & cost tracker.

    Records are kept in memory and optionally appended to a JSONL file.
    """

    def __init__(
        self,
        log_path: Path | str | None = None,
        default_user: str = "system",
    ) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._default_user = default_user

        if log_path is not None:
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._log_path = None

    # ── public API ──────────────────────────────────────────────────────

    def record(
        self,
        *,
        model: str,
        provider: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        user: str | None = None,
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """
        Record a single LLM invocation.

        Returns the created UsageRecord.
        """
        pricing = _resolve_pricing(model)
        input_cost = input_tokens * pricing["input"] / 1_000_000
        output_cost = output_tokens * pricing["output"] / 1_000_000

        rec = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user=user or self._default_user,
            session_id=session_id,
            model=model,
            provider=provider,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            total_cost_usd=round(input_cost + output_cost, 8),
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(rec)

        # Persist to file
        if self._log_path is not None:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(rec.to_json_line() + "\n")
            except OSError as exc:
                logger.warning("Failed to write token log: %s", exc)

        logger.info(
            "LLM usage | model=%s op=%s in=%d out=%d cost=$%.6f",
            model,
            operation,
            input_tokens,
            output_tokens,
            rec.total_cost_usd,
        )
        return rec

    def record_from_response(
        self,
        *,
        response: Any,
        model: str,
        provider: str,
        operation: str,
        user: str | None = None,
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """
        Extract token counts from a LangChain response and record.

        Works with ChatOpenAI / ChatAnthropic response objects.
        """
        usage = getattr(response, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)

        return self.record(
            model=model,
            provider=provider,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user=user,
            session_id=session_id,
            metadata=metadata,
        )

    # ── summary helpers ─────────────────────────────────────────────────

    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.output_tokens for r in self._records)

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return round(sum(r.total_cost_usd for r in self._records), 6)

    def summary(self) -> dict[str, Any]:
        """Return an aggregated summary of all recorded usage."""
        records = self.records
        by_op: dict[str, dict] = {}
        for r in records:
            bucket = by_op.setdefault(
                r.operation,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += r.input_tokens
            bucket["output_tokens"] += r.output_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + r.total_cost_usd, 8)

        return {
            "total_calls": len(records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_operation": by_op,
        }

    def reset(self) -> None:
        """Clear in-memory records (does not delete the log file)."""
        with self._lock:
            self._records.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_tracker: TokenTracker | None = None


def get_tracker(
    log_path: Path | str | None = "data/logs/token_usage.jsonl",
    default_user: str = "system",
) -> TokenTracker:
    """Get or create the global TokenTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker(log_path=log_path, default_user=default_user)
    return _tracker
