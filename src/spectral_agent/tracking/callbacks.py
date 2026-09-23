"""
LangChain callback handler for automatic token & cost tracking.

Hooks into ``on_llm_end`` so that every LLM call made by the agent
(tool-calling loops included) is automatically recorded by the
:class:`TokenTracker`.

Usage::

    from spectral_agent.tracking.callbacks import CostTrackingHandler
    from spectral_agent.tracking import get_tracker

    handler = CostTrackingHandler(tracker=get_tracker(), session_id="sess-1")
    llm = ChatOpenAI(..., callbacks=[handler])
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from spectral_agent.tracking.token_tracker import TokenTracker, get_tracker

logger = logging.getLogger(__name__)


class CostTrackingHandler(BaseCallbackHandler):
    """
    LangChain callback that records token usage after every LLM call.

    Parameters
    ----------
    tracker : TokenTracker | None
        Tracker instance (defaults to the module-level singleton).
    session_id : str
        Identifier for the current user / session.
    """

    def __init__(
        self,
        tracker: TokenTracker | None = None,
        session_id: str = "system",
    ) -> None:
        super().__init__()
        self._tracker = tracker or get_tracker()
        self._session_id = session_id

    def set_session_id(self, session_id: str) -> None:
        """Update the session ID used for subsequent LLM usage records."""
        self._session_id = session_id

    # ── LangChain callback hooks ────────────────────────────────────────

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call finishes — record token usage."""
        try:
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage", {})

            # Some providers nest differently
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            output_tokens = (
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )

            # Determine model / provider from llm_output
            model = llm_output.get("model_name") or llm_output.get("model", "unknown")
            provider = _infer_provider(model)

            self._tracker.record(
                model=model,
                provider=provider,
                operation="agent",
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                user=self._session_id,
                session_id=self._session_id,
                metadata={
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                },
            )
        except Exception:
            logger.debug("Failed to record LLM usage from callback", exc_info=True)


def _infer_provider(model_name: str) -> str:
    """Best-effort guess of the provider from the model string."""
    model_lower = model_name.lower()
    if any(k in model_lower for k in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_cost_handler: CostTrackingHandler | None = None


def get_cost_handler() -> CostTrackingHandler:
    """Get or create the global :class:`CostTrackingHandler` singleton."""
    global _cost_handler
    if _cost_handler is None:
        _cost_handler = CostTrackingHandler(tracker=get_tracker())
    return _cost_handler
