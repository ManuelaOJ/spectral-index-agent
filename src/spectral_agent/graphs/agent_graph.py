"""
LangGraph orchestrator agent with conversational memory.

This module replaces the stateless prebuilt ReAct agent with a
``MemorySaver``-backed graph so that chat history is automatically
persisted across turns within the same *thread_id*.

The agent still uses ``create_react_agent`` under the hood (which builds a
proper ``StateGraph`` with tool-calling loops) but now:

1. A ``MemorySaver`` checkpointer persists the message list.
2. A ``SessionStore`` (injected) tracks application artefacts (downloads,
   computed indices, generated maps) so that tools can avoid duplicate work.

Usage
-----
::

    from spectral_agent.graphs.agent_graph import build_agent, AgentDeps

    deps = AgentDeps()                         # defaults
    agent = build_agent(deps)

    # First turn
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Busca Landsat 9 2024 en Colombia")]},
        config={"configurable": {"thread_id": "sess-1"}},
    )

    # Second turn — chat history is remembered
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Ahora calcula NDVI")]},
        config={"configurable": {"thread_id": "sess-1"}},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from spectral_agent.config import get_settings
from spectral_agent.memory.session_store import SessionStore
from spectral_agent.tools._registry import get_all_tools
from spectral_agent.tracking.callbacks import CostTrackingHandler, get_cost_handler
from spectral_agent.tracking import get_tracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RECURSION_LIMIT: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI Geologist and Remote Sensing Specialist.

Capabilities: search Landsat 4–9 & Sentinel-2 scenes, download archives,
crop bands, compute indices (NDVI, EVI, SAVI, NDWI, NBR, NDBI), and
generate thematic maps (PNG + interactive HTML).

## STEP 1 — PARSE the user's message (do this BEFORE anything else)

Read the ENTIRE user message and extract these three parameters:

### A) DATES — extract start_date and end_date
Look for ANY temporal reference in ANY language.  Apply the first match:
- "entre el 10 de enero y el 20 de abril de 2011" → 2011-01-10 / 2011-04-20
- "el 25 de julio de 2013" → 2013-07-25 / 2013-07-25
- "entre enero y febrero de 2022" → 2022-01-01 / 2022-02-28
- "septiembre de 2014" or "julio 2014" → first day / last day of that month
- "2020", "año 2012", "para el año 2012" → YYYY-01-01 / YYYY-12-31
- "2018 a 2022" → 2018-01-01 / 2022-12-31
- "2025-09-17" → 2025-09-17 / 2025-09-17
ANY year number (2011, 2013, 2020…), ANY month name (marzo, julio, jan,
sept…), or ANY range word (entre, between, from, to, durante) is a date.

### B) AOI — extract bounding box
- Array like [-73.84, 6.88, -73.81, 6.91] → interpret as [lon1, lat1, lon2, lat2].
  west=min(lon1,lon2), south=min(lat1,lat2), east=max(lon1,lon2), north=max(lat1,lat2).
- [Uploaded geometry file: ...] block → use the bbox from that block.
- Explicit west/south/east/north values → use directly.

### C) SATELLITE — extract satellite name
- "Landsat", "imágenes Landsat", "Landsat 8" → Landsat
- "Sentinel-2", "Sentinel", "imágenes Sentinel" → Sentinel-2
- Scene ID prefix: LC08/LC09 → Landsat, S2A/S2B → Sentinel-2

## STEP 2 — CHECK for missing parameters

After parsing, check which of the three parameters (dates, AOI, satellite)
you could NOT find.  Only ask about genuinely missing ones:
- **No satellite found** → respond ONLY with: "Landsat or Sentinel-2?"
- **No AOI found** → ask for geometry upload or bounding-box coordinates.
- **No dates found** → ask for a date or date range.
- **Multiple missing** → ask for ALL missing pieces in a SINGLE response.
If ALL three parameters were found, proceed to Step 3 immediately.
NEVER ask for a parameter you already extracted.

## STEP 3 — EXECUTE the full pipeline (no pausing)

Once satellite, dates, and AOI are all available, run the entire pipeline
in one turn without asking "shall I proceed?":

### Landsat pipeline
search_landsat_tool → download_landsat_tool (entity_id + collection)
→ crop_landsat_bands_tool → compute_spectral_index_tool
→ generate_thematic_map_tool

### Sentinel-2 pipeline (different — indices computed server-side)
search_sentinel_tool → download_sentinel_index_tool (scene_id + index_names + bbox)
→ generate_thematic_map_tool
Do NOT use crop/compute tools for Sentinel — they only work with Landsat.

### Execution rules
- Pick the best scene: first result (lowest cloud cover).
- If a tool fails → retry with the next best scene automatically.
- Only stop to ask the user when ALL scenes have failed.
- Sensor name from scene_id: LC08→Landsat 8, LC09→Landsat 9,
  LE07→Landsat 7, LT05→Landsat 5, LT04→Landsat 4.

## Additional rules

### Cloud cover
- Do NOT set max_cloud_cover unless the user explicitly requests a limit.
  Leave it at the default (100).  Results are sorted by lowest cloud cover.
- Only set max_cloud_cover < 100 if the user says "less than X% clouds".

### Satellite selection
- If the user names a satellite → use ONLY that satellite.
- NEVER silently search both.  NEVER ask "which satellite?" again once known.
- Use search_landsat_tool or search_sentinel_tool directly.

### Search returns 0 scenes
DO NOT widen the date range on your own.  Inform the user:
"No scenes found for [date]. Would you like me to search [wider range]?"
Apply this ONLY when search succeeded with total_found=0.
If the tool returned an error, report the failure instead.

### Date precision
- NEVER widen the date window beyond what the user stated.
- NEVER fabricate coordinates.

## Debug / Testing Information
After computing indices, include in your reply:
- **Formula used** (from debug_info). For SAVI mention L=0.5.
- **Bands selected** (e.g. SR_B4, SR_B5). Specify sensor band mapping:
    Landsat 4-7: NIR=SR_B4, RED=SR_B3
    Landsat 8-9: NIR=SR_B5, RED=SR_B4
    Sentinel-2: NIR=B08, RED=B04
- **Satellite & sensor**, **index parameters**, **debug log path**, **GeoTIFF path**.
Format as a "Debug Info" section at the end.

Reuse cached data when possible. Answer in the user's language.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Dependencies container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentDeps:
    """
    Everything the agent graph needs, bundled for easy injection.

    Parameters
    ----------
    provider : str
        ``"openai"`` or ``"anthropic"``.
    model : str
        Model name, e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max output tokens per LLM call.
    system_prompt : str | None
        Custom system prompt (defaults to ``SYSTEM_PROMPT``).
    extra_tools : list[BaseTool]
        Additional tools to attach alongside the built-in ones.
    session_store : SessionStore
        Application-level session state store.
    checkpointer : MemorySaver | None
        LangGraph checkpoint saver.  A fresh ``MemorySaver`` is created if
        *None*.
    """

    provider: Literal["openai", "anthropic", "google"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    usgs_username: str = ""
    usgs_token: str = ""
    copernicus_client_id: str = ""
    copernicus_client_secret: str = ""
    system_prompt: str | None = None
    extra_tools: list[BaseTool] = field(default_factory=list)
    session_store: SessionStore = field(default_factory=SessionStore)
    checkpointer: MemorySaver | None = None

    @classmethod
    def from_settings(cls, **overrides: Any) -> "AgentDeps":
        """Build from ``Settings`` with optional field overrides."""
        settings = get_settings()
        kwargs: dict[str, Any] = {
            "provider": settings.default_llm_provider,
            "model": settings.default_llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        kwargs.update(overrides)
        return cls(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm(deps: AgentDeps) -> BaseChatModel:
    settings = get_settings()

    # Use the module-level singleton so chat.py can update session_id later
    cost_handler = get_cost_handler()

    if deps.provider == "openai":
        api_key = deps.openai_api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key not configured (SPECTRAL_OPENAI_API_KEY)")
        return ChatOpenAI(
            model=deps.model,
            temperature=deps.temperature,
            max_tokens=deps.max_tokens,
            api_key=api_key,
            max_retries=3,
            callbacks=[cost_handler],
        )

    if deps.provider == "anthropic":
        api_key = deps.anthropic_api_key or settings.anthropic_api_key
        if not api_key:
            raise ValueError(
                "Anthropic API key not configured (SPECTRAL_ANTHROPIC_API_KEY)"
            )
        return ChatAnthropic(
            model=deps.model,
            temperature=deps.temperature,
            max_tokens=deps.max_tokens,
            api_key=api_key,
            callbacks=[cost_handler],
        )

    if deps.provider == "google":
        api_key = deps.google_api_key or settings.google_api_key
        if not api_key:
            raise ValueError("Google API key not configured (GOOGLE_API_KEY)")
        return ChatGoogleGenerativeAI(
            model=deps.model,
            temperature=deps.temperature,
            max_output_tokens=deps.max_tokens,
            google_api_key=api_key,
            callbacks=[cost_handler],
        )

    raise ValueError(f"Unknown LLM provider: {deps.provider}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────


def build_agent(
    deps: AgentDeps | None = None,
    *,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> CompiledGraph:
    """
    Build and compile the orchestrator agent graph.

    Parameters
    ----------
    deps : AgentDeps | None
        Dependency bundle.  Falls back to ``AgentDeps.from_settings()`` when
        *None*.

    Returns
    -------
    CompiledGraph
        Ready to ``invoke`` / ``ainvoke`` with
        ``config={"configurable": {"thread_id": "<session-id>"}}``.
    """
    deps = deps or AgentDeps.from_settings()

    llm = _build_llm(deps)

    tools: list[BaseTool] = get_all_tools()
    if deps.extra_tools:
        tools.extend(deps.extra_tools)

    checkpointer = deps.checkpointer or MemorySaver()
    system_prompt = deps.system_prompt or SYSTEM_PROMPT

    # Pass system prompt as a plain string — langgraph handles prepending
    # it as a SystemMessage automatically. This is the most reliable
    # approach across langgraph versions.
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
        name="spectral_agent",
    )

    # Attach recursion_limit so callers can read it when building configs.
    agent._default_recursion_limit = recursion_limit  # type: ignore[attr-defined]

    logger.info(
        "Built spectral agent (%s/%s) with %d tools, recursion_limit=%d",
        deps.provider,
        deps.model,
        len(tools),
        recursion_limit,
    )

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_thread_config(thread_id: str) -> dict[str, Any]:
    """
    Build the ``config`` dict expected by ``agent.invoke()``::

        agent.invoke({"messages": [...]}, config=make_thread_config("sess-1"))
    """
    return {"configurable": {"thread_id": thread_id}}
