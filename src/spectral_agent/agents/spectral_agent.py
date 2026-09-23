"""
Spectral Reasoning Agent.

The main agent that orchestrates satellite imagery analysis and mineral targeting
using LangChain tools and LangGraph workflows.

.. note::
   For the *memory-enabled* agent backed by ``MemorySaver``, prefer
   :func:`spectral_agent.graphs.agent_graph.build_agent`.  This module is
   retained for backward compatibility and simpler one-shot usage.
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph

from spectral_agent.config import get_settings
from spectral_agent.tools._registry import get_all_tools
from spectral_agent.tracking.callbacks import CostTrackingHandler
from spectral_agent.tracking import get_tracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SpectralAgentConfig:
    """Configuration for the Spectral Agent."""

    # LLM settings
    provider: Literal["openai", "anthropic", "google"] = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Agent behavior
    system_prompt: str | None = None
    verbose: bool = False
    use_memory: bool = True  # enable MemorySaver checkpointer

    @classmethod
    def from_settings(cls) -> "SpectralAgentConfig":
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            provider=settings.default_llm_provider,
            model=settings.default_llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are an expert AI Geologist and Remote Sensing Specialist with deep knowledge of:

1. **Satellite Imagery Analysis**
   - Landsat (4–9) and Sentinel-2 data products
   - Spectral characteristics of minerals, vegetation, water, and urban areas
   - Atmospheric effects and image preprocessing

2. **Spectral Indices**
   - Vegetation indices: NDVI, EVI, SAVI
   - Water index: NDWI
   - Burn index: NBR
   - Urban/built-up index: NDBI
   - Landsat Collection 2 Level-2 band mappings for sensors 4–9

3. **Mineral Exploration & Land Analysis**
   - Alteration mapping
   - Land-use / land-cover change detection
   - Target prioritisation

You have access to tools that can:
- **Search** for satellite imagery (Landsat and Sentinel-2)
- **Download** full Landsat scenes (.tar) or Sentinel-2 products
- **Crop** Landsat bands to a bounding box (caching shared bands)
- **Compute spectral indices** from cropped bands (NDVI, EVI, SAVI, NDWI, NBR, NDBI)
- **Generate thematic maps** — static PNG and interactive HTML
- **List cached bands** to avoid redundant downloads
- **List available indices** with required bands and value ranges

Typical workflow:
1. Search for imagery (search_landsat_tool / search_sentinel_tool)
2. Download the best scene (download_landsat_tool)
3. Crop bands for requested indices (crop_landsat_bands_tool)
4. Compute indices (compute_spectral_index_tool)
5. Generate maps (generate_thematic_map_tool)

When helping users:
- Always validate the area of interest and date range
- Consider cloud cover when selecting imagery
- Recommend appropriate spectral indices for the analysis goal
- Explain what each index measures and how to interpret values
- Use list_available_indices_tool when users ask what’s available

CRITICAL RULES — never break these:
- You MUST call search_landsat_tool or search_sentinel_tool BEFORE concluding that no imagery is available. Never assume, guess, or infer availability without actually calling the search tool first.
- If the user provides a year without specific dates, search the full year (start_date = YYYY-01-01, end_date = YYYY-12-31).
- If the initial search returns no results, try widening the date range or lowering the cloud cover threshold before giving up.

Be precise with coordinates (WGS84 decimal degrees) and dates (YYYY-MM-DD).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent Factory
# ─────────────────────────────────────────────────────────────────────────────


def get_llm(config: SpectralAgentConfig) -> BaseChatModel:
    """Create the appropriate LLM based on configuration."""
    settings = get_settings()
    cost_handler = CostTrackingHandler(tracker=get_tracker())

    if config.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=settings.openai_api_key,
            callbacks=[cost_handler],
        )
    elif config.provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=settings.anthropic_api_key,
            callbacks=[cost_handler],
        )
    elif config.provider == "google":
        if not settings.google_api_key:
            raise ValueError("Google API key not configured (GOOGLE_API_KEY)")
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            google_api_key=settings.google_api_key,
            callbacks=[cost_handler],
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


def create_spectral_agent(
    config: SpectralAgentConfig | None = None,
    additional_tools: list | None = None,
    checkpointer: MemorySaver | None = None,
) -> CompiledGraph:
    """
    Create a Spectral Reasoning Agent.

    The agent uses a ReAct pattern with access to imagery ingestion tools.
    When ``config.use_memory`` is *True* (default) a ``MemorySaver``
    checkpointer is attached so that chat history persists across turns
    within the same ``thread_id``.

    Args:
        config: Agent configuration (uses defaults from settings if None)
        additional_tools: Extra tools to add to the agent
        checkpointer: Optional pre-built checkpointer.  One is created
            automatically when ``config.use_memory`` is True and no
            checkpointer is supplied.

    Returns:
        CompiledGraph: A LangGraph agent ready to invoke.

    Example:
        >>> agent = create_spectral_agent()
        >>> result = await agent.ainvoke(
        ...     {"messages": [HumanMessage(content="Find Landsat imagery for Nevada")]},
        ...     config={"configurable": {"thread_id": "demo-1"}},
        ... )
    """
    config = config or SpectralAgentConfig.from_settings()

    # Get LLM
    llm = get_llm(config)

    # Collect tools
    tools = get_all_tools()
    if additional_tools:
        tools.extend(additional_tools)

    # Get system prompt
    system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT

    # Memory checkpointer
    if config.use_memory and checkpointer is None:
        checkpointer = MemorySaver()

    # Create the agent using LangGraph's prebuilt ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )

    logger.info(
        "Created Spectral Agent with %d tools using %s/%s (memory=%s)",
        len(tools),
        config.provider,
        config.model,
        config.use_memory,
    )

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


async def run_agent_query(
    query: str,
    config: SpectralAgentConfig | None = None,
    thread_id: str | None = None,
    agent: CompiledGraph | None = None,
) -> dict[str, Any]:
    """
    Run a single query through the agent.

    Convenience function for simple use cases.  If ``thread_id`` is provided
    the agent will remember previous messages in the same thread.

    Args:
        query: User's question or request
        config: Optional agent configuration
        thread_id: Optional thread identifier for multi-turn memory
        agent: Optional pre-built agent (avoids re-creation every call)

    Returns:
        dict: Agent response with messages

    Example:
        >>> result = await run_agent_query(
        ...     "Search for cloud-free Sentinel imagery over the Andes in 2024",
        ...     thread_id="demo-1",
        ... )
        >>> print(result["messages"][-1].content)
    """
    if agent is None:
        agent = create_spectral_agent(config)

    invoke_config: dict[str, Any] = {}
    if thread_id:
        invoke_config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=invoke_config,
    )

    return result
