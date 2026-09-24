# System Architecture

## High-Level Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           AI SPECTRAL INDEX AGENT PLATFORM                      │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐   │
│  │   USER / API    │───▶│              LANGGRAPH ORCHESTRATOR              │   │
│  └─────────────────┘    │                                                   │   │
│                         │  ┌─────────┐   ┌─────────┐   ┌─────────┐        │   │
│                         │  │ INGEST  │──▶│ RASTER  │──▶│  MAPS   │        │   │
│                         │  └─────────┘   └─────────┘   └─────────┘        │   │
│                         │       │             │             │              │   │
│                         │       ▼             ▼             ▼              │   │
│                         │  ┌─────────────────────────────────────┐        │   │
│                         │  │         SHARED STATE GRAPH          │        │   │
│                         │  └─────────────────────────────────────┘        │   │
│                         └─────────────────────────────────────────────────┘   │
│                                              │                                  │
│                    ┌─────────────────────────┼─────────────────────────┐       │
│                    │                         │                         │       │
│              ┌─────▼─────┐            ┌─────▼─────┐            ┌─────▼─────┐  │
│              │  LANGCHAIN │            │  LANGCHAIN │            │  LANGCHAIN │  │
│              │   TOOLS    │            │   TOOLS    │            │   TOOLS    │  │
│              │ (Ingestion)│            │  (Raster)  │            │  (Maps)    │  │
│              └─────┬─────┘            └─────┬─────┘            └─────┬─────┘  │
│                    │                         │                         │       │
│              ┌─────▼─────┐            ┌─────▼─────┐            ┌─────▼─────┐  │
│              │  Landsat  │            │ Crop bands│            │ Thematic  │  │
│              │  Sentinel │            │  Compute  │            │  PNG and  │  │
│              │    API    │            │   index   │            │ HTML maps │  │
│              └───────────┘            └───────────┘            └───────────┘  │
│                                                                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                              EXTERNAL SERVICES                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  USGS M2M    │   │  Copernicus  │   │   OpenAI /   │   │  Local disk  │    │
│  │  (Landsat)   │   │  (Sentinel)  │   │   Anthropic  │   │   (data/)    │    │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘    │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. LangGraph Orchestrator

The **LangGraph Orchestrator** manages workflow execution through stateful graphs:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION GRAPH EXAMPLE                      │
│                                                                  │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────┐ │
│   │   START   │───▶│  VALIDATE │───▶│  DOWNLOAD │───▶│  END  │ │
│   │           │    │   INPUT   │    │  IMAGERY  │    │       │ │
│   └───────────┘    └─────┬─────┘    └─────┬─────┘    └───────┘ │
│                          │                │                      │
│                          │ invalid        │ error                │
│                          ▼                ▼                      │
│                    ┌───────────┐    ┌───────────┐               │
│                    │   ERROR   │    │   RETRY   │               │
│                    │  HANDLER  │    │   LOGIC   │               │
│                    └───────────┘    └───────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- **Stateful execution**: State persists across nodes
- **Conditional routing**: Dynamic path selection based on results
- **Conversation memory**: chat history kept per session (`MemorySaver`)

---

### 2. LangChain Tools

Tools are the atomic units of capability, wrapped for agent consumption:

```python
# Tool Architecture
┌─────────────────────────────────────────────────────────┐
│                    LANGCHAIN TOOL                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │  @tool decorator / BaseTool inheritance            │ │
│  │                                                     │ │
│  │  - name: str           # Tool identifier           │ │
│  │  - description: str    # LLM-readable purpose      │ │
│  │  - args_schema: Type   # Pydantic input model      │ │
│  │  - _run(): Result      # Sync execution            │ │
│  │  - _arun(): Result     # Async execution           │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │              IMPLEMENTATION LAYER                   │ │
│  │  - API clients (Landsat, Sentinel)                 │ │
│  │  - Data processing logic                           │ │
│  │  - Error handling & retries                        │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Tool Categories:**
| Category | Tools | Location |
|----------|-------|----------|
| Ingestion | `search_landsat_tool`, `download_landsat_tool`, `search_sentinel_tool`, `download_sentinel_tool`, `download_sentinel_index_tool`, `search_satellite_imagery_tool` | `tools/ingestion/lc_tools.py` |
| Raster | `crop_landsat_bands_tool`, `compute_spectral_index_tool`, `list_cached_bands_tool`, `list_available_indices_tool` | `tools/raster/lc_tools.py` |
| Visualization | `generate_thematic_map_tool` | `tools/visualization/lc_tools.py` |

Index formulas and band requirements are defined in `tools/indices/spectral_indices.py`.
All tools are collected in `tools/_registry.py` (`get_all_tools()`).

---

### 3. State Management

LangGraph state flows through the graph, accumulating results:

```python
class IngestionState(TypedDict):
    """State schema for ingestion workflow"""
    
    # Input parameters
    aoi: dict                    # Area of interest (GeoJSON)
    date_range: tuple            # (start_date, end_date)
    satellites: list[str]        # ["landsat", "sentinel"]
    
    # Processing state
    search_results: list[dict]   # Found scenes
    download_status: dict        # {scene_id: status}
    
    # Output
    downloaded_files: list[str]  # File paths
    errors: list[str]            # Error messages
    metadata: dict               # Scene metadata
```

**State Flow:**
```
START → Node1(add search_results) → Node2(add downloaded_files) → END
         ↓                           ↓
    State grows              State is complete
```

---

### 4. Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPECTRAL REASONING AGENT                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                         LLM CORE                            ││
│  │  - Receives user query                                      ││
│  │  - Reasons about required steps                             ││
│  │  - Selects appropriate tools                                ││
│  │  - Interprets results                                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│   ┌───────────┐        ┌───────────┐        ┌───────────┐      │
│   │ Ingestion │        │  Raster   │        │    Map    │      │
│   │   Tools   │        │   Tools   │        │   Tools   │      │
│   └───────────┘        └───────────┘        └───────────┘      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      MEMORY STORE                           ││
│  │  - Conversation history                                     ││
│  │  - Previous results cache                                   ││
│  │  - Last area of interest and date range                     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

### 5. Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY (tracking/)                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  LangSmith   │  │  Token and   │  │  Step and    │          │
│  │   tracing    │  │  cost usage  │  │ conversation │          │
│  │  (optional)  │  │              │  │     logs     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Written as JSONL under data/logs/:                              │
│  - token_usage.jsonl          (TokenTracker, CostTrackingHandler)│
│  - pipeline_metrics.jsonl     (PipelineMetrics, step timings)    │
│  - conversations.jsonl        (ConversationLogger)               │
│  - debug_computations.jsonl   (DebugLogger)                      │
└─────────────────────────────────────────────────────────────────┘
```

LangSmith tracing is not configured by the code. LangChain turns it on by itself
when its standard variables are set (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`).

**Performance:**
- **Async**: the ingestion graph nodes are `async` functions
- **Band cache**: `BandCache` (`tools/raster/band_cache.py`) keeps clipped bands,
  so indices that share bands (e.g. NDVI and SAVI) do not clip them twice
- **Retries**: USGS requests are retried with an increasing delay on HTTP 429 and 5xx

---

### 6. Extension Points

Where to extend the code:

| Extension Point | Purpose | Implementation |
|-----------------|---------|----------------|
| New Satellites | Add data sources | Implement `BaseIngestionTool` (`core/base_tool.py`) |
| Custom Indices | User-defined formulas | Add an entry to `SPECTRAL_INDICES` (`tools/indices/spectral_indices.py`) |
| New Agents | Specialized reasoning | Change the system prompt or tool list in `graphs/agent_graph.py` |
| LLM Providers | Model flexibility | Add a provider branch in `graphs/agent_graph.py` (OpenAI, Anthropic and Google today) |

---

## Security Considerations

- **API Keys**: Stored in environment variables, never in code
- **Input Validation**: All user inputs validated via Pydantic
- **Rate Limiting**: Implemented at tool level
- **Audit Logging**: All operations logged with context
