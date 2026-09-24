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
│  │  USGS M2M    │   │  Copernicus  │   │   OpenAI /   │   │   Storage    │    │
│  │  (Landsat)   │   │  (Sentinel)  │   │   Anthropic  │   │  (S3/Local)  │    │
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
- **Checkpointing**: Resume from failures
- **Human-in-the-loop**: Optional approval gates

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
│  │  - User preferences                                         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

### 5. Observability & Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY STACK                         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  LangSmith   │  │  Prometheus  │  │   Grafana    │          │
│  │   Tracing    │  │   Metrics    │  │  Dashboards  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                   │
│                           │                                      │
│  Tracked:                                                        │
│  - LLM calls (tokens, latency)                                  │
│  - Tool executions (success/failure)                            │
│  - Graph traversals                                              │
│  - Download speeds & sizes                                       │
│  - Error rates                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Scaling Strategy:**
- **Horizontal**: Multiple worker processes for downloads
- **Async**: Non-blocking IO for API calls
- **Caching**: Redis for search results, file checksums
- **Queue**: Celery/RQ for background processing

---

### 6. Extension Points

The architecture supports future extensions:

| Extension Point | Purpose | Implementation |
|-----------------|---------|----------------|
| New Satellites | Add data sources | Implement `BaseIngestionTool` (`core/base_tool.py`) |
| Custom Indices | User-defined formulas | `IndicesRegistry.register()` |
| New Agents | Specialized reasoning | Subclass `BaseAgent` |
| Storage Backends | S3, GCS, Azure | `StorageAdapter` interface |
| LLM Providers | Model flexibility | LangChain model abstraction |

---

## Security Considerations

- **API Keys**: Stored in environment variables, never in code
- **Input Validation**: All user inputs validated via Pydantic
- **Rate Limiting**: Implemented at tool level
- **Audit Logging**: All operations logged with context
