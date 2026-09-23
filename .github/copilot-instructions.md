# Copilot Instructions — Spectral Index Agent

## Project Overview

This is an **AI-powered geospatial agent** that automates satellite imagery workflows for spectral index computation. It is a **master's thesis project** in Applied Analytics and Artificial Intelligence for geospatial analysis.

The agent can: search satellite scenes, download imagery, crop bands to an AOI, compute spectral indices, and generate publication-quality maps — all orchestrated by an LLM via natural language.

## Tech Stack

- **Python 3.11+** with type hints everywhere
- **LangChain + LangGraph** — ReAct agent with tool-calling and `MemorySaver` checkpointer
- **LLM providers** — OpenAI (`gpt-4o`) and Anthropic (`claude`), configured via `pydantic-settings`
- **Geospatial** — `rasterio`, `geopandas`, `shapely`, `pyproj` for raster/vector ops
- **Satellite APIs** — USGS M2M (Landsat 4–9), Copernicus Data Space (Sentinel-2)
- **Visualization** — `matplotlib` (static PNG maps), `folium` (interactive HTML maps)
- **UI** — Streamlit app in `app/`
- **Testing** — `pytest` with `pytest-asyncio`

## Architecture

```
src/spectral_agent/
├── agents/           # Agent definitions (spectral_agent.py)
├── config/           # pydantic-settings (Settings with SPECTRAL_ env prefix)
├── core/             # Base classes, exceptions
├── graphs/           # LangGraph workflows
│   ├── agent_graph.py      # ReAct agent with memory + system prompt
│   └── ingestion_graph.py  # Stateful ingestion pipeline
├── memory/           # SessionStore for artifact tracking
├── pipeline/         # Extraction → normalization → rules → request builder
├── schemas/          # Pydantic data models
├── tools/
│   ├── _registry.py        # get_all_tools() — single source of truth
│   ├── ingestion/          # search_landsat_tool, download_*, sentinel tools
│   │   ├── landsat.py      # USGS M2M API client
│   │   ├── sentinel.py     # Copernicus API client
│   │   └── lc_tools.py     # @tool wrappers with Pydantic input schemas
│   ├── raster/             # crop_landsat_bands_tool, compute_spectral_index_tool
│   │   ├── index_calculator.py  # NumPy index math + GeoTIFF I/O
│   │   ├── landsat_processor.py # Band selection per sensor
│   │   └── lc_tools.py
│   └── visualization/      # generate_thematic_map_tool
│       ├── thematic_map.py      # Publication-quality matplotlib maps
│       ├── interactive_map.py   # Folium HTML maps
│       └── lc_tools.py
├── tracking/         # Observability: metrics, token tracking, debug logs
│   ├── metrics.py          # PipelineMetrics, StepRecord, track_step()
│   ├── token_tracker.py    # LLM token/cost accounting
│   └── callbacks.py        # LangChain callback handler
└── utils/
```

## Key Patterns & Conventions

### Tool Creation
Every agent-callable tool follows this pattern:
1. A **Pydantic `BaseModel`** defines the input schema (in `lc_tools.py`)
2. A **`@tool(args_schema=...)`** decorated function wraps the implementation
3. Each tool call is wrapped with **`track_step()`** for timing and metrics
4. Tools are registered in `_registry.py` via `get_all_tools()`

### Observability
All pipeline steps use the `track_step()` context manager:
```python
with track_step("scene_search", satellite="landsat") as step:
    results = client.search(...)
    step.set_metadata(scenes_found=len(results))
```
Records go to `data/logs/pipeline_metrics.jsonl`.

### Configuration
- All settings via `pydantic-settings` with `SPECTRAL_` env prefix
- Access with `get_settings()` (cached singleton)
- Credentials: USGS (`usgs_username`, `usgs_token`), Copernicus (`copernicus_client_id`, `copernicus_client_secret`), LLM keys

### Spectral Indices
Supported: **NDVI, EVI, SAVI, NDWI, NBR, NDBI**. Landsat Collection 2 Level-2 scale factors are applied automatically: `reflectance = DN × 0.0000275 − 0.2`. Index formulas are in `index_calculator.py`.

### Map Generation
- **Static PNG**: matplotlib with SVG north arrow, box-style scale bar, coordinate grid (4 sides), framed colorbar showing min/max
- **Interactive HTML**: Folium with ImageOverlay, satellite basemaps, CSS gradient legend
- Both generators read the same GeoTIFF and compute vmin/vmax from `valid.min()`/`valid.max()`

## Code Style

- **Type hints** on all function signatures and return types
- **Docstrings** in NumPy/Google style with Parameters/Returns sections
- **`from __future__ import annotations`** at the top of every module
- **Pydantic v2** models with `Field(description=...)` for all schema fields
- **`pathlib.Path`** everywhere — never raw string paths
- **f-strings** for formatting; `logging` module (not print) for output
- **No wildcard imports**; explicit imports only
- Prefer **`numpy`** vectorized operations over loops for raster data
- Error handling: raise domain exceptions from `core/exceptions.py`

## File Naming

- Tool wrappers: `lc_tools.py` (LangChain tools)
- Implementation: descriptive name (`landsat.py`, `thematic_map.py`, `index_calculator.py`)
- Schemas: Pydantic models in `schemas/` or co-located as `*Input` classes in `lc_tools.py`
- Tests: `tests/unit/test_<module>.py`

## Working with the Agent Graph

The main agent is built in `graphs/agent_graph.py`:
- Uses `create_react_agent()` from LangGraph with a `MemorySaver`
- System prompt defines the agent's persona, satellite selection rules, and temporal precision rules
- `AgentDeps` dataclass holds injectable dependencies (LLM, tools, session store)
- Thread-based memory via `config={"configurable": {"thread_id": "..."}}`

## Data Flow (End-to-End)

```
User prompt
  → Agent (LLM) interprets intent
    → search_landsat_tool (USGS M2M API)
      → download_landsat_tool (tar.gz extraction)
        → crop_landsat_bands_tool (rasterio clip to AOI)
          → compute_spectral_index_tool (NumPy math → GeoTIFF)
            → generate_thematic_map_tool (PNG + HTML)
```

## Important Notes

- The project has an `evaluation/` folder with an experimental plan and Excel workbook for thesis evaluation metrics (RMSE, IoU, success rate, speedup)
- Landsat search results are sorted by **date proximity to target** (not just cloud cover)
- The system prompt includes **temporal precision rules** that map user date expressions to start_date/end_date ranges
- AOI for experiments: `notebooks/data/raw/aoi_colombia.geojson`
