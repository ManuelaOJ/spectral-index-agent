# Spectral Index Agent

An LLM agent that turns a natural-language request into spectral index maps
computed from Landsat and Sentinel-2 imagery.

The user describes what they need, for example *"NDVI for this area in January
2024 with little cloud cover"*, and uploads an area of interest. The agent
searches the satellite catalogues, downloads the scene, clips it to the area of
interest, computes the requested index and renders the map.

The code accompanies the master's thesis cited [below](#citation).

---

## What it does

- **Agent**: a ReAct agent built with LangChain and LangGraph
  (`create_react_agent`) with conversational memory (`MemorySaver`
  checkpointer). The LLM decides which tool to call at each step.
  Supported LLM providers: OpenAI, Anthropic and Google (Gemini).
- **Landsat (Landsat 4-9, Collection 2 Level-2)**: scene search and download
  through the USGS M2M API, band clipping to the area of interest, and local
  index computation with NumPy and rasterio.
- **Sentinel-2 (L2A)**: scene search through the Sentinel Hub catalogue of the
  Copernicus Data Space Ecosystem; indices are computed server side with
  Sentinel Hub and downloaded as a GeoTIFF clipped to the area of interest.
- **Spectral indices**: NDVI, EVI, SAVI, NDWI, NBR and NDBI.
- **Maps**: static PNG maps (matplotlib, with colour bar, scale bar and north
  arrow) and interactive HTML maps (folium).
- **Area of interest**: KML, Shapefile or GeoJSON, with CRS detection and
  reprojection.
- **User interface**: a Streamlit chat application (in Spanish) with a
  sidebar for credentials, LLM settings and the area-of-interest upload.

### Agent tools

| Tool | Description |
|------|-------------|
| `search_landsat_tool` | Search USGS for Landsat scenes |
| `download_landsat_tool` | Download a Landsat scene |
| `search_sentinel_tool` | Search for Sentinel-2 scenes |
| `download_sentinel_tool` | Download a Sentinel-2 product |
| `download_sentinel_index_tool` | Compute indices on Sentinel Hub and download the GeoTIFF |
| `search_satellite_imagery_tool` | Unified Landsat + Sentinel-2 search |
| `crop_landsat_bands_tool` | Clip Landsat bands to the area of interest |
| `compute_spectral_index_tool` | Compute spectral indices from clipped bands |
| `list_cached_bands_tool` | List bands already processed in the cache |
| `list_available_indices_tool` | List available indices and their required bands |
| `generate_thematic_map_tool` | Render static PNG and interactive HTML maps |

All tools are collected in `spectral_agent.tools._registry.get_all_tools()`.

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/). `uv.lock` pins every
transitive dependency, so all machines resolve to the same environment.
Python 3.11 or later is required (uv downloads it if missing).

```bash
# TODO: replace <PUBLIC_REPO_URL> once the public repository exists
git clone <PUBLIC_REPO_URL> spectral-index-agent
cd spectral-index-agent

uv sync --extra ui     # agent + Streamlit interface
uv sync --extra dev    # agent + test and lint tooling
uv sync --all-extras   # everything (dev, ui, notebooks)
```

Without uv, `pip install -e ".[ui,dev]"` also works, but it resolves
dependencies fresh and ignores `uv.lock`.

## Credentials

Copy the template and fill in your own keys:

```bash
cp .env.example .env      # Linux / macOS
copy .env.example .env    # Windows
```

| Variable | Needed for |
|----------|-----------|
| `SPECTRAL_USGS_USERNAME`, `SPECTRAL_USGS_TOKEN` | Landsat (USGS M2M token, register at <https://ers.cr.usgs.gov/register>) |
| `SPECTRAL_COPERNICUS_CLIENT_ID`, `SPECTRAL_COPERNICUS_CLIENT_SECRET` | Sentinel-2 (OAuth client from <https://dataspace.copernicus.eu/>) |
| `SPECTRAL_OPENAI_API_KEY` | OpenAI models |
| `SPECTRAL_ANTHROPIC_API_KEY` | Anthropic models |
| `GOOGLE_API_KEY` | Google Gemini models (note: no `SPECTRAL_` prefix) |

At least one LLM key is required. `.env.example` also lists the optional
settings (default provider and model, temperature, data paths, LangSmith
tracing). The Streamlit sidebar also lets you enter credentials for the
current session.

## Usage

### Streamlit app

```bash
uv run streamlit run app/main.py
```

### From Python

```python
from langchain_core.messages import HumanMessage
from spectral_agent.graphs import AgentDeps, build_agent, make_thread_config

agent = build_agent(AgentDeps.from_settings())   # provider and model from .env
result = agent.invoke(
    {"messages": [HumanMessage(content="Compute NDVI with Sentinel-2 for "
                               "bbox -75.60,6.20,-75.55,6.25 in January 2024")]},
    config=make_thread_config("session-1"),
)
print(result["messages"][-1].content)
```

API keys and satellite credentials can be passed as `AgentDeps` fields
(`openai_api_key`, `usgs_token`, `copernicus_client_id`, ...). Outputs
(raw scenes, clipped bands, index GeoTIFFs and maps) are written under
`data/raw/<session_id>/` and `data/processed/<session_id>/`, which are
git-ignored.

## Tests

```bash
uv sync --extra dev
uv run pytest -q
```

The unit tests do not call the satellite APIs or the LLM providers and need
no credentials.

## Project structure

```
src/spectral_agent/
├── agents/          # LLM factory and agent helpers
├── config/          # pydantic-settings (SPECTRAL_ prefix)
├── core/            # base classes and exceptions
├── graphs/          # LangGraph agent (agent_graph.py) and ingestion workflow
├── memory/          # session store for generated artefacts
├── pipeline/        # request extraction, geometry input, normalization, rules
├── schemas/         # Pydantic models
├── tools/           # LangChain tools: ingestion, raster, indices, visualization
├── tracking/        # step metrics, token and cost tracking, conversation logs
└── utils/
app/                 # Streamlit interface
tests/               # unit tests
docs/                # architecture notes
```

## Documentation

- [Architecture](docs/architecture.md)
- [Pipeline architecture](docs/pipeline_architecture.md)
- [Landsat imagery notes](docs/landsat_imagery.md)

## Citation

If you use this code, please cite the thesis:

> Orozco Jiménez, M. (2026). *Desarrollo de un agente de inteligencia
> artificial para la generación de índices espectrales a partir de imágenes
> satelitales* [Master's thesis, Maestría en Ingeniería Analítica].
> Universidad Nacional de Colombia, sede Medellín.

A machine-readable version is in [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgments

- [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- [USGS](https://www.usgs.gov/) for Landsat data
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) and Sentinel Hub for Sentinel-2 data
