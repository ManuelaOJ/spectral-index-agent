"""
Layer 1 – NLP Extraction.

Uses an LLM with structured output to parse the user's natural language
request into an ExtractionResult.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from spectral_agent.schemas.spectral_request import ExtractionResult
from spectral_agent.tracking.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt for extraction
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You are a geospatial request parser.  Your ONLY job is to extract structured
information from a user's natural-language request about spectral indices.

Return a JSON object with EXACTLY these fields (use null when absent):

{
  "indices": ["NDVI"],          // list of spectral index names (NDVI, EVI, SAVI, NDWI, NBR, NDBI)
  "location_description": null, // string: city name or region described
  "geometry": null,             // object with type/coordinates/source if coords given
  "start_date": null,           // ISO date string if explicit
  "end_date": null,             // ISO date string if explicit
  "year": null,                 // integer if only one year given
  "years": null,                // list of integers for comparison
  "month": null,                // integer 1-12 if specific month
  "sensor_requested": null,     // e.g. "Landsat 8" only if user explicitly says
  "cloud_cover_max": null,      // float 0-100 only if user specifies
  "comparison_mode": false,     // true only if user wants same index across dates
  "raw_query": ""               // original user text (copy verbatim)
}

Rules:
- Only output valid JSON.  No markdown, no explanation.
- Only include indices that exist: NDVI, EVI, SAVI, NDWI, NBR, NDBI.

## Date interpretation (CRITICAL — follow exactly)
Always resolve the user's temporal reference into the tightest start_date /
end_date window.  Use the "year" field ONLY when no start/end can be derived.

Apply the FIRST matching rule:
1. Explicit date range with day precision ("entre el 14 de marzo y el 14 de abril de 2011",
   "between March 14 and April 14, 2011", "del 1 al 10 de abril de 2025")
   → start_date = first date, end_date = second date.
2. Month range ("entre enero y febrero de 2022", "jan to mar 2020")
   → start_date = 1st of first month, end_date = last day of last month.
3. Year range ("2018 to 2022", "2018 a 2022")
   → start_date = YYYY-01-01, end_date = YYYY-12-31.
4. Specific date ("2025-09-17", "el 4 de enero de 2015")
   → start_date = end_date = that date.
5. Month + year ("septiembre de 2014", "julio 2014", "sept 2025")
   → start_date = 1st of month, end_date = last day of month.
6. Standalone year ("2020", "año 2012", "in the year 2010")
   → start_date = YYYY-01-01, end_date = YYYY-12-31.
7. Comparison across years ("jan 2019 and jan 2023") → years=[2019,2023], month=1,
   comparison_mode=true. Leave start_date/end_date null.

- NEVER leave start_date/end_date as null when the user provides any temporal reference
  (except comparison mode).

## Coordinate parsing
- For coordinates like "lat X, lon Y" or "6.25, -75.56", create geometry
  with type="point", coordinates=[lon, lat], source="text".
- For a bounding box given as "west, south, east, north", type="bbox",
  coordinates=[[west, south],[east, north]].
- For a bounding box given as an array of two corner points like
  [lon1, lat1, lon2, lat2] or [-73.84, 6.88, -73.81, 6.91], interpret as
  two (longitude, latitude) pairs and compute:
  west=min(lon1,lon2), south=min(lat1,lat2), east=max(lon1,lon2), north=max(lat1,lat2).
  Then emit type="bbox", coordinates=[[west, south],[east, north]].
- For coordinates given as a pair of [lat, lon] points like
  [[lat1, lon1], [lat2, lon2]], detect by checking if values > 90 could be
  longitudes. Compute west/south/east/north the same way.
- Never invent data the user did not provide.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Extraction function
# ─────────────────────────────────────────────────────────────────────────────


async def extract_request(
    query: str,
    llm: BaseChatModel,
    tracker: TokenTracker,
    *,
    user: str = "system",
) -> ExtractionResult:
    """
    Parse a natural-language spectral index request via LLM.

    Args:
        query: User's text request.
        llm: LangChain chat model (ChatOpenAI / ChatAnthropic).
        tracker: Token usage tracker.
        user: User identifier for logging.

    Returns:
        ExtractionResult with all extracted fields.
    """
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    response = await llm.ainvoke(messages)

    # Track tokens
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    provider = "openai" if "gpt" in str(model_name).lower() else "anthropic"
    tracker.record_from_response(
        response=response,
        model=model_name,
        provider=provider,
        operation="extraction",
        user=user,
        metadata={"query_length": len(query)},
    )

    # Parse LLM output
    raw_text = response.content.strip()
    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON: %s\nRaw: %s", exc, raw_text)
        raise ValueError(f"LLM extraction returned invalid JSON: {exc}") from exc

    # Ensure raw_query is set
    data["raw_query"] = query

    return ExtractionResult(**data)
