"""
Spectral Indices definitions and evalscript generation for Sentinel Hub.

This module defines spectral indices that can be calculated directly
in Sentinel Hub using evalscripts, returning computed indices instead
of raw bands. This is more efficient as it reduces data transfer.

Reference: https://www.indexdatabase.de/
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SpectralIndex:
    """Definition of a spectral index."""

    name: str
    """Short name (e.g., 'NDVI')"""

    full_name: str
    """Full descriptive name"""

    formula: str
    """JavaScript formula for evalscript (uses sample.BXX notation)"""

    bands: list[str]
    """Required Sentinel-2 bands"""

    value_range: tuple[float, float]
    """Expected output value range (min, max)"""

    description: str
    """What the index measures"""

    category: Literal[
        "vegetation", "water", "soil", "burn", "snow", "built_up", "other"
    ]
    """Index category"""

    preamble: str = ""
    """Optional JavaScript variable declarations prepended in the evalscript
    (e.g. 'const L = 0.5;' for soil brightness correction factor)"""

    parameters: dict[str, float] | None = None
    """Named parameters used by the formula (e.g. {'L': 0.5} for SAVI)"""


# Spectral indices catalog for Sentinel-2
SPECTRAL_INDICES: dict[str, SpectralIndex] = {
    # ==================== VEGETATION INDICES ====================
    "NDVI": SpectralIndex(
        name="NDVI",
        full_name="Normalized Difference Vegetation Index",
        formula="(sample.B08 - sample.B04) / (sample.B08 + sample.B04)",
        bands=["B04", "B08"],
        value_range=(-1.0, 1.0),
        description="Most common vegetation index. High values indicate healthy vegetation.",
        category="vegetation",
    ),
    "EVI": SpectralIndex(
        name="EVI",
        full_name="Enhanced Vegetation Index",
        formula="2.5 * (sample.B08 - sample.B04) / (sample.B08 + 6 * sample.B04 - 7.5 * sample.B02 + 1)",
        bands=["B02", "B04", "B08"],
        value_range=(-1.0, 1.0),
        description="Improved vegetation index that corrects for atmospheric and canopy background noise.",
        category="vegetation",
    ),
    "SAVI": SpectralIndex(
        name="SAVI",
        full_name="Soil Adjusted Vegetation Index",
        formula="(sample.B08 - sample.B04) / (sample.B08 + sample.B04 + L) * (1.0 + L)",
        bands=["B04", "B08"],
        value_range=(-1.5, 1.5),
        description=(
            "Vegetation index that minimizes soil brightness influences. "
            "Uses soil brightness correction factor L=0.5 to accommodate "
            "most land cover types. "
            "Landsat 4-7: ((Band4 - Band3) / (Band4 + Band3 + 0.5)) * 1.5; "
            "Landsat 8-9: ((Band5 - Band4) / (Band5 + Band4 + 0.5)) * 1.5; "
            "Sentinel-2: ((B08 - B04) / (B08 + B04 + L)) * (1 + L)."
        ),
        category="vegetation",
        preamble="const L = 0.5;",
        parameters={"L": 0.5},
    ),
    # ==================== WATER INDICES ====================
    "NDWI": SpectralIndex(
        name="NDWI",
        full_name="Normalized Difference Water Index",
        formula="(sample.B03 - sample.B08) / (sample.B03 + sample.B08)",
        bands=["B03", "B08"],
        value_range=(-1.0, 1.0),
        description="Detects water bodies. Positive values indicate water.",
        category="water",
    ),
    # ==================== BURN INDICES ====================
    "NBR": SpectralIndex(
        name="NBR",
        full_name="Normalized Burn Ratio",
        formula="(sample.B08 - sample.B12) / (sample.B08 + sample.B12)",
        bands=["B08", "B12"],
        value_range=(-1.0, 1.0),
        description="Identifies burned areas. Low/negative values indicate burns.",
        category="burn",
    ),
    # ==================== BUILT-UP INDICES ====================
    "NDBI": SpectralIndex(
        name="NDBI",
        full_name="Normalized Difference Built-up Index",
        formula="(sample.B11 - sample.B08) / (sample.B11 + sample.B08)",
        bands=["B08", "B11"],
        value_range=(-1.0, 1.0),
        description="Identifies built-up/urban areas. Positive values indicate buildings.",
        category="built_up",
    ),
}


def get_index(name: str) -> SpectralIndex:
    """
    Get a spectral index by name.

    Args:
        name: Index name (case-insensitive)

    Returns:
        SpectralIndex definition

    Raises:
        ValueError: If index name is not found
    """
    name_upper = name.upper()
    if name_upper not in SPECTRAL_INDICES:
        available = ", ".join(sorted(SPECTRAL_INDICES.keys()))
        raise ValueError(f"Unknown index '{name}'. Available: {available}")
    return SPECTRAL_INDICES[name_upper]


def list_indices(category: str | None = None) -> list[SpectralIndex]:
    """
    List all available spectral indices.

    Args:
        category: Optional filter by category (vegetation, water, soil, burn, snow, built_up, other)

    Returns:
        List of SpectralIndex objects
    """
    indices = list(SPECTRAL_INDICES.values())
    if category:
        indices = [idx for idx in indices if idx.category == category]
    return indices


def generate_evalscript(indices: list[str] | str) -> str:
    """
    Generate Sentinel Hub evalscript for calculating spectral indices.

    Args:
        indices: Single index name or list of index names to calculate

    Returns:
        Evalscript string for Sentinel Hub Process API

    Example:
        >>> evalscript = generate_evalscript(["NDVI", "NDWI"])
        >>> # Returns evalscript that calculates both indices
    """
    if isinstance(indices, str):
        indices = [indices]

    # Get index definitions
    index_defs = [get_index(name) for name in indices]

    # Collect all required bands (deduplicated)
    all_bands = set()
    for idx in index_defs:
        all_bands.update(idx.bands)
    bands_list = sorted(all_bands)

    # Build bands array for input
    bands_str = ", ".join(f'"{b}"' for b in bands_list)

    # Build preamble constants and calculations
    preamble_lines = []
    seen_preambles: set[str] = set()
    for idx in index_defs:
        if idx.preamble and idx.preamble not in seen_preambles:
            preamble_lines.append(f"    {idx.preamble}")
            seen_preambles.add(idx.preamble)

    calc_lines = []
    return_values = []
    for i, idx in enumerate(index_defs):
        var_name = f"idx_{i}"
        calc_lines.append(f"    let {var_name} = {idx.formula};")
        return_values.append(var_name)

    preamble_block = "\n".join(preamble_lines)
    calculations = "\n".join(calc_lines)
    body_lines = "\n".join(filter(None, [preamble_block, calculations]))
    return_str = ", ".join(return_values)

    # Build parameter comment
    param_comments = []
    for idx in index_defs:
        if idx.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in idx.parameters.items())
            param_comments.append(f"// {idx.name} parameters: {param_str}")
    param_comment_block = "\n".join(param_comments)
    if param_comment_block:
        param_comment_block = param_comment_block + "\n"

    return f"""//VERSION=3
// Spectral Indices: {", ".join(indices)}
// Generated by spectral_agent
{param_comment_block}
function setup() {{
    return {{
        input: [{{
            bands: [{bands_str}],
            units: "REFLECTANCE"
        }}],
        output: {{
            bands: {len(indices)},
            sampleType: "FLOAT32"
        }}
    }};
}}

function evaluatePixel(sample) {{
{body_lines}
    return [{return_str}];
}}
"""


def generate_evalscript_with_rgb(indices: list[str] | str) -> str:
    """
    Generate evalscript that returns indices plus RGB visualization.

    First 3 bands are RGB (B04, B03, B02), followed by indices.

    Args:
        indices: Single index name or list of index names

    Returns:
        Evalscript string
    """
    if isinstance(indices, str):
        indices = [indices]

    index_defs = [get_index(name) for name in indices]

    # Collect all required bands plus RGB
    all_bands = {"B02", "B03", "B04"}  # RGB
    for idx in index_defs:
        all_bands.update(idx.bands)
    bands_list = sorted(all_bands)

    bands_str = ", ".join(f'"{b}"' for b in bands_list)

    # Build preamble constants and calculations
    preamble_lines = []
    seen_preambles: set[str] = set()
    for idx in index_defs:
        if idx.preamble and idx.preamble not in seen_preambles:
            preamble_lines.append(f"    {idx.preamble}")
            seen_preambles.add(idx.preamble)

    calc_lines = []
    return_values = [
        "sample.B04 * 2.5",
        "sample.B03 * 2.5",
        "sample.B02 * 2.5",
    ]  # RGB scaled
    for i, idx in enumerate(index_defs):
        var_name = f"idx_{i}"
        calc_lines.append(f"    let {var_name} = {idx.formula};")
        return_values.append(var_name)

    preamble_block = "\n".join(preamble_lines)
    calculations = "\n".join(calc_lines) if calc_lines else ""
    body_lines = "\n".join(filter(None, [preamble_block, calculations]))
    return_str = ", ".join(return_values)

    return f"""//VERSION=3
// RGB + Spectral Indices: {", ".join(indices)}
// Bands: 0=Red, 1=Green, 2=Blue, then indices
// Generated by spectral_agent

function setup() {{
    return {{
        input: [{{
            bands: [{bands_str}],
            units: "REFLECTANCE"
        }}],
        output: {{
            bands: {3 + len(indices)},
            sampleType: "FLOAT32"
        }}
    }};
}}

function evaluatePixel(sample) {{
{body_lines}
    return [{return_str}];
}}
"""
