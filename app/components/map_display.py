"""
Map display component — shows the latest generated maps and offers downloads.

Renders:
* A Folium interactive map (via ``streamlit-folium``) if an HTML artefact
  is available.
* A static PNG thumbnail (via ``st.image``) as fallback or alongside.
* Download buttons for both artefacts.
* A styled placeholder when no map has been generated yet.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.styles.theme import MAP_PLACEHOLDER_HTML


def render_map_display() -> None:
    """Show the latest map artefacts in the main content area."""

    png_path = st.session_state.get("last_static_png")
    html_path = st.session_state.get("last_interactive_html")
    tif_path = st.session_state.get("last_index_raster")

    if not png_path and not html_path:
        st.markdown(MAP_PLACEHOLDER_HTML, unsafe_allow_html=True)
        return

    # ── Interactive map (folium HTML) ────────────────────────────────────
    if html_path and Path(html_path).exists():
        st.markdown(
            '<p class="section-title">Mapa interactivo</p>',
            unsafe_allow_html=True,
        )
        html_bytes = Path(html_path).read_text(encoding="utf-8")
        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st.components.v1.html(html_bytes, height=460, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Static map (PNG) ─────────────────────────────────────────────────
    if png_path and Path(png_path).exists():
        st.markdown(
            '<p class="section-title">Mapa estático</p>',
            unsafe_allow_html=True,
        )
        st.image(str(png_path), use_container_width=True)

    # ── Download buttons ─────────────────────────────────────────────────
    dl_cols = st.columns(3)

    with dl_cols[0]:
        if png_path and Path(png_path).exists():
            data = Path(png_path).read_bytes()
            st.download_button(
                label="Descargar PNG",
                data=data,
                file_name=Path(png_path).name,
                mime="image/png",
                use_container_width=True,
            )

    with dl_cols[1]:
        if html_path and Path(html_path).exists():
            data = Path(html_path).read_bytes()
            st.download_button(
                label="Descargar HTML",
                data=data,
                file_name=Path(html_path).name,
                mime="text/html",
                use_container_width=True,
            )

    with dl_cols[2]:
        if tif_path and Path(tif_path).exists():
            data = Path(tif_path).read_bytes()
            st.download_button(
                label="🌍 Descargar GeoTIFF",
                data=data,
                file_name=Path(tif_path).name,
                mime="image/tiff",
                use_container_width=True,
            )
