"""Streamlit sidebar: brand, credentials, model picker, geometry upload, metrics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app.config import GEOMETRY_EXTENSIONS, reset_chat

# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    """Draw the full sidebar.  Call once per Streamlit rerun."""

    with st.sidebar:
        # ── Brand ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-brand">'
            '  <span class="brand-icon">&#x1F6F0;&#xFE0F;</span>'
            "  <span>"
            '    <span class="brand-text">Spectral Agent</span><br>'
            '    <span class="brand-version">v0.3 · Edición tesis</span>'
            "  </span>"
            "</div>",
            unsafe_allow_html=True,
        )

        _credentials_section()
        _model_section()
        _geometry_upload_section()
        _metrics_section()
        _actions_section()


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────


def _credentials_section() -> None:
    st.markdown('<div class="sidebar-section">Credenciales</div>', unsafe_allow_html=True)

    # Show a hint when keys are pre-filled from .env (dev mode)
    _any_prefilled = any(
        st.session_state.get(k) for k in ("openai_api_key", "usgs_username", "copernicus_client_id")
    )
    if _any_prefilled:
        st.caption("Claves cargadas desde `.env`")

    with st.expander("USGS  ·  Landsat", expanded=False):
        st.session_state["usgs_username"] = st.text_input(
            "Usuario",
            value=st.session_state.get("usgs_username", ""),
            key="input_usgs_user",
        )
        st.session_state["usgs_token"] = st.text_input(
            "Token",
            value=st.session_state.get("usgs_token", ""),
            type="password",
            key="input_usgs_token",
        )

    with st.expander("Copernicus  ·  Sentinel", expanded=False):
        st.session_state["copernicus_client_id"] = st.text_input(
            "Client ID",
            value=st.session_state.get("copernicus_client_id", ""),
            key="input_cop_id",
        )
        st.session_state["copernicus_client_secret"] = st.text_input(
            "Client Secret",
            value=st.session_state.get("copernicus_client_secret", ""),
            type="password",
            key="input_cop_secret",
        )

    with st.expander("Claves de API  ·  LLM", expanded=False):
        st.session_state["openai_api_key"] = st.text_input(
            "Clave de OpenAI",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            key="input_openai_key",
        )
        st.session_state["google_api_key"] = st.text_input(
            "Clave de Google  (Gemini)",
            value=st.session_state.get("google_api_key", ""),
            type="password",
            key="input_google_key",
        )


# Modelos ofrecidos por proveedor.  El primero de cada lista es el que se
# usa por defecto; para OpenAI y Google son los dos modelos evaluados en el
# documento (gpt-4o y gemini-2.5-flash).
_PROVIDERS = ["openai", "google"]
_MODELS_BY_PROVIDER: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "google": ["gemini-2.5-flash"],
    # El agente tambien soporta Anthropic
    # (src/spectral_agent/graphs/agent_graph.py).  No se ofrece aqui
    # porque ningun modelo de ese proveedor se evaluo en el documento;
    # para reactivarlo basta agregarlo a _PROVIDERS y listar sus modelos.
}


def _model_section() -> None:
    st.markdown('<div class="sidebar-section">Modelo</div>', unsafe_allow_html=True)

    current_provider = st.session_state.get("llm_provider", "openai")
    provider = st.selectbox(
        "Proveedor",
        options=_PROVIDERS,
        index=_PROVIDERS.index(current_provider) if current_provider in _PROVIDERS else 0,
        key="select_provider",
        label_visibility="collapsed",
    )
    st.session_state["llm_provider"] = provider

    # ``index`` respeta lo que venga de .env en el primer render.  Sin el,
    # el selector se queda en la opcion 0 y sobrescribe la configuracion.
    options = _MODELS_BY_PROVIDER.get(provider, ["gpt-4o"])
    current_model = st.session_state.get("llm_model")
    model = st.selectbox(
        "Modelo",
        options=options,
        index=options.index(current_model) if current_model in options else 0,
        # Una clave por proveedor: al cambiar de proveedor el selector no
        # arrastra un modelo que no pertenece a la lista nueva.
        key=f"select_model_{provider}",
        label_visibility="collapsed",
    )
    st.session_state["llm_model"] = model


def _geometry_upload_section() -> None:
    st.markdown('<div class="sidebar-section">Área de interés</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Cargue el área de interés (KML / SHP / GeoJSON)",
        type=GEOMETRY_EXTENSIONS,
        key="geo_upload",
        label_visibility="collapsed",
        accept_multiple_files=True,
        help="Para un shapefile seleccione todos los archivos juntos (.shp, .shx, .dbf, .prj).",
    )

    if uploaded:
        _handle_geometry_upload(uploaded)
    elif st.session_state.get("uploaded_geometry_path"):
        fname = Path(st.session_state["uploaded_geometry_path"]).name
        st.caption(f"📎 {fname}")


# Shapefile companion extensions required / optional
_SHP_REQUIRED = {".shp", ".shx", ".dbf"}
_SHP_OPTIONAL = {".prj", ".cpg"}
_SHP_ALL = _SHP_REQUIRED | _SHP_OPTIONAL


def _handle_geometry_upload(files: list) -> None:
    """Process one or more uploaded geometry files."""
    # Build a map of extension → uploaded file
    by_ext: dict[str, object] = {}
    for f in files:
        ext = Path(f.name).suffix.lower()
        by_ext[ext] = f

    has_shp = ".shp" in by_ext

    if has_shp:
        # ── Shapefile bundle: write all parts and verify ─────────────
        tmp_dir = Path(tempfile.mkdtemp(prefix="spectral_aoi_"))
        stem = Path(by_ext[".shp"].name).stem

        for ext, fobj in by_ext.items():
            if ext in _SHP_ALL:
                dest = tmp_dir / f"{stem}{ext}"
                dest.write_bytes(fobj.read())

        # Check required companions exist
        missing = [ext for ext in _SHP_REQUIRED if not (tmp_dir / f"{stem}{ext}").exists()]
        if missing:
            st.error(
                f"El shapefile está incompleto. Faltan: "
                f"**{', '.join(missing)}**. Cargue todos los archivos "
                f"juntos (.shp, .shx, .dbf y de preferencia .prj)."
            )
            return

        shp_path = str(tmp_dir / f"{stem}.shp")
        st.session_state["uploaded_geometry_path"] = shp_path

        extras = [ext for ext in sorted(by_ext) if ext in _SHP_ALL]
        st.success(
            f"Se cargó **{stem}.shp** con sus archivos: {', '.join(extras)}",
            icon="✅",
        )
        if ".prj" not in by_ext:
            st.warning("Sin archivo .prj. Se asume el sistema WGS84 (EPSG:4326).")
    else:
        # ── Single-file format (KML, GeoJSON) ────────────────────────
        f = files[0]
        suffix = Path(f.name).suffix.lower()
        tmp = Path(tempfile.gettempdir()) / f"spectral_aoi{suffix}"
        tmp.write_bytes(f.read())
        st.session_state["uploaded_geometry_path"] = str(tmp)
        st.success(f"Se cargó **{f.name}**", icon="✅")


def _metrics_section() -> None:
    """Live observability panel — compact chip-style layout."""
    st.markdown('<div class="sidebar-section">Proceso</div>', unsafe_allow_html=True)

    try:
        from spectral_agent.tracking import get_tracker
        from spectral_agent.tracking.metrics import get_metrics

        pm = get_metrics()
        summary = pm.summary()

        if summary["total_steps"] == 0:
            st.caption("Aún sin actividad.")
            return

        # ── KPI chips row ────────────────────────────────────────────────
        chips_html = '<div class="sidebar-metrics">'
        chips_html += f'<span class="sidebar-chip">{summary["total_steps"]} pasos</span>'
        if summary["successes"]:
            chips_html += f'<span class="sidebar-chip green">✓ {summary["successes"]}</span>'
        if summary["failures"]:
            chips_html += f'<span class="sidebar-chip red">✗ {summary["failures"]}</span>'
        chips_html += f'<span class="sidebar-chip">{summary["total_duration_s"]:.1f}s</span>'
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

        # ── Token cost chip ──────────────────────────────────────────────
        tracker = get_tracker()
        cost_summary = tracker.summary()
        if cost_summary["total_calls"] > 0:
            cost_html = (
                '<div class="sidebar-metrics" style="margin-top:4px">'
                f'<span class="sidebar-chip blue">'
                f"LLM {cost_summary['total_calls']} llamadas  ·  "
                f"${cost_summary['total_cost_usd']:.4f}</span>"
                "</div>"
            )
            st.markdown(cost_html, unsafe_allow_html=True)

        # ── Per-step detail (collapsed) ──────────────────────────────────
        with st.expander("Detalle por paso", expanded=False):
            for step_name, data in summary["by_step"].items():
                st.caption(
                    f"**{step_name}** · {data['calls']}× · "
                    f"avg {data['avg_s']:.2f}s · "
                    f"✓{data['successes']} ✗{data['failures']}"
                )

    except Exception:
        st.caption("Las métricas aparecerán cuando el agente ejecute el primer paso.")


def _actions_section() -> None:
    st.markdown('<div class="sidebar-section">Acciones</div>', unsafe_allow_html=True)

    if st.button("🔄 Nueva sesión", use_container_width=True, type="secondary"):
        reset_chat()
        st.rerun()
