"""
Chat component — modern LLM-style conversational interface.

Features:
* Scrollable message list with chat bubbles (Streamlit ``chat_message``).
* Animated typing indicator while the agent is thinking.
* Welcome card with example prompts on first visit.
* Inline map preview + download buttons rendered inside assistant messages.
* Agent is built lazily on the first user message.
* No business logic here — orchestration stays in LangGraph.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.styles.theme import TYPING_INDICATOR_HTML, WELCOME_HTML

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar context injection
# ─────────────────────────────────────────────────────────────────────────────


def _build_context_message(user_text: str) -> str:
    """Enrich the user message with sidebar context (uploaded geometry, indices).

    If the user uploaded a geometry file in the sidebar, the bounding box and
    area are extracted and appended so the agent can act on them.  Selected
    spectral indices are also injected.
    """
    parts: list[str] = [user_text]

    # ── Uploaded geometry context ────────────────────────────────────────
    geo_path = st.session_state.get("uploaded_geometry_path")
    if geo_path:
        try:
            from pathlib import Path as _P

            from spectral_agent.pipeline.geo_input import (
                geometry_area_km2,
                geometry_to_bbox,
                read_geometry_file,
            )

            geom = read_geometry_file(geo_path)
            bbox = geometry_to_bbox(geom)
            area = geometry_area_km2(geom)
            fname = _P(geo_path).name

            parts.append(
                f"\n[Uploaded geometry file: {fname}]\n"
                f"Bounding box (WGS84): "
                f"west={bbox['west']:.6f}, south={bbox['south']:.6f}, "
                f"east={bbox['east']:.6f}, north={bbox['north']:.6f}\n"
                f"Approximate area: {area:.2f} km²\n"
                f"Geometry type: {geom.type.value}"
            )
        except Exception as exc:
            logger.warning("Failed to parse uploaded geometry: %s", exc)
            parts.append(f"\n[Uploaded geometry file could not be parsed: {exc}]")

    # ── Selected spectral indices ────────────────────────────────────────
    selected = st.session_state.get("selected_indices", [])
    if selected:
        parts.append(f"\n[Selected spectral indices: {', '.join(selected)}]")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Agent bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_agent() -> None:
    """Lazily create the agent + thread_id on the first invocation."""
    if st.session_state.get("agent") is not None:
        return

    from spectral_agent.graphs.agent_graph import AgentDeps, build_agent

    deps = AgentDeps(
        provider=st.session_state.get("llm_provider", "openai"),
        # gpt-4o es el modelo que evaluo el documento.  El valor solo actua si
        # la sesion llegara sin modelo, cosa que app/config.py ya evita, pero
        # dejarlo en gpt-4o-mini era una trampa: fue el modelo que termino
        # corriendo por error en las capturas del anexo.
        model=st.session_state.get("llm_model", "gpt-4o"),
        openai_api_key=st.session_state.get("openai_api_key", ""),
        anthropic_api_key=st.session_state.get("anthropic_api_key", ""),
        usgs_username=st.session_state.get("usgs_username", ""),
        usgs_token=st.session_state.get("usgs_token", ""),
        copernicus_client_id=st.session_state.get("copernicus_client_id", ""),
        copernicus_client_secret=st.session_state.get("copernicus_client_secret", ""),
    )
    st.session_state["agent"] = build_agent(deps)

    # thread_id is already assigned by init_session_state(); reuse it.
    # Register session ID with all tracking singletons so every log entry
    # in this session is tagged automatically.
    tid = st.session_state["thread_id"]

    from spectral_agent.tracking.callbacks import get_cost_handler
    from spectral_agent.tracking.conversation_logger import get_conversation_logger
    from spectral_agent.tracking.debug_log import get_debug_logger

    get_debug_logger().set_session_id(tid)
    get_cost_handler().set_session_id(tid)
    get_conversation_logger().set_session_id(tid)

    logger.info(
        "Agent created: %s/%s  thread=%s",
        deps.provider,
        deps.model,
        st.session_state["thread_id"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Async helper
# ─────────────────────────────────────────────────────────────────────────────


def _run_async(coro: Any) -> Any:
    """Run a coroutine from synchronous Streamlit context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint repair
# ─────────────────────────────────────────────────────────────────────────────


def _repair_orphan_tool_calls(agent: Any, config: dict[str, Any]) -> None:
    """Inject synthetic ToolMessages for orphaned tool_calls in the checkpoint.

    When a tool node crashes mid-execution, the checkpoint retains the
    AIMessage with ``tool_calls`` but no corresponding ``ToolMessage``.
    On the next invocation LangGraph raises ``INVALID_CHAT_HISTORY``.
    This function detects and patches such orphans *before* invoking the
    agent so the conversation can continue normally.
    """
    try:
        state = agent.get_state(config)
    except Exception:
        return  # first invocation — no checkpoint yet

    messages = state.values.get("messages") if state.values else None
    if not messages:
        return

    # Collect every tool_call_id that already has a ToolMessage reply
    answered_ids: set[str] = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}

    # Walk backward — only the most-recent AIMessage can be orphaned
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            patches: list[ToolMessage] = []
            for tc in msg.tool_calls:
                if tc["id"] not in answered_ids:
                    patches.append(
                        ToolMessage(
                            content=(
                                f"Error: tool '{tc['name']}' crashed in a "
                                "previous turn. Please retry or choose "
                                "another approach."
                            ),
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
            if patches:
                agent.update_state(config, {"messages": patches})
                logger.warning(
                    "Repaired %d orphaned tool_call(s) in thread %s",
                    len(patches),
                    config["configurable"]["thread_id"],
                )
            return

        # If the last meaningful message is already a ToolMessage or
        # HumanMessage, nothing is broken.
        if isinstance(msg, ToolMessage | HumanMessage):
            return


# ─────────────────────────────────────────────────────────────────────────────
# Agent invocation
# ─────────────────────────────────────────────────────────────────────────────


# Maximum number of LangGraph super-steps per invocation.  The Landsat
# pipeline needs ~11 steps (5 tools) and retries / extra look-ups can
# push it higher, so 50 gives plenty of headroom.
_RECURSION_LIMIT: int = 50


def _log_error_turn(user_text: str, exc: Exception) -> None:
    """Log a failed conversation turn so errors are also captured."""
    try:
        from spectral_agent.tracking.conversation_logger import get_conversation_logger

        enriched = _build_context_message(user_text)
        get_conversation_logger().log_turn(
            user_prompt=user_text,
            enriched_prompt=enriched,
            agent_response=f"Error: {exc}",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception:
        logger.debug("Could not log error turn", exc_info=True)


def _invoke_agent(user_text: str) -> str:
    """Send *user_text* to the agent and return the assistant reply.

    The raw user text is enriched with sidebar context (uploaded geometry,
    selected indices) before being sent to the agent so it has the full
    picture without requiring the user to retype coordinates.

    Each invocation is logged to ``data/logs/conversations.jsonl`` with
    the original prompt, enriched prompt, response, tool calls, token
    usage, cost, and duration for thesis reproducibility.
    """
    _ensure_agent()

    enriched = _build_context_message(user_text)

    agent = st.session_state["agent"]
    thread_id = st.session_state["thread_id"]

    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _RECURSION_LIMIT,
    }

    # Repair orphaned tool_calls left by a previous crash
    _repair_orphan_tool_calls(agent, config)

    # Snapshot token tracker before invocation to compute delta
    from spectral_agent.tracking.token_tracker import get_tracker

    tracker = get_tracker()
    tokens_before = len(tracker.records)

    t0 = time.perf_counter()

    result = _run_async(
        agent.ainvoke(
            {"messages": [HumanMessage(content=enriched)]},
            config=config,
        )
    )

    duration_s = time.perf_counter() - t0

    # Extract the last AI message
    messages = result.get("messages", [])

    # Collect tool calls and log debug info
    tool_calls_log: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            logger.debug(
                "ToolMessage [%s]: %s",
                getattr(msg, "name", "?"),
                content[:2000],
            )
            _extract_map_artefacts(content)
        elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                logger.debug(
                    "AIMessage tool_call: %s(%s)",
                    tc.get("name"),
                    tc.get("args"),
                )
                tool_calls_log.append(
                    {
                        "tool": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    }
                )

    # Match tool call entries with their ToolMessage results
    tool_msg_iter = iter(msg for msg in messages if isinstance(msg, ToolMessage))
    for entry in tool_calls_log:
        tool_msg = next(tool_msg_iter, None)
        if tool_msg is not None:
            raw = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content)
            entry["result_summary"] = raw[:500]

    # Extract reply text
    reply = "(no response)"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            reply = msg.content
            break

    # Compute token/cost delta for this turn
    new_records = tracker.records[tokens_before:]
    turn_input_tokens = sum(r.input_tokens for r in new_records)
    turn_output_tokens = sum(r.output_tokens for r in new_records)
    turn_cost = sum(r.total_cost_usd for r in new_records)
    turn_model = new_records[-1].model if new_records else ""
    turn_provider = new_records[-1].provider if new_records else ""

    # Log conversation turn
    from spectral_agent.tracking.conversation_logger import get_conversation_logger

    get_conversation_logger().log_turn(
        user_prompt=user_text,
        enriched_prompt=enriched,
        agent_response=reply,
        tool_calls=tool_calls_log,
        model=turn_model,
        provider=turn_provider,
        input_tokens=turn_input_tokens,
        output_tokens=turn_output_tokens,
        total_cost_usd=round(turn_cost, 8),
        duration_s=duration_s,
    )

    return reply


# ─────────────────────────────────────────────────────────────────────────────
# Map artefact extraction & inline rendering
# ─────────────────────────────────────────────────────────────────────────────


_FILE_RE = re.compile(r"([\w/\\:.~\-\s]+\.(?:png|html|tiff|tif))", re.IGNORECASE)


# Raiz del repositorio, para resolver rutas relativas de los artefactos.
_RAIZ = Path(__file__).resolve().parents[2]


def _resolver_ruta(texto: str) -> Path | None:
    """Convierte una ruta escrita por el modelo en un archivo real, o None.

    El modelo suele anteponer `sandbox:/mnt/`, una convencion del entorno de
    ChatGPT que aqui no corresponde a nada: lo que escribe como
    `sandbox:/mnt/data/processed/...` esta en disco en `<repo>/data/processed/...`.
    Se retiran esos prefijos y se prueba la ruta tal cual, luego relativa al
    directorio de trabajo y por ultimo relativa a la raiz del repositorio.
    """
    if not texto:
        return None

    candidato = texto.strip()
    for prefijo in ("sandbox:", "file://"):
        if candidato.lower().startswith(prefijo):
            candidato = candidato[len(prefijo) :]
    for prefijo in ("/mnt/", "mnt/"):
        if candidato.lower().startswith(prefijo):
            candidato = candidato[len(prefijo) :]
            break
    candidato = candidato.lstrip("/")

    for ruta in (Path(texto.strip()), Path(candidato), _RAIZ / candidato):
        try:
            if ruta.exists() and ruta.is_file():
                return ruta
        except OSError:
            # Rutas con caracteres que Windows no admite.
            continue
    return None


def _extract_map_artefacts(text: str) -> None:
    """Scan agent reply for file paths and update session_state map artefacts."""
    # Se revisan todas las coincidencias, no solo la primera: la respuesta
    # puede nombrar varios archivos y solo alguno existir en disco.
    for clave, patron in (
        ("last_static_png", r"([\w/\\:.~\-]+\.png)"),
        ("last_interactive_html", r"([\w/\\:.~\-]+\.html)"),
        ("last_index_raster", r"([\w/\\:.~\-]+\.(?:tiff|tif))"),
    ):
        for bruta in re.findall(patron, text, re.IGNORECASE):
            ruta = _resolver_ruta(bruta)
            if ruta is not None:
                st.session_state[clave] = str(ruta)
                break


def _render_inline_artefacts(text: str, usar_respaldo: bool = True) -> None:
    """If the reply mentions a generated file, show inline preview + download.

    First tries regex on the reply text; falls back to session_state
    artefacts previously extracted from ToolMessages.

    ``usar_respaldo`` desactiva ese respaldo. Se apaga para los mensajes que
    no son el ultimo del agente: los artefactos de la sesion son los de la
    corrida mas reciente y no pertenecen a las respuestas anteriores.
    """

    def resolver(patron: str, clave: str) -> Path | None:
        """Primero lo que nombra la respuesta, luego lo que dejaron las
        herramientas en la sesion."""
        for bruta in re.findall(patron, text, re.IGNORECASE):
            ruta = _resolver_ruta(bruta)
            if ruta is not None:
                return ruta
        if not usar_respaldo:
            return None
        return _resolver_ruta(st.session_state.get(clave, ""))

    png_path = resolver(r"([\w/\\:.~\-]+\.png)", "last_static_png")
    html_path = resolver(r"([\w/\\:.~\-]+\.html)", "last_interactive_html")
    tif_path = resolver(r"([\w/\\:.~\-]+\.(?:tiff|tif))", "last_index_raster")

    # ── Inline PNG preview ───────────────────────────────────────────────
    if png_path:
        try:
            b64 = base64.b64encode(png_path.read_bytes()).decode()
            st.markdown(
                f'<div class="chat-map-preview">'
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="Mapa temático" /></div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Descargar PNG",
                data=png_path.read_bytes(),
                file_name=png_path.name,
                mime="image/png",
                key=f"dl_png_{png_path.stem}_{id(text)}",
            )
        except Exception as exc:
            logger.debug("Could not render inline PNG: %s", exc)

    # ── Interactive HTML download ────────────────────────────────────────
    if html_path:
        try:
            st.download_button(
                label="⬇️ Descargar mapa interactivo (HTML)",
                data=html_path.read_bytes(),
                file_name=html_path.name,
                mime="text/html",
                key=f"dl_html_{html_path.stem}_{id(text)}",
            )
        except Exception as exc:
            logger.debug("Could not offer HTML download: %s", exc)

    # ── GeoTIFF download ─────────────────────────────────────────────────
    if tif_path:
        try:
            st.download_button(
                label="🌍 Descargar GeoTIFF",
                data=tif_path.read_bytes(),
                file_name=tif_path.name,
                mime="image/tiff",
                key=f"dl_tif_{tif_path.stem}_{id(text)}",
            )
        except Exception as exc:
            logger.debug("Could not offer GeoTIFF download: %s", exc)


def _bajar_al_ultimo_mensaje() -> None:
    """Deja a la vista el mensaje mas reciente de la conversacion.

    ``st.markdown`` no ejecuta JavaScript, de modo que el salto se hace desde
    un componente en iframe, que si lo ejecuta y, por ser del mismo origen,
    puede alcanzar el documento de la pagina.  Se usa ``scrollIntoView`` sobre
    la ultima burbuja para no depender de que contenedor sea el que desplaza.
    """
    components.html(
        """
        <script>
        const doc = window.parent.document;
        window.setTimeout(function () {
            const burbujas = doc.querySelectorAll('[data-testid="stChatMessage"]');
            if (burbujas.length) {
                burbujas[burbujas.length - 1].scrollIntoView(
                    {block: "end", behavior: "smooth"});
            }
        }, 250);
        </script>
        """,
        height=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────


# Marca invisible que permite al CSS distinguir la burbuja del usuario de la
# del agente.  Streamlit no expone el rol del mensaje en el DOM.
_MARCA_USUARIO = '<span class="marca-usuario"></span>'


def _marcar_usuario() -> None:
    """Inyecta la marca de rol dentro de la burbuja actual."""
    st.html(_MARCA_USUARIO)


# Errores técnicos traducidos a algo accionable.  El detalle crudo no se
# esconde: se muestra debajo, para que un fallo siga siendo diagnosticable.
def _mensaje_de_error(exc: Exception) -> str:
    tipo = type(exc).__name__
    crudo = str(exc)
    bajo = crudo.lower()

    if isinstance(exc, ModuleNotFoundError):
        falta = getattr(exc, "name", None) or "una dependencia"
        causa = (
            f"Al entorno de Python le falta el paquete `{falta}`. "
            "Instale el proyecto con sus dependencias declaradas "
            "(`uv pip install -e .`) y vuelva a levantar la aplicación."
        )
    elif "api key" in bajo or "unauthorized" in bajo or "401" in bajo:
        causa = (
            "El servicio rechazó la clave de API. Revísela en el panel "
            "lateral, en **Claves de API**."
        )
    elif "429" in bajo or "rate limit" in bajo or "quota" in bajo:
        causa = (
            "El proveedor del modelo está limitando las peticiones. Espere un momento y reintente."
        )
    elif "timeout" in bajo or "timed out" in bajo:
        causa = (
            "El servicio no respondió a tiempo. Puede ser una escena grande "
            "o una conexión lenta; reintente la misma solicitud."
        )
    elif "connection" in bajo or "network" in bajo or "dns" in bajo:
        causa = (
            "No se pudo contactar el servicio remoto. Revise la conexión a "
            "internet y que los catálogos de USGS y Copernicus respondan."
        )
    else:
        causa = "La solicitud no se pudo completar."

    return "\n\n".join(
        [
            "**No se pudo completar la solicitud.**",
            causa,
            f"Detalle técnico: `{tipo}: {crudo}`",
        ]
    )


def render_chat() -> None:
    """Render just the message history area (inside a column).

    The chat input is rendered separately via ``render_chat_input()``
    at page level so Streamlit pins it to the viewport bottom.
    """

    messages = st.session_state.get("messages", [])

    # ── Scrollable message / welcome area ────────────────────────────────
    # Mismo alto que el panel del mapa, para que las dos columnas
    # terminen a la misma altura.
    chat_container = st.container(height=560, border=False)

    with chat_container:
        if not messages:
            st.markdown(WELCOME_HTML, unsafe_allow_html=True)

        # Los artefactos guardados en la sesion son los de la ultima corrida,
        # asi que solo el ultimo mensaje del agente puede recurrir a ellos.
        # Sin esta restriccion, una respuesta que solo pide un dato ("¿de que
        # fecha?") aparecia con el mapa de la corrida anterior debajo.
        ultimo_agente = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"),
            default=-1,
        )

        for i, msg in enumerate(messages):
            role = msg["role"]
            avatar = "🧑‍💻" if role == "user" else "🛰️"
            with st.chat_message(role, avatar=avatar):
                if role == "user":
                    _marcar_usuario()
                st.markdown(msg["content"])
                if role == "assistant":
                    _render_inline_artefacts(msg["content"], usar_respaldo=(i == ultimo_agente))

        if messages:
            _bajar_al_ultimo_mensaje()

    # Store a reference so render_chat_input can write to it
    st.session_state["_chat_container"] = chat_container


def render_chat_input() -> None:
    """Render the chat input at **page level** (outside columns).

    Streamlit natively pins ``st.chat_input`` to the viewport bottom
    when it is not nested inside a column or container.  Call this
    after the column block in ``main.py``.
    """
    messages = st.session_state.get("messages", [])
    is_busy = st.session_state.get("is_processing", False)
    chat_container = st.session_state.get("_chat_container")

    user_input = st.chat_input(
        "Describa el análisis que necesita…"
        if not messages
        else "Haga otra pregunta o una nueva solicitud…",
        disabled=is_busy,
    )

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        if chat_container:
            with chat_container, st.chat_message("user", avatar="🧑‍💻"):
                _marcar_usuario()
                st.markdown(user_input)

        # Validate LLM key
        provider = st.session_state.get("llm_provider", "openai")
        key_field = f"{provider}_api_key"
        if not st.session_state.get(key_field):
            warning = (
                f"Ingrese la clave de API de **{provider.title()}** "
                "en el panel lateral antes de enviar un mensaje."
            )
            st.session_state["messages"].append({"role": "assistant", "content": warning})
            if chat_container:
                with chat_container, st.chat_message("assistant", avatar="🛰️"):
                    st.warning(warning, icon="⚠️")
            return

        # ── Invoke agent ─────────────────────────────────────────────────
        st.session_state["is_processing"] = True
        if chat_container:
            with chat_container, st.chat_message("assistant", avatar="🛰️"):
                typing_placeholder = st.empty()
                typing_placeholder.markdown(TYPING_INDICATOR_HTML, unsafe_allow_html=True)

                try:
                    reply = _invoke_agent(user_input)
                except Exception as exc:
                    logger.exception("Agent error")
                    reply = _mensaje_de_error(exc)
                    _log_error_turn(user_input, exc)

                typing_placeholder.markdown(reply)
                _render_inline_artefacts(reply)
        else:
            try:
                reply = _invoke_agent(user_input)
            except Exception as exc:
                logger.exception("Agent error")
                reply = _mensaje_de_error(exc)
                _log_error_turn(user_input, exc)

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.session_state["is_processing"] = False
        _extract_map_artefacts(reply)

        # Trigger a full rerun so the Map Output panel (rendered before
        # the chat column) picks up the newly discovered artefacts.
        st.rerun()
