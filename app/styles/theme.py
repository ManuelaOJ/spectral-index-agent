"""
CSS styles for the Spectral Agent Streamlit app.

Loaded via ``st.markdown(CUSTOM_CSS, unsafe_allow_html=True)`` in main.py.

Design System — "Terra"
========================
Built for geospatial / remote-sensing professionals.

Palette
-------
  Primary   : #4CAF50  (Earth Green)
  Secondary : #2196F3  (Sky Blue)
  Accent    : #26A69A  (Teal)
  Surface   : #111827  (Deep Navy)
  Base      : #0a0f1a  (Near-black)
  Card      : #1e2433  (Elevated surface)
  Border    : #2d3548  (Subtle divider)
  Text      : #E2E8F0  (Warm white)
  TextMuted : #94A3B8  (Slate)
  Success   : #66BB6A
  Error     : #EF5350
  Warning   : #FFA726

Typography
----------
  Font stack : "Inter", system-ui, sans-serif
  Headings   : 600–700 weight, tight letter-spacing
  Body       : 400 weight, generous line-height (1.6)
"""

CUSTOM_CSS = """\
<style>
/* ─── Google Font ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ─── Design Tokens ──────────────────────────────────────────────── */
:root {
    /* Palette */
    --primary:    #4CAF50;
    --primary-hover: #66BB6A;
    --secondary:  #2196F3;
    --accent:     #26A69A;
    --bg-base:    #0a0f1a;
    --bg-surface: #111827;
    --bg-card:    #1e2433;
    --bg-elevated:#263046;
    --border:     #2d3548;
    --border-hover: #4a5568;
    --text:       #E2E8F0;
    --text-muted: #94A3B8;
    --text-dim:   #64748B;
    --success:    #66BB6A;
    --error:      #EF5350;
    --warning:    #FFA726;

    /* Typography */
    --font-family: 'Inter', system-ui, -apple-system, sans-serif;
    --font-xs:   0.7rem;
    --font-sm:   0.8rem;
    --font-base: 0.9rem;
    --font-lg:   1.05rem;
    --font-xl:   1.3rem;
    --font-2xl:  1.75rem;
    --lh-tight:  1.3;
    --lh-normal: 1.6;

    /* Spacing */
    --space-xs:  0.25rem;
    --space-sm:  0.5rem;
    --space-md:  0.75rem;
    --space-lg:  1rem;
    --space-xl:  1.5rem;

    /* Radius */
    --radius-sm: 0.375rem;
    --radius-md: 0.625rem;
    --radius-lg: 0.875rem;
    --radius-xl: 1rem;
    --radius-pill: 50rem;
}

/* ─── Global Resets ──────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--font-family) !important;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-family) !important;
    letter-spacing: -0.015em;
}

/* Remove default top padding and reduce bottom so chat input is near viewport edge */
.stMainBlockContainer {
    padding-top: 1.5rem !important;
    padding-bottom: 0.5rem !important;
}

/* Reduce the gap Streamlit adds below columns */
[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

/* ─── Header Band ────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1a3a2a 0%, #0d2137 50%, #14243d 100%);
    border: 1px solid var(--border);
    padding: 1rem 1.5rem;
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-lg);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg,
        rgba(76, 175, 80, 0.12) 0%,
        rgba(33, 150, 243, 0.08) 100%);
    pointer-events: none;
}
.app-header h1 {
    margin: 0;
    font-size: var(--font-xl);
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
    position: relative;
}
.app-header p {
    margin: 0.25rem 0 0 0;
    font-size: var(--font-sm);
    color: var(--text-muted);
    position: relative;
}

/* ─── Status Bar ─────────────────────────────────────────────────── */
.status-bar {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    flex-wrap: wrap;
    padding: var(--space-sm) var(--space-lg);
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-lg);
    font-size: var(--font-xs);
    color: var(--text-muted);
}
.status-bar .status-item {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}
.status-bar .status-item .label { color: var(--text-dim); }
.status-bar .status-item .value {
    color: var(--text);
    font-weight: 600;
}
.status-bar .divider {
    width: 1px;
    height: 14px;
    background: var(--border);
}

/* ─── Section Titles ─────────────────────────────────────────────── */
.section-title {
    font-size: var(--font-sm);
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 var(--space-sm) 0;
}

/* ─── Chat Column Flex Layout ────────────────────────────────────── */
/* The chat column becomes a flex column so the input sticks to the
   bottom while the message area fills the remaining vertical space. */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 220px);   /* viewport minus header+status */
    min-height: 400px;
}
.chat-wrapper .chat-messages {
    flex: 1 1 auto;
    overflow-y: auto;
    scroll-behavior: smooth;
}
.chat-wrapper .chat-input-anchor {
    flex: 0 0 auto;
    padding-top: var(--space-sm);
    background: var(--bg-base);
}

/* ─── Chat Input ─────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0;
    z-index: 100;
    background: var(--bg-base);
    padding-top: var(--space-sm);
}

/* Chat input field styling */
[data-testid="stChatInput"] textarea {
    font-family: var(--font-family) !important;
    font-size: var(--font-base) !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.15) !important;
}

/* ─── Chat Bubbles ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-sm);
    padding: var(--space-md) var(--space-lg);
    border: 1px solid var(--border);
    background: var(--bg-card);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    font-size: var(--font-base);
    line-height: var(--lh-normal);
}
[data-testid="stChatMessage"]:hover {
    border-color: var(--border-hover);
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
}

/* Distincion usuario / agente.
   Streamlit no marca el rol en el DOM, asi que chat.py inyecta una
   marca invisible en la burbuja del usuario y aqui se selecciona con
   :has().  El contenedor de la marca se oculta; :has() sigue
   funcionando porque es estructural, no depende del layout. */
[data-testid="stChatMessage"] [data-testid="stElementContainer"]:has(.marca-usuario) {
    display: none !important;
}
[data-testid="stChatMessage"]:has(.marca-usuario) {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.10) 0%,
                                        var(--bg-card) 60%);
    border-color: rgba(76, 175, 80, 0.28);
}
/* La del agente lleva un filo de color a la izquierda: se distingue de
   un vistazo cual mensaje es la respuesta, incluso sin mirar el icono. */
[data-testid="stChatMessage"]:not(:has(.marca-usuario)) {
    border-left: 3px solid rgba(76, 175, 80, 0.45);
}

/* ─── Typing Indicator ───────────────────────────────────────────── */
@keyframes pulse-dot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.3; }
    40% { transform: scale(1); opacity: 1; }
}
.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: var(--space-sm) var(--space-md);
}
.typing-indicator span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--primary);
    animation: pulse-dot 1.4s infinite both;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

/* ─── Processing Banner ──────────────────────────────────────────── */
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.processing-banner {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-lg);
    background: linear-gradient(90deg,
        var(--bg-card) 25%,
        var(--bg-elevated) 50%,
        var(--bg-card) 75%);
    background-size: 200% 100%;
    animation: shimmer 2s infinite linear;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    font-size: var(--font-sm);
    color: var(--text-muted);
    margin-bottom: var(--space-sm);
}
.processing-banner .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--primary);
    animation: pulse-dot 1.4s infinite;
}

/* ─── Inline Map Preview ─────────────────────────────────────────── */
.chat-map-preview {
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    margin: var(--space-md) 0;
    overflow: hidden;
    background: var(--bg-surface);
}
.chat-map-preview img {
    width: 100%;
    max-height: 340px;
    object-fit: contain;
    display: block;
}

/* ─── Map Placeholder ────────────────────────────────────────────── */
.map-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 560px;
    background: var(--bg-card);
    border: 1px dashed var(--border);
    border-radius: var(--radius-lg);
    color: var(--text-dim);
    text-align: center;
    gap: var(--space-md);
}
.map-placeholder .icon {
    font-size: 2.5rem;
    opacity: 0.4;
}
.map-placeholder .title {
    font-size: var(--font-base);
    font-weight: 600;
    color: var(--text-muted);
}
.map-placeholder .subtitle {
    font-size: var(--font-sm);
    max-width: 280px;
}

/* ─── Map Container ──────────────────────────────────────────────── */
.map-frame {
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    overflow: hidden;
    background: var(--bg-card);
}
.map-frame iframe {
    border: none !important;
}

/* ─── Metrics / KPI Cards ────────────────────────────────────────── */
.kpi-row {
    display: flex;
    gap: var(--space-sm);
    margin-bottom: var(--space-md);
}
.kpi-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: var(--space-md);
    text-align: center;
}
.kpi-card .kpi-value {
    font-size: var(--font-lg);
    font-weight: 700;
    color: var(--text);
}
.kpi-card .kpi-label {
    font-size: var(--font-xs);
    color: var(--text-muted);
    margin-top: 2px;
}
.kpi-card.success .kpi-value { color: var(--success); }
.kpi-card.error   .kpi-value { color: var(--error); }

/* ─── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    min-width: 300px;
    max-width: 340px;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: var(--space-lg);
}

/* Sidebar brand / logo area */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-md) 0;
    margin-bottom: var(--space-md);
    border-bottom: 1px solid var(--border);
}
.sidebar-brand .brand-icon {
    font-size: 1.5rem;
}
.sidebar-brand .brand-text {
    font-size: var(--font-base);
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
}
.sidebar-brand .brand-version {
    font-size: var(--font-xs);
    color: var(--text-dim);
    font-weight: 400;
}

/* Sidebar section headers */
.sidebar-section {
    font-size: var(--font-xs);
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: var(--space-lg) 0 var(--space-sm) 0;
}

/* Sidebar metrics row */
.sidebar-metrics {
    display: flex;
    gap: var(--space-xs);
    flex-wrap: wrap;
}
.sidebar-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: var(--radius-pill);
    font-size: var(--font-xs);
    font-weight: 500;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    color: var(--text-muted);
}
.sidebar-chip.green  { border-color: rgba(76,175,80,0.3);  color: var(--success); }
.sidebar-chip.red    { border-color: rgba(239,83,80,0.3);  color: var(--error); }
.sidebar-chip.blue   { border-color: rgba(33,150,243,0.3); color: var(--secondary); }

/* ─── Buttons ────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--font-family) !important;
    font-weight: 600;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease;
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button:not([kind]) {
    border: 1px solid var(--border) !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Download buttons */
.stDownloadButton > button {
    font-family: var(--font-family) !important;
    font-weight: 600;
    width: 100%;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text) !important;
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--primary) !important;
}

/* Satellite quick-pick buttons */
.sat-button-row {
    display: flex;
    gap: var(--space-sm);
    margin-top: var(--space-sm);
}

/* ─── Expanders ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-card);
    margin-bottom: var(--space-sm);
}

/* ─── Dividers ───────────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}

/* ─── Status Badge ───────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 10px;
    border-radius: var(--radius-pill);
    font-size: var(--font-xs);
    font-weight: 600;
}
.status-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.status-badge.connected {
    background: rgba(76, 175, 80, 0.12);
    color: var(--success);
}
.status-badge.connected::before { background: var(--success); }
.status-badge.disconnected {
    background: rgba(239, 83, 80, 0.12);
    color: var(--error);
}
.status-badge.disconnected::before { background: var(--error); }

/* ─── Welcome Card ───────────────────────────────────────────────── */
.welcome-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--space-xl);
    text-align: center;
    margin: 0 auto;
    max-width: 560px;
}
.welcome-card h3 {
    margin: 0 0 var(--space-sm) 0;
    font-size: var(--font-lg);
    font-weight: 700;
    color: var(--text);
}
.welcome-card p {
    color: var(--text-muted);
    font-size: var(--font-sm);
    margin-bottom: var(--space-lg);
    line-height: var(--lh-normal);
}
.welcome-card .suggestions {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
}
.welcome-card .suggestion {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: var(--space-sm) var(--space-md);
    font-size: var(--font-sm);
    color: var(--text-muted);
    text-align: left;
    cursor: default;
    transition: border-color 0.2s ease;
}
.welcome-card .suggestion:hover {
    border-color: var(--primary);
    color: var(--text);
}
.welcome-card .suggestion .prompt-icon {
    margin-right: 6px;
    opacity: 0.6;
}

/* ─── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--border-hover);
}

/* ─── Toast / Notification ───────────────────────────────────────── */
.toast {
    position: fixed;
    top: 1rem;
    right: 1rem;
    padding: var(--space-sm) var(--space-lg);
    border-radius: var(--radius-md);
    font-size: var(--font-sm);
    font-weight: 500;
    z-index: 1000;
    animation: slide-in 0.3s ease;
}
@keyframes slide-in {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
.toast.success { background: rgba(76,175,80,0.15); border: 1px solid var(--success); color: var(--success); }
.toast.error   { background: rgba(239,83,80,0.15); border: 1px solid var(--error);   color: var(--error); }

/* --- Chrome de Streamlit oculto (modo presentacion) -------------- */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stAppDeployButton,
#MainMenu,
footer {
    display: none !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* --- Columnas a la misma altura, sin franja muerta abajo --------- */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex;
    flex-direction: column;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}
.welcome-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 520px;
}

/* --- Cargador de archivos en espanol -------------------------- */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: block !important;
    width: 100%;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] > div > div {
    display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "Arrastre aquí el archivo del área";
    display: block;
    font-size: var(--font-sm);
    font-weight: 600;
    line-height: var(--lh-tight);
    color: var(--text-muted);
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "KML, GeoJSON o SHP completo  ·  máximo 200 MB";
    display: block;
    margin-top: 3px;
    font-size: var(--font-xs);
    line-height: var(--lh-tight);
    color: var(--text-dim);
}
/* Streamlit 1.62 pone el boton antes de las instrucciones; con ``order`` se
   deja el texto arriba y el boton abajo, como en el resto del panel. */
[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: var(--space-sm);
}
[data-testid="stFileUploaderDropzoneInstructions"] { order: 1; }
[data-testid="stFileUploaderDropzone"] > span { order: 2; }

/* El rotulo del boton llega como icono ("upload") mas texto ("Upload") en un
   contenedor propio, asi que ``font-size: 0`` en el boton ya no basta: hay
   que ocultar los dos hijos y poner el rotulo con ::after.

   El selector va acotado a ``> span > button``, el boton de examinar del
   estado vacio.  Sin esa restriccion el rotulo se colaba tambien en los dos
   botones que aparecen con un archivo ya cargado, el de borrarlo y el de
   agregar otro, y los tres se solapaban dentro de la tarjeta. */
[data-testid="stFileUploaderDropzone"] > span > button [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] > span > button [data-testid="stMarkdownContainer"] {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] > span > button::after {
    content: "Buscar archivos";
    font-size: var(--font-sm);
}
</style>
"""

# ─── HTML fragments for chat UI ──────────────────────────────────────────

TYPING_INDICATOR_HTML = """\
<div class="typing-indicator">
    <span></span><span></span><span></span>
</div>
"""

MAP_PREVIEW_TEMPLATE = """\
<div class="chat-map-preview">
    <img src="data:image/png;base64,{b64}" alt="Vista previa del mapa temático" />
</div>
"""

WELCOME_HTML = """\
<div class="welcome-wrap">
<div class="welcome-card">
    <h3>Agente de índices espectrales</h3>
    <p>Describa lo que necesita en lenguaje natural. El agente busca la
    escena satelital, la descarga, calcula el índice y genera el mapa
    temático.</p>
    <div class="suggestions">
        <div class="suggestion">
            <span class="prompt-icon">&#x1F33F;</span>
            Calcula el NDVI para esta área en enero de 2024 usando Sentinel-2
        </div>
        <div class="suggestion">
            <span class="prompt-icon">&#x1F30A;</span>
            Estima el NDWI para el área cargada en marzo de 2024 con Landsat
        </div>
        <div class="suggestion">
            <span class="prompt-icon">&#x1F525;</span>
            Genera el NBR para esta zona en julio de 2020 usando Landsat
        </div>
    </div>
</div>
</div>
"""

MAP_PLACEHOLDER_HTML = """\
<div class="map-placeholder">
    <div class="icon">&#x1F5FA;</div>
    <div class="title">Mapa generado</div>
    <div class="subtitle">Aquí aparecerán los mapas cuando el agente
    termine el análisis.</div>
</div>
"""

PROCESSING_BANNER_HTML = """\
<div class="processing-banner">
    <div class="dot"></div>
    El agente está trabajando&hellip;
</div>
"""
