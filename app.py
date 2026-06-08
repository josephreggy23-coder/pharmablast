from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st
    import streamlit.components.v1 as components
except Exception:
    st = None
    components = None

from src.config import (
    DATA_DIR,
    EXPORTED_DATA_DIR,
    FIGURE_DIR,
    POLYMER_QUANTUM_YIELDS,
    VISUAL_3D_DIR,
    ensure_directories,
)
from src.insight_engine import Condition, compare_polymers_for_condition, diagnose_condition, suggest_operating_condition
from src.physics_model import model_factors, predict_fluorescence_power_w
from src.sensor_model import sensor_response
from src.utils import save_validation_data


# ----------------------------------------------------------------------------
# Navigation + sweep model
# ----------------------------------------------------------------------------
PAGES = [
    ("Simulator", "Interactive fluorescence model", ""),
    ("Exports", "Output center", "Download datasets, reports, 2D figures, and interactive 3D models."),
]
PAGE_META = {name: (title, sub) for name, title, sub in PAGES}

# Which parameter is swept across the x-axis of the live graph
SWEEP = {
    "Dye concentration": {"key": "dye", "lo": 0.5, "hi": 10.0, "unit": "µg/mL", "axis": "Nile Red concentration (µg/mL)"},
    "Particle size": {"key": "diameter", "lo": 20.0, "hi": 1000.0, "unit": "µm", "axis": "Particle diameter (µm)"},
    "Excitation wavelength": {"key": "wavelength", "lo": 430.0, "hi": 590.0, "unit": "nm", "axis": "Excitation wavelength (nm)"},
    "Pharmaceutical concentration": {"key": "pharma", "lo": 0.0, "hi": 10.0, "unit": "units", "axis": "Pharmaceutical concentration (model units)"},
}


def _css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    :root {
        --ink: #1a1d24;
        --ink-soft: #444b57;
        --muted: #717784;
        --faint: #a2a8b4;
        --hair: #ecedf1;
        --hair-2: #f4f5f8;
        --surface: #ffffff;
        --surface-2: #fafbfc;
        --surface-3: #f3f4f7;
        --violet: #6d28d9;
        --violet-deep: #4c1d95;
        --violet-soft: #f6f3fe;
        --violet-line: #e7e0fb;
        --good: #067a55;
        --warn: #b45309;
        --mono: "SFMono-Regular", "JetBrains Mono", "Consolas", ui-monospace, monospace;
        --shadow-sm: 0 1px 2px rgba(24,27,34,0.04), 0 1px 3px rgba(24,27,34,0.05);
        --shadow-md: 0 6px 20px rgba(24,27,34,0.06), 0 2px 6px rgba(24,27,34,0.04);
        --radius: 14px;
        --radius-lg: 18px;
    }

    html, body, .stApp,
    section[data-testid="stSidebar"],
    button, input, select, textarea,
    [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-baseweb="select"] div, [data-testid="stExpander"] summary,
    .stApp h1, .stApp h2, .stApp h3 {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
    }
    .stApp { background: var(--surface); color: var(--ink); -webkit-font-smoothing: antialiased; }
    html { scroll-behavior: smooth; }
    [data-testid="stIconMaterial"], .material-icons, span[class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    }
    .metric-value, .factor-value, .score-value, .coord { font-family: var(--mono) !important; }

    header[data-testid="stHeader"] { background: transparent; height: 0; }
    #MainMenu, footer { visibility: hidden; }

    section.main > div { max-width: 1320px; padding-top: 0.6rem; }
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; }

    h1, h2, h3 { color: var(--ink); letter-spacing: -0.015em; }

    /* Thin, quiet scrollbars — a small designed touch */
    *::-webkit-scrollbar { width: 11px; height: 11px; }
    *::-webkit-scrollbar-track { background: transparent; }
    *::-webkit-scrollbar-thumb { background: #e2e4ea; border-radius: 999px; border: 3px solid var(--surface); }
    *::-webkit-scrollbar-thumb:hover { background: #d0d4dc; }

    /* Focus rings */
    *:focus-visible { outline: 2px solid var(--violet); outline-offset: 2px; border-radius: 6px; }

    /* ------------------------------------------------------------------ */
    /* Sidebar                                                            */
    /* ------------------------------------------------------------------ */
    section[data-testid="stSidebar"] {
        background: var(--surface-2);
        border-right: 1px solid var(--hair);
        min-width: 344px; max-width: 344px;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

    .sb-brand { display: flex; align-items: center; gap: 0.65rem; padding: 0 0.25rem 0.1rem; }
    .sb-mark {
        width: 32px; height: 32px; border-radius: 9px; flex: none; position: relative;
        background: linear-gradient(150deg, var(--violet) 0%, var(--violet-deep) 100%);
        box-shadow: 0 6px 16px rgba(76,29,149,0.30);
    }
    .sb-mark::after { content: ""; position: absolute; inset: 10px; border-radius: 50%; background: #fff; opacity: 0.94; }
    .sb-name { font-size: 1.2rem; font-weight: 900; color: var(--ink); line-height: 1; letter-spacing: -0.02em; }
    .sb-tag { font-size: 0.6rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: var(--violet-deep); margin-top: 0.18rem; }
    .sb-divider { height: 1px; background: var(--hair); margin: 1.1rem 0 0.8rem; }
    .sb-label {
        color: var(--faint); font-size: 0.66rem; font-weight: 800; letter-spacing: 0.14em;
        text-transform: uppercase; margin: 0.1rem 0 0.5rem 0.3rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.2rem; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        width: 100%; border: 1px solid transparent; border-radius: 10px; padding: 0.55rem 0.75rem;
        color: var(--ink-soft); font-weight: 600;
        transition: background-color 160ms ease, color 160ms ease, box-shadow 160ms ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: var(--surface-3); color: var(--ink); }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: var(--surface); color: var(--violet-deep); font-weight: 750;
        box-shadow: var(--shadow-sm), inset 3px 0 0 var(--violet);
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 0.82rem; color: var(--ink-soft); font-weight: 600; }

    /* Sliders — smooth thumb */
    [data-testid="stSlider"] [role="slider"] { transition: box-shadow 160ms ease, transform 120ms ease; }
    [data-testid="stSlider"] [role="slider"]:hover { transform: scale(1.12); }

    /* Selectbox / inputs */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        border-radius: 10px !important; border-color: var(--hair) !important; transition: border-color 160ms ease, box-shadow 160ms ease;
    }
    [data-baseweb="select"] > div:hover { border-color: var(--violet-line) !important; }

    /* ------------------------------------------------------------------ */
    /* Top bar — minimal                                                  */
    /* ------------------------------------------------------------------ */
    .topbar { padding-bottom: 1.15rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--hair); }
    .topbar-eyebrow { color: var(--violet); font-size: 0.66rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; }
    .topbar-title { font-size: 1.55rem; font-weight: 800; color: var(--ink); line-height: 1.1; margin-top: 0.3rem; letter-spacing: -0.025em; }
    .topbar-sub { color: var(--muted); margin-top: 0.4rem; max-width: 660px; line-height: 1.5; font-size: 0.95rem; }

    /* ------------------------------------------------------------------ */
    /* Metric / readout cards                                             */
    /* ------------------------------------------------------------------ */
    .metric-card {
        background: var(--surface); border: 1px solid var(--hair); border-radius: var(--radius);
        padding: 1rem 1.15rem; min-height: 110px; display: flex; flex-direction: column;
        box-shadow: var(--shadow-sm); transition: box-shadow 200ms ease, transform 200ms ease, border-color 200ms ease;
    }
    .metric-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
    .metric-card.primary { border-color: var(--violet-line); }
    .metric-label { color: var(--muted); font-size: 0.68rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .metric-value { color: var(--ink); font-size: 1.55rem; line-height: 1.02; font-weight: 800; letter-spacing: -0.02em; white-space: nowrap; }
    .metric-note { color: var(--muted); font-size: 0.8rem; margin-top: auto; padding-top: 0.4rem; }
    .status-good { color: var(--good); }
    .status-warn { color: var(--warn); }

    /* Graph caption / coordinate readout */
    .graph-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin: 0.15rem 0 0.65rem; flex-wrap: wrap; }
    .graph-title { font-weight: 750; color: var(--ink); font-size: 1.02rem; letter-spacing: -0.01em; }
    .coord {
        font-family: var(--mono); font-size: 0.82rem; font-weight: 700; color: var(--violet-deep);
        background: var(--violet-soft); border: 1px solid var(--violet-line); border-radius: 999px; padding: 0.32rem 0.78rem;
    }

    /* Factor bars */
    .factor-row { display: grid; grid-template-columns: 168px 1fr 60px; gap: 0.85rem; align-items: center; margin: 0.6rem 0; }
    .factor-label { color: var(--muted); font-size: 0.9rem; }
    .bar-track { background: var(--surface-3); border-radius: 999px; height: 8px; overflow: hidden; }
    .bar-fill { background: linear-gradient(90deg, #8b5cf6 0%, var(--violet-deep) 100%); height: 100%; border-radius: 999px; transition: width 320ms cubic-bezier(0.22,1,0.36,1); }
    .factor-value { color: var(--ink); font-weight: 700; text-align: right; font-size: 0.88rem; }

    /* Diagnostics */
    .score-ring {
        width: 100%; border-radius: var(--radius); display: block; margin: 0.1rem 0 0.8rem;
        background: var(--surface); border: 1px solid var(--hair); box-shadow: var(--shadow-sm); padding: 1rem 1.05rem;
    }
    .score-value { font-size: 1.95rem; line-height: 1; font-weight: 900; color: var(--violet-deep); letter-spacing: -0.02em; }
    .score-label { color: var(--muted); font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.32rem; }
    .score-ring::after { content: ""; display: block; width: var(--score); height: 7px; margin-top: 0.8rem; background: linear-gradient(90deg, #8b5cf6, var(--violet-deep)); border-radius: 999px; transition: width 320ms cubic-bezier(0.22,1,0.36,1); }
    .chip-list { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.6rem 0 0.4rem; }
    .chip { border: 1px solid var(--hair); background: var(--surface-2); color: var(--ink-soft); border-radius: 999px; padding: 0.32rem 0.66rem; font-size: 0.78rem; font-weight: 650; }
    .recommendation { background: var(--surface-2); border: 1px solid var(--hair); border-left: 3px solid var(--violet); border-radius: 10px; padding: 0.65rem 0.8rem; color: var(--ink-soft); margin: 0.5rem 0; line-height: 1.45; font-size: 0.92rem; }
    .small-note { color: var(--muted); font-size: 0.88rem; line-height: 1.55; }

    .interp { display: flex; gap: 0.75rem; align-items: flex-start; background: var(--violet-soft); border-radius: var(--radius); padding: 0.85rem 1rem; margin-top: 1rem; color: var(--ink-soft); line-height: 1.55; font-size: 0.94rem; }
    .interp-k { flex: none; font-family: var(--mono); font-size: 0.62rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: var(--violet-deep); background: var(--surface); border: 1px solid var(--violet-line); border-radius: 999px; padding: 0.26rem 0.6rem; margin-top: 0.08rem; }

    .section-kicker { color: var(--faint); font-weight: 800; font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase; margin: 1.1rem 0 0.6rem; }

    /* ------------------------------------------------------------------ */
    /* Native widgets                                                     */
    /* ------------------------------------------------------------------ */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px; border: 1px solid var(--hair); background: var(--surface); color: var(--ink-soft);
        font-weight: 650; box-shadow: var(--shadow-sm);
        transition: border-color 160ms ease, background-color 160ms ease, transform 140ms ease, box-shadow 160ms ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--violet-line); color: var(--violet-deep); background: var(--surface);
        transform: translateY(-1px); box-shadow: var(--shadow-md);
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--hair); border-radius: var(--radius); overflow: hidden; }

    /* Bordered container (the graph canvas) + expanders — soft, designed */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--hair) !important; border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-md); background: var(--surface); padding: 0.4rem 0.3rem;
    }
    div[data-testid="stExpander"] { border: 1px solid var(--hair); border-radius: var(--radius); box-shadow: var(--shadow-sm); overflow: hidden; }
    div[data-testid="stExpander"] summary { font-weight: 700; color: var(--ink-soft); }
    div[data-testid="stExpander"] summary:hover { color: var(--violet-deep); }

    @media (max-width: 900px) {
        section[data-testid="stSidebar"] { min-width: 300px; max-width: 300px; }
    }

    /* ------------------------------------------------------------------ */
    /* Loading splash — dark, once per page load, then fades to the app   */
    /* ------------------------------------------------------------------ */
    #pb-splash {
        position: fixed; inset: 0; z-index: 999999; overflow: hidden;
        display: flex; align-items: center; justify-content: center;
        background:
            radial-gradient(circle at 50% 38%, rgba(124,58,237,0.22), transparent 52%),
            radial-gradient(circle at 50% 120%, rgba(76,29,149,0.30), transparent 60%),
            #0b0912;
        animation: pbSplashOut 2.1s ease forwards;
    }
    #pb-splash::before {
        content: ""; position: absolute; inset: 0;
        background-image: radial-gradient(rgba(139,92,246,0.13) 1px, transparent 1.4px);
        background-size: 26px 26px;
        -webkit-mask: radial-gradient(circle at 50% 44%, #000 0%, transparent 66%);
                mask: radial-gradient(circle at 50% 44%, #000 0%, transparent 66%);
    }
    @keyframes pbSplashOut { 0%, 62% { opacity: 1; visibility: visible; } 100% { opacity: 0; visibility: hidden; } }
    .pb-splash-inner { position: relative; display: flex; flex-direction: column; align-items: center; animation: pbRise 0.6s cubic-bezier(0.22,1,0.36,1) both; }
    @keyframes pbRise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

    .pb-orb { position: relative; width: 104px; height: 104px; margin-bottom: 1.7rem; }
    .pb-ring {
        position: absolute; inset: 0; border-radius: 50%;
        background: conic-gradient(from 0deg, transparent 0deg, rgba(139,92,246,0.05) 110deg, #8b5cf6 300deg, #ddd6fe 360deg);
        -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 calc(100% - 5px));
                mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 calc(100% - 5px));
        filter: drop-shadow(0 0 9px rgba(139,92,246,0.65));
        animation: pbSpin 0.85s linear infinite;
    }
    @keyframes pbSpin { to { transform: rotate(360deg); } }
    .pb-ring2 { position: absolute; inset: -16px; border-radius: 50%; border: 1px solid rgba(139,92,246,0.30); animation: pbHalo 1.9s ease-out infinite; }
    @keyframes pbHalo { 0% { transform: scale(0.82); opacity: 0.6; } 100% { transform: scale(1.35); opacity: 0; } }
    .pb-core {
        position: absolute; inset: 23px; border-radius: 19px;
        background: linear-gradient(150deg, #a78bfa, #6d28d9 55%, #4c1d95);
        box-shadow: 0 0 34px rgba(124,58,237,0.70), inset 0 0 16px rgba(255,255,255,0.22);
        animation: pbPulse 1.2s ease-in-out infinite;
    }
    .pb-core::after { content: ""; position: absolute; inset: 15px; border-radius: 50%; background: radial-gradient(circle at 38% 32%, #ffffff, #ede9fe); box-shadow: 0 0 14px rgba(255,255,255,0.55); }
    @keyframes pbPulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(0.85); } }

    .pb-word {
        font-family: 'Inter', sans-serif; font-weight: 900; font-size: 2.25rem; letter-spacing: -0.035em;
        background: linear-gradient(100deg, #ede9fe 0%, #ffffff 28%, #c4b5fd 50%, #ffffff 72%, #ede9fe 100%);
        background-size: 220% auto; -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent;
        animation: pbSheen 2.4s linear infinite;
    }
    @keyframes pbSheen { to { background-position: 220% center; } }
    .pb-sub { font-family: var(--mono); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.36em; text-transform: uppercase; color: rgba(196,181,253,0.82); margin-top: 0.7rem; padding-left: 0.36em; }
    .pb-bar { width: 244px; height: 3px; border-radius: 999px; background: rgba(255,255,255,0.09); overflow: hidden; margin-top: 1.9rem; }
    .pb-bar-fill { height: 100%; width: 0; border-radius: 999px; background: linear-gradient(90deg, #8b5cf6, #ddd6fe); box-shadow: 0 0 12px rgba(139,92,246,0.85); animation: pbFill 1.7s cubic-bezier(0.4,0,0.2,1) forwards; }
    @keyframes pbFill { 0% { width: 0; } 100% { width: 100%; } }
    </style>
    """


def _html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


def _metric(label: str, value: str, note: str = "", primary: bool = False, status: str | None = None) -> None:
    css = "metric-card primary" if primary else "metric-card"
    value_class = f"metric-value {status}" if status else "metric-value"
    _html(
        f"""
        <div class="{css}">
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """
    )


def _factor_bar(label: str, value: float, max_value: float = 1.0) -> None:
    pct = max(0.0, min(100.0, value / max_value * 100.0))
    _html(
        f"""
        <div class="factor-row">
            <div class="factor-label">{label}</div>
            <div class="bar-track"><div class="bar-fill" style="width: {pct:.1f}%"></div></div>
            <div class="factor-value">{value:.3f}</div>
        </div>
        """
    )


def _score_ring(score: float, label: str) -> None:
    pct = max(0.0, min(100.0, score))
    _html(
        f"""
        <div class="score-ring" style="--score: {pct:.1f}%">
            <div class="score-value">{pct:.0f}</div>
            <div class="score-label">{label}</div>
        </div>
        """
    )


def _chip_list(items: list[str]) -> None:
    chips = "".join(f'<span class="chip">{item}</span>' for item in items)
    _html(f'<div class="chip-list">{chips}</div>')


def _recommendations(items: list[str]) -> None:
    for item in items[:4]:
        _html(f'<div class="recommendation">{item}</div>')


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _download_file(path: Path, label: str, key_suffix: str) -> None:
    if path.exists():
        key_source = f"{key_suffix}|{path.resolve()}|{label}|{path.stat().st_size}"
        key = "download_" + hashlib.md5(key_source.encode("utf-8")).hexdigest()
        st.download_button(label, path.read_bytes(), file_name=path.name, width="stretch", key=key)
    else:
        st.caption(f"{path.name} has not been generated yet.")


# ----------------------------------------------------------------------------
# Physics helpers
# ----------------------------------------------------------------------------
def _signal_mV(p: dict, **override) -> float:
    args = {
        "polymer": p["polymer"], "count": p["count"], "diameter": p["diameter"], "dye": p["dye"],
        "wavelength": p["wavelength"], "excitation": p["excitation"], "pharma": p["pharma"], "alpha": p["alpha"],
    }
    args.update(override)
    power = predict_fluorescence_power_w(
        args["polymer"], args["count"], args["diameter"], args["dye"],
        args["wavelength"], args["excitation"], args["pharma"], args["alpha"],
    )
    return float(sensor_response(power, noise_level=p["noise_level"], include_random=False)["ideal_voltage_mV"])


def _live_response_figure(p: dict, sweep_label: str, threshold: float, curx: float, cury: float) -> go.Figure:
    cfg = SWEEP[sweep_label]
    key, lo, hi = cfg["key"], cfg["lo"], cfg["hi"]
    xs = np.linspace(lo, hi, 200)
    ys = [_signal_mV(p, **{key: float(x)}) for x in xs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines", name="Modeled signal",
            line={"color": "#6d28d9", "width": 3, "shape": "spline", "smoothing": 0.6},
            fill="tozeroy", fillcolor="rgba(109,40,217,0.06)",
            hovertemplate="%{x:.1f}  ·  %{y:.1f} mV<extra></extra>",
        )
    )
    # Detection threshold reference line
    fig.add_hline(
        y=threshold, line={"color": "#dc2626", "width": 1.4, "dash": "dash"},
        annotation_text="detection threshold", annotation_position="top left",
        annotation_font={"color": "#dc2626", "size": 11},
    )
    # Desmos-style guide lines down to the axes
    fig.add_trace(go.Scatter(x=[curx, curx], y=[0, cury], mode="lines", line={"color": "#c0c4cd", "width": 1, "dash": "dot"}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[lo, curx], y=[cury, cury], mode="lines", line={"color": "#c0c4cd", "width": 1, "dash": "dot"}, hoverinfo="skip", showlegend=False))
    # Soft halo + crisp operating point
    fig.add_trace(go.Scatter(x=[curx], y=[cury], mode="markers", marker={"size": 26, "color": "rgba(109,40,217,0.14)"}, hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=[curx], y=[cury], mode="markers+text",
            marker={"size": 13, "color": "#6d28d9", "line": {"color": "#ffffff", "width": 2.5}},
            text=[f"  {cury:.0f} mV"], textposition="middle right", textfont={"color": "#4c1d95", "size": 13, "family": "Inter, sans-serif"},
            hovertemplate=f"operating point · {cury:.1f} mV<extra></extra>", showlegend=False,
        )
    )
    axis_common = {"gridcolor": "#f1f2f6", "linecolor": "#e6e8ec", "showline": True, "zeroline": False, "ticks": "outside", "tickcolor": "#e6e8ec", "tickfont": {"size": 12, "color": "#717784"}, "title_font": {"size": 12.5, "color": "#717784"}, "automargin": True}
    fig.update_layout(
        template="plotly_white", height=540, margin={"l": 12, "r": 16, "t": 14, "b": 8},
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#1a1d24", "size": 13},
        showlegend=False, hovermode="closest", dragmode="pan",
        hoverlabel={"bgcolor": "#1a1d24", "font": {"color": "white", "family": "Inter, sans-serif", "size": 12}, "bordercolor": "#1a1d24"},
        xaxis={**axis_common, "title": cfg["axis"]},
        yaxis={**axis_common, "title": "Predicted fluorescence (mV)", "rangemode": "tozero"},
    )
    return fig


# ----------------------------------------------------------------------------
# Shell + controls
# ----------------------------------------------------------------------------
def _sidebar_shell() -> str:
    with st.sidebar:
        _html(
            """
            <div class="sb-brand">
                <div class="sb-mark"></div>
                <div>
                    <div class="sb-name">Pharmablast</div>
                    <div class="sb-tag">Modeling Platform</div>
                </div>
            </div>
            <div class="sb-divider"></div>
            <div class="sb-label">Workspace</div>
            """
        )
        selected = st.radio(
            "Workspace navigation",
            [name for name, _, _ in PAGES],
            label_visibility="collapsed",
            key="main_navigation",
        )
    return selected


def _simulator_controls() -> dict:
    with st.sidebar:
        _html('<div class="sb-divider"></div><div class="sb-label">Model parameters</div>')
        polymer = st.selectbox("Polymer type", list(POLYMER_QUANTUM_YIELDS), help="Selects the polymer quantum yield Qp.")
        count = st.slider("Particle count", 1, 200, 80, help="Number of particles contributing fluorescence.")
        diameter = st.slider("Mean particle size (µm)", 20, 1000, 500, help="Particle diameter used for surface-area scaling.")
        dye = st.slider("Dye concentration (µg/mL)", 0.5, 10.0, 4.0, 0.1, help="Nile Red concentration. Saturation appears above the useful range.")
        wavelength = st.select_slider("Excitation wavelength (nm)", options=[450, 488, 550], value=550, help="Wavelength used in the Gaussian absorption efficiency.")
        excitation = st.slider("Excitation intensity", 0.2, 2.0, 1.0, 0.05, help="Relative excitation light intensity E.")
        pharma = st.slider("Pharmaceutical concentration", 0.0, 10.0, 2.0, 0.1, help="Modeled pharmaceutical concentration Cdrug.")
        alpha = st.slider("Alpha modulation coefficient", -0.08, 0.08, 0.0, 0.005, help="Negative alpha quenches; positive alpha enhances.")
        noise_level = st.slider("Sensor noise level", 0.5, 2.0, 1.0, 0.05, help="Scales shot, Johnson, flicker, and background noise.")

        _html('<div class="sb-divider"></div><div class="sb-label">Graph</div>')
        sweep = st.selectbox("Plot variable (x-axis)", list(SWEEP), index=0, help="Choose which parameter the curve sweeps across.")

    return {
        "polymer": polymer, "count": int(count), "diameter": float(diameter), "dye": float(dye),
        "wavelength": int(wavelength), "excitation": float(excitation), "pharma": float(pharma),
        "alpha": float(alpha), "noise_level": float(noise_level), "sweep": sweep,
    }


def _splash() -> None:
    """Branded loading splash. Rendered once per page load; fades out via CSS."""
    _html(
        """
        <div id="pb-splash">
            <div class="pb-splash-inner">
                <div class="pb-orb">
                    <div class="pb-ring2"></div>
                    <div class="pb-ring"></div>
                    <div class="pb-core"></div>
                </div>
                <div class="pb-word">Pharmablast</div>
                <div class="pb-sub">Modeling Platform</div>
                <div class="pb-bar"><div class="pb-bar-fill"></div></div>
            </div>
        </div>
        """
    )


def _topbar(page: str) -> None:
    title, sub = PAGE_META.get(page, (page, ""))
    sub_html = f'<div class="topbar-sub">{sub}</div>' if sub else ""
    _html(
        f"""
        <div class="topbar">
            <div class="topbar-eyebrow">{page}</div>
            <div class="topbar-title">{title}</div>
            {sub_html}
        </div>
        """
    )


# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------
def page_simulator(p: dict) -> None:
    power = predict_fluorescence_power_w(p["polymer"], p["count"], p["diameter"], p["dye"], p["wavelength"], p["excitation"], p["pharma"], p["alpha"])
    sensor = sensor_response(power, noise_level=p["noise_level"], include_random=False)
    factors = model_factors(p["polymer"], p["count"], p["diameter"], p["dye"], p["wavelength"], p["excitation"], p["pharma"], p["alpha"])

    signal = float(sensor["ideal_voltage_mV"])
    snr = float(sensor["signal_to_noise_ratio"])
    threshold = float(sensor["detection_threshold_mV"])
    detected = bool(sensor["detected"])
    status = "Detected" if detected else "Below"

    cfg = SWEEP[p["sweep"]]
    curx = float(p[cfg["key"]])

    # Readout strip
    c1, c2, c3 = st.columns(3)
    with c1:
        _metric("Fluorescence", f"{signal:.1f} mV", "Predicted TIA output", primary=True)
    with c2:
        _metric("SNR", f"{snr:.1f}", "Signal-to-noise ratio", primary=True)
    with c3:
        _metric("Detection", status, f"{threshold:.2f} mV cutoff", primary=True, status="status-good" if detected else "status-warn")

    st.write("")
    with st.container(border=True):
        _html(
            f"""
            <div class="graph-head">
                <div class="graph-title">Response curve · fluorescence vs {p['sweep'].lower()}</div>
                <div class="coord">{cfg['axis'].split('(')[0].strip()} = {curx:g} {cfg['unit']}  →  {signal:.0f} mV</div>
            </div>
            """
        )
        fig = _live_response_figure(p, p["sweep"], threshold, curx, signal)
        st.plotly_chart(
            fig, width="stretch",
            config={"displayModeBar": False, "scrollZoom": True, "displaylogo": False},
        )

    # Live interpretation
    if p["dye"] > 5:
        interp = "Dye is in the saturation zone, so additional dye gives diminishing returns."
    elif 3 <= p["dye"] <= 5:
        interp = "Dye is in the useful operating range, balancing signal strength and saturation."
    else:
        interp = "Dye is below the useful operating range, so increasing concentration can improve signal."
    if p["alpha"] < 0:
        interp += " The selected pharmaceutical coefficient is quenching the modeled fluorescence."
    elif p["alpha"] > 0:
        interp += " The selected pharmaceutical coefficient is enhancing the modeled fluorescence."
    _html(f'<div class="interp"><span class="interp-k">Reading</span><span>{interp}</span></div>')

    with st.expander("Physical factor profile", expanded=False):
        _factor_bar("Quantum yield", float(factors["quantum_yield"]), 0.45)
        _factor_bar("Wavelength efficiency", float(factors["wavelength_efficiency"]), 1.0)
        _factor_bar("Dye saturation", float(factors["saturation_factor"]), 1.0)
        _factor_bar("Drug modulation", float(factors["drug_modulation_factor"]), 1.8)

    with st.expander("Model diagnostics & recommended adjustment", expanded=False):
        condition = Condition(
            polymer=p["polymer"], particle_count=p["count"], diameter_um=p["diameter"],
            dye_concentration_ug_ml=p["dye"], wavelength_nm=p["wavelength"], excitation_intensity=p["excitation"],
            pharmaceutical_concentration=p["pharma"], alpha=p["alpha"], noise_level=p["noise_level"],
        )
        diagnosis = diagnose_condition(condition)
        suggestion = suggest_operating_condition(condition)
        polymer_rows = compare_polymers_for_condition(condition)

        top_l, top_r = st.columns([0.36, 0.64], gap="large")
        with top_l:
            _score_ring(float(diagnosis["score"]), diagnosis["label"])
        with top_r:
            _html(f'<div class="small-note">{diagnosis["summary"]}</div>')
            _chip_list(diagnosis["limiting_factors"])

        rec = suggestion["condition"]
        _html('<div class="section-kicker">Suggested operating adjustment</div>')
        m1, m2, m3 = st.columns(3)
        with m1:
            _metric("Dye", f"{rec.dye_concentration_ug_ml:.2f}", "µg/mL")
        with m2:
            _metric("Wavelength", f"{rec.wavelength_nm}", "nm")
        with m3:
            _metric("SNR", f"{suggestion['snr']:.1f}", suggestion["label"])

        _html('<div class="section-kicker">Guidance</div>')
        _recommendations(diagnosis["recommendations"])

        _html('<div class="section-kicker">Polymer brightness under this condition</div>')
        polymer_df = pd.DataFrame(polymer_rows).rename(
            columns={"polymer": "Polymer", "quantum_yield": "Quantum yield", "signal_mV": "Signal (mV)", "snr": "SNR"}
        )
        st.dataframe(polymer_df, width="stretch", hide_index=True)
        _html('<div class="small-note">Deterministic model diagnostics — not claims of global optimization or field deployment performance.</div>')


def page_exports() -> None:
    groups = {
        "Data and reports": [
            DATA_DIR / "synthetic_modeling_data.csv",
            DATA_DIR / "validation_data.csv",
            EXPORTED_DATA_DIR / "synthetic_modeling_data.csv",
            Path("outputs/reports/software_summary.md"),
        ],
        "2D PNG figures": sorted(FIGURE_DIR.glob("*.png")),
        "3D HTML files": sorted(VISUAL_3D_DIR.glob("*.html")),
    }
    for title, files in groups.items():
        _html(f'<div class="section-kicker">{title}</div>')
        if not files:
            st.caption("Nothing generated yet. Run `python main.py` to build outputs.")
            continue
        cols = st.columns(3)
        for index, path in enumerate(files):
            with cols[index % 3]:
                safe_title = title.lower().replace(" ", "_")
                _download_file(path, f"{path.name}", f"{safe_title}_{index}")


def main() -> None:
    if st is None:
        print("Streamlit is not installed. Install requirements, then run: streamlit run app.py")
        return

    ensure_directories()
    save_validation_data()
    st.set_page_config(page_title="Pharmablast Modeling Platform", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
    st.markdown(_css(), unsafe_allow_html=True)

    # Loading splash — show once per page load, never on parameter reruns
    if not st.session_state.get("pb_splash_done"):
        st.session_state["pb_splash_done"] = True
        _splash()

    page = _sidebar_shell()
    _topbar(page)

    if page == "Simulator":
        params = _simulator_controls()
        page_simulator(params)
    elif page == "Exports":
        page_exports()


if __name__ == "__main__":
    main()
