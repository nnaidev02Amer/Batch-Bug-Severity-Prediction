"""Visual layer: lightweight CSS + small HTML components (Streamlit-native otherwise).

Colours use translucent tints so the same CSS reads well in Streamlit's light and
dark themes. Only trusted, app-generated strings are ever placed in raw HTML -
user-entered text is always rendered through normal Streamlit widgets.
"""
from __future__ import annotations

import html

import streamlit as st

COLORS = {"Critical": "#dc2626", "Normal": "#d97706", "Non-Critical": "#16a34a"}
ICONS = {"Critical": "🔴", "Normal": "🟠", "Non-Critical": "🟢"}
BADGE = {"Critical": "red", "Normal": "orange", "Non-Critical": "green"}
PRIORITY_ICONS = {"High Priority": "⬆️", "Medium Priority": "➡️", "Low Priority": "⬇️"}
ACCENT = "#4f46e5"

_CSS = """
<style>
:root { --bsp-accent: #4f46e5; --bsp-border: rgba(128,128,128,.22); --bsp-soft: rgba(128,128,128,.07); }
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button, input, textarea {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
h1 { font-weight: 750; } h2, h3 { font-weight: 680; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--bsp-border); }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .55rem .75rem; border-radius: 10px; width: 100%; transition: background .15s; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--bsp-soft); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(79,70,229,.14); font-weight: 600; }
.bsp-brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem 0; }
.bsp-brand .logo { font-size:1.6rem; width:2.6rem; height:2.6rem; display:flex; align-items:center;
  justify-content:center; border-radius:12px; background: linear-gradient(135deg,#4f46e5,#7c3aed); }
.bsp-brand b { font-size:1.02rem; display:block; line-height:1.15; }
.bsp-brand span { font-size:.78rem; opacity:.7; }

/* Hero */
.bsp-hero { border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem; color: #fff;
  background: linear-gradient(120deg, #4338ca 0%, #6d28d9 55%, #9333ea 100%); }
.bsp-hero h1 { color:#fff; margin:0 0 .35rem 0; font-size: 1.9rem; padding:0; }
.bsp-hero p { margin:0; opacity:.92; font-size:1.02rem; max-width: 60rem; }
.bsp-hero .tag { display:inline-block; font-size:.75rem; font-weight:600; letter-spacing:.04em;
  text-transform:uppercase; background: rgba(255,255,255,.18); padding:.2rem .6rem; border-radius:999px;
  margin-bottom:.6rem; }

/* Cards */
.bsp-card { border:1px solid var(--bsp-border); background: var(--bsp-soft); border-radius:14px;
  padding: 1rem 1.15rem; height: 100%; }
.bsp-card h4 { margin:.1rem 0 .35rem 0; font-size:1rem; }
.bsp-card p { margin:0; opacity:.85; font-size:.92rem; }
.bsp-kpi .label { font-size:.8rem; opacity:.75; text-transform:uppercase; letter-spacing:.04em; }
.bsp-kpi .value { font-size:1.9rem; font-weight:750; line-height:1.2; }
.bsp-kpi .sub { font-size:.8rem; opacity:.7; }

/* Section header */
.bsp-section { display:flex; align-items:center; gap:.65rem; margin: 1.1rem 0 .4rem 0; }
.bsp-section .num { min-width:1.9rem; height:1.9rem; border-radius:999px; display:flex; align-items:center;
  justify-content:center; font-weight:700; font-size:.9rem; color:#fff; background: var(--bsp-accent); }
.bsp-section h3 { margin:0; padding:0; font-size:1.12rem; }
.bsp-section span { font-size:.85rem; opacity:.7; }

/* Workflow steps */
.bsp-steps { display:flex; flex-wrap:wrap; gap:.6rem; margin:.4rem 0 1rem 0; }
.bsp-step { flex:1 1 9rem; border:1px solid var(--bsp-border); border-radius:12px; padding:.7rem .8rem;
  background: var(--bsp-soft); }
.bsp-step .n { font-size:.72rem; font-weight:700; color: var(--bsp-accent); letter-spacing:.05em; }
.bsp-step .t { font-weight:600; font-size:.93rem; }

/* Severity result */
.bsp-result { border-radius:16px; padding:1.2rem 1.4rem; border:1px solid var(--c);
  background: color-mix(in srgb, var(--c) 12%, transparent); }
.bsp-result .k { font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; opacity:.75; }
.bsp-result .v { font-size:1.8rem; font-weight:780; color: var(--c); line-height:1.25; }
.bsp-result .p { font-size:1.05rem; font-weight:650; margin-top:.15rem; }

/* Buttons & inputs */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 10px; font-weight: 600; padding: .5rem 1rem; }
.stTextInput input, .stTextArea textarea { border-radius: 10px !important; }
[data-testid="stFileUploaderDropzone"] { border-radius: 14px; border: 2px dashed rgba(79,70,229,.45); }
[data-testid="stMetric"] { border:1px solid var(--bsp-border); background: var(--bsp-soft);
  border-radius:14px; padding:.75rem 1rem; }
[data-testid="stExpander"] details { border-radius: 12px; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _e(s) -> str:
    return html.escape(str(s))


def hero(title: str, subtitle: str, tag: str | None = None) -> None:
    t = f'<div class="tag">{_e(tag)}</div>' if tag else ""
    st.markdown(f'<div class="bsp-hero">{t}<h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>',
                unsafe_allow_html=True)


def section(num: str, title: str, subtitle: str = "") -> None:
    sub = f"<span>{_e(subtitle)}</span>" if subtitle else ""
    st.markdown(f'<div class="bsp-section"><div class="num">{_e(num)}</div><h3>{_e(title)}</h3>{sub}</div>',
                unsafe_allow_html=True)


def card(title: str, body: str, icon: str = "") -> None:
    st.markdown(f'<div class="bsp-card"><h4>{_e(icon)} {_e(title)}</h4><p>{_e(body)}</p></div>',
                unsafe_allow_html=True)


def kpi(label: str, value, sub: str = "", color: str | None = None) -> None:
    style = f' style="color:{color}"' if color else ""
    st.markdown(f'<div class="bsp-card bsp-kpi"><div class="label">{_e(label)}</div>'
                f'<div class="value"{style}>{_e(value)}</div><div class="sub">{_e(sub)}</div></div>',
                unsafe_allow_html=True)


def steps(items: list[str]) -> None:
    cells = "".join(f'<div class="bsp-step"><div class="n">STEP {i}</div><div class="t">{_e(t)}</div></div>'
                    for i, t in enumerate(items, 1))
    st.markdown(f'<div class="bsp-steps">{cells}</div>', unsafe_allow_html=True)


def severity_result(severity: str, priority: str) -> None:
    """Large result card. ``severity`` must be one of the model's classes."""
    c = COLORS[severity]
    st.markdown(
        f'<div class="bsp-result" style="--c:{c}">'
        f'<div class="k">Predicted severity</div><div class="v">{ICONS[severity]} {_e(severity)}</div>'
        f'<div class="k" style="margin-top:.6rem">Recommended priority</div>'
        f'<div class="p">{PRIORITY_ICONS.get(priority, "")} {_e(priority)}</div></div>',
        unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown('<div class="bsp-brand"><div class="logo">🐞</div><div><b>Bug Severity</b>'
                        '<span>Prediction System</span></div></div>', unsafe_allow_html=True)


def style_fig(fig, height: int | None = None):
    """Consistent, quiet Plotly styling (transparent so it follows the app theme)."""
    fig.update_layout(font_family="Inter, Segoe UI, system-ui, sans-serif", title_font_size=15,
                      margin=dict(t=48, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    if height:
        fig.update_layout(height=height)
    return fig
