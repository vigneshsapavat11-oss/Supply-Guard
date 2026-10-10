"""Shared presentation layer for the SupplyGuard Streamlit application."""

import streamlit as st


def apply_enterprise_theme() -> None:
    """Apply the shared SupplyGuard theme and appearance control."""
    st.sidebar.markdown(
        """
        <div class="sg-brand">
            <div class="sg-brand-mark">SG</div>
            <div class="sg-brand-copy">
                <strong>SUPPLYGUARD</strong>
                <span>SUPPLY CHAIN INTELLIGENCE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    dark_mode = st.sidebar.toggle(
        "Dark appearance",
        key="supplyguard_dark_mode",
        help="Switch between light and dark dashboard themes.",
    )

    if dark_mode:
        colors = {
            "background": "#101b21",
            "surface": "#17262e",
            "surface_alt": "#1d3039",
            "sidebar": "#122129",
            "text": "#e8f0f2",
            "muted": "#9dafb6",
            "border": "rgba(184, 210, 218, 0.14)",
            "primary": "#62c9bb",
            "primary_hover": "#7bd8cb",
            "accent": "#f2a56f",
            "shadow": "0 12px 32px rgba(0, 0, 0, 0.20)",
            "grid": "rgba(210, 230, 235, 0.12)",
        }
    else:
        colors = {
            "background": "#f2f6f6",
            "surface": "#ffffff",
            "surface_alt": "#f7faf9",
            "sidebar": "#fbfdfc",
            "text": "#19343b",
            "muted": "#63777d",
            "border": "rgba(35, 74, 81, 0.12)",
            "primary": "#087f78",
            "primary_hover": "#066b66",
            "accent": "#c66b3d",
            "shadow": "0 12px 32px rgba(26, 63, 68, 0.07)",
            "grid": "rgba(35, 74, 81, 0.12)",
        }

    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --sg-bg: %(background)s;
        --sg-surface: %(surface)s;
        --sg-surface-alt: %(surface_alt)s;
        --sg-sidebar: %(sidebar)s;
        --sg-text: %(text)s;
        --sg-muted: %(muted)s;
        --sg-border: %(border)s;
        --sg-primary: %(primary)s;
        --sg-primary-hover: %(primary_hover)s;
        --sg-accent: %(accent)s;
        --sg-shadow: %(shadow)s;
        --sg-grid: %(grid)s;
        color-scheme: %(scheme)s;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
        letter-spacing: 0;
    }

    [data-testid="stAppViewContainer"], .stApp {
        color: var(--sg-text);
        background: var(--sg-bg);
    }

    [data-testid="stAppViewContainer"] > .main {
        background:
            radial-gradient(ellipse at 6%% 0%%, rgba(18, 145, 132, 0.055), transparent 36rem),
            var(--sg-bg);
    }

    .main .block-container {
        max-width: 1560px;
        padding: 2rem clamp(1rem, 3.1vw, 3.25rem) 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--sg-text);
        font-family: 'Manrope', 'DM Sans', sans-serif;
        letter-spacing: 0;
    }

    .stMarkdown h1 {
        margin: 0.35rem 0 0.45rem;
        font-size: clamp(1.8rem, 2.6vw, 2.35rem);
        font-weight: 800;
        line-height: 1.18;
    }

    .stMarkdown h2 { font-size: 1.55rem; font-weight: 750; }
    .stMarkdown h3 { font-size: 1.15rem; font-weight: 700; margin-top: 1.65rem; }
    .stMarkdown h4 { font-size: 1rem; font-weight: 700; }
    .stMarkdown p, .stMarkdown li { color: var(--sg-text); line-height: 1.65; }
    .stCaption, [data-testid="stCaption"] { color: var(--sg-muted); }

    .main-header {
        color: var(--sg-text);
        font-family: 'Manrope', 'DM Sans', sans-serif;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0.2rem 0 0.4rem;
    }

    .sub-header {
        color: var(--sg-muted);
        font-size: 1.02rem;
        margin-bottom: 1.7rem;
    }

    [data-testid="stSidebar"] {
        background: var(--sg-sidebar);
        border-right: 1px solid var(--sg-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: var(--sg-text) !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaption"] {
        color: var(--sg-muted) !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }

    .sg-brand {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        padding: 0.6rem 0.15rem 0.35rem;
        color: var(--sg-text);
    }

    .sg-brand-mark {
        display: grid;
        width: 2.55rem;
        height: 2.55rem;
        flex: 0 0 2.55rem;
        place-items: center;
        border-radius: 11px;
        background: var(--sg-primary);
        color: #ffffff;
        font-family: 'Manrope', sans-serif;
        font-size: 0.96rem;
        font-weight: 800;
    }

    .sg-brand-copy { display: grid; gap: 0.15rem; }
    .sg-brand-copy strong { font: 800 0.86rem 'Manrope', sans-serif; letter-spacing: 0.04em; }
    .sg-brand-copy span { color: var(--sg-muted); font-size: 0.58rem; font-weight: 700; }

    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebarNav"] a span,
    [data-testid="stSidebarNav"] a p {
        border-radius: 9px;
        color: var(--sg-muted) !important;
        transition: background 160ms ease, color 160ms ease, transform 160ms ease;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: var(--sg-surface-alt);
        color: var(--sg-text);
        transform: translateX(2px);
    }

    [data-testid="stMetric"] {
        min-height: 104px;
        padding: 1rem 1.15rem;
        border: 1px solid var(--sg-border);
        border-radius: 13px;
        background: var(--sg-surface);
        box-shadow: var(--sg-shadow);
        transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: var(--sg-primary);
        box-shadow: 0 15px 34px rgba(26, 63, 68, 0.12);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--sg-muted);
        font-size: 0.78rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--sg-text);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 750;
    }

    [data-testid="stPlotlyChart"] {
        overflow: hidden;
        border: 1px solid var(--sg-border);
        border-radius: 13px;
        background: var(--sg-surface);
        box-shadow: var(--sg-shadow);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        overflow: hidden;
        border: 1px solid var(--sg-border);
        border-radius: 11px;
        background: var(--sg-surface);
        box-shadow: 0 5px 18px rgba(26, 63, 68, 0.045);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--sg-border);
        border-radius: 11px;
        background: var(--sg-surface);
    }

    [data-testid="stAlert"] {
        border: 1px solid var(--sg-border);
        border-radius: 11px;
        box-shadow: 0 5px 18px rgba(26, 63, 68, 0.04);
    }

    .stButton > button, [data-testid="stDownloadButton"] > button {
        min-height: 2.65rem;
        border: 1px solid var(--sg-border);
        border-radius: 9px;
        color: var(--sg-text);
        background: var(--sg-surface);
        font-weight: 700;
        transition: transform 150ms ease, background 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
    }

    .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px);
        border-color: var(--sg-primary);
        color: var(--sg-primary);
        box-shadow: 0 7px 18px rgba(26, 63, 68, 0.09);
    }

    .stButton > button[kind="primary"] {
        border-color: var(--sg-primary);
        color: #ffffff;
        background: var(--sg-primary);
    }

    .stButton > button[kind="primary"]:hover {
        border-color: var(--sg-primary-hover);
        color: #ffffff;
        background: var(--sg-primary-hover);
    }

    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stMultiSelect"] [data-baseweb="select"],
    [data-testid="stDateInput"] input {
        border-color: var(--sg-border);
        border-radius: 8px;
        background: var(--sg-surface);
    }

    [data-testid="stTabs"] [role="tab"] {
        color: var(--sg-muted);
        font-weight: 650;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--sg-primary);
    }

    hr { border-color: var(--sg-border); }
    code, pre { border-color: var(--sg-border); border-radius: 8px; }

    .risk-badge { padding: 0.3rem 0.72rem; border-radius: 999px; font-weight: 700; font-size: 0.78rem; }
    .risk-LOW { background: rgba(37, 157, 111, 0.14); color: #1c9164; }
    .risk-MEDIUM { background: rgba(224, 157, 55, 0.16); color: #a76b12; }
    .risk-HIGH { background: rgba(218, 101, 66, 0.15); color: #ba4b32; }
    .risk-CRITICAL { background: rgba(181, 66, 86, 0.17); color: #ae344f; }

    @media (max-width: 760px) {
        .main .block-container { padding: 1.25rem 0.9rem 2.5rem; }
        .main-header { font-size: 1.9rem; }
        [data-testid="stMetric"] { min-height: 90px; padding: 0.8rem; }
        [data-testid="stMetricValue"] { font-size: 1.28rem; }
        [data-testid="stPlotlyChart"] { border-radius: 9px; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
    }
    </style>
    """ % {**colors, "scheme": "dark" if dark_mode else "light"}
    st.markdown(css, unsafe_allow_html=True)


def render_plotly(fig, **kwargs):
    """Render a Plotly figure using the active dashboard appearance."""
    dark_mode = st.session_state.get("supplyguard_dark_mode", False)
    if dark_mode:
        template = "plotly_dark"
        font_color = "#e8f0f2"
        grid_color = "rgba(210, 230, 235, 0.12)"
    else:
        template = "plotly_white"
        font_color = "#19343b"
        grid_color = "rgba(35, 74, 81, 0.12)"

    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=font_color,
        xaxis={"gridcolor": grid_color},
        yaxis={"gridcolor": grid_color},
    )
    return st.plotly_chart(fig, **kwargs)


def style_risk_table(df, column: str):
    """Color risk-level cells while leaving the displayed data unchanged."""
    palettes = {
        "LOW": ("#e7f5ed", "#18734e"),
        "MEDIUM": ("#fff2d9", "#8c5a08"),
        "HIGH": ("#ffebe4", "#a13d26"),
        "CRITICAL": ("#fce8eb", "#9f3048"),
    }

    if column not in df.columns:
        return df

    def risk_style(value):
        background, foreground = palettes.get(str(value).upper(), ("", ""))
        if not background:
            return ""
        return (
            f"background-color: {background}; color: {foreground}; "
            "font-weight: 700; text-align: center; border-radius: 7px"
        )

    return df.style.apply(lambda values: values.map(risk_style), subset=[column])
