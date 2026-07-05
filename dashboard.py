"""
dashboard.py
--------------
JAM//WATCH -- RF spectrum anomaly detection console (v2).

Adds on top of v1:
  - Stats overview strip (total scans, anomaly rate, avg error)
  - Gauge meter for the current scan's anomaly score
  - IQ constellation scatter (extra signal-domain view)
  - Error trend chart across recent scans with threshold reference line
  - Batch-scan button to quickly populate history for a live demo
  - CSV export of scan history

Run with:
  python -m streamlit run dashboard.py
"""

import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from autoencoder_model import SpectrogramAutoencoder, pad_to_even_dims
from signal_generator import (
    generate_clean_signal,
    generate_interference_signal,
    generate_jammed_signal,
    SAMPLE_RATE,
)
from spectrogram_pipeline import iq_to_spectrogram

# --------------------------------------------------------------------------
# Page config + theme tokens
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="JAM//WATCH",
    page_icon="\U0001F4E1",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BG_VOID = "#080B0F"
BG_PANEL = "#10151C"
BG_PANEL_RAISED = "#161D26"
BORDER = "#26313E"
BORDER_BRIGHT = "#3A4A5C"
TEXT_PRIMARY = "#EDF2F7"
TEXT_MUTED = "#7C8A99"
TEXT_DIM = "#4A5563"
AMBER = "#FFA940"
AMBER_SOFT = "#8A5A22"
TEAL = "#2DD4BF"
RED = "#FF5C6C"

SIGNAL_GENERATORS = {
    "Clean signal": ("clean", generate_clean_signal),
    "Jammed (CW tone)": ("jammed", generate_jammed_signal),
    "Interference (wideband)": ("interference", generate_interference_signal),
}

# --------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{
            background-color: {BG_VOID};
            background-image:
                radial-gradient(ellipse 900px 500px at 15% 0%, rgba(255,169,64,0.05), transparent),
                linear-gradient(rgba(255,169,64,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,169,64,0.02) 1px, transparent 1px);
            background-size: auto, 26px 26px, 26px 26px;
        }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1240px; }}

        .jw-header {{
            display: flex; justify-content: space-between; align-items: flex-end;
            border-bottom: 1px solid {BORDER}; padding-bottom: 16px; margin-bottom: 22px;
        }}
        .jw-logo {{ font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 800;
            color: {TEXT_PRIMARY}; letter-spacing: -0.5px; }}
        .jw-logo span {{ color: {AMBER}; text-shadow: 0 0 18px rgba(255,169,64,0.5); }}
        .jw-subtitle {{ font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: {TEXT_MUTED};
            letter-spacing: 1.8px; text-transform: uppercase; margin-top: 3px; }}
        .jw-status {{ display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px; color: {TEAL}; letter-spacing: 1px; padding-bottom: 4px; }}
        .jw-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {TEAL};
            box-shadow: 0 0 8px {TEAL}; animation: pulse 1.8s ease-in-out infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}

        .jw-stat {{ background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 6px;
            padding: 14px 16px; text-align: left; }}
        .jw-stat-label {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {TEXT_DIM};
            letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 6px; }}
        .jw-stat-value {{ font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700;
            color: {TEXT_PRIMARY}; }}
        .jw-stat-value.amber {{ color: {AMBER}; }}
        .jw-stat-value.red {{ color: {RED}; }}
        .jw-stat-value.teal {{ color: {TEAL}; }}

        .jw-panel {{ background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
            padding: 20px 22px; height: 100%; box-shadow: 0 4px 24px rgba(0,0,0,0.25); }}
        .jw-panel-label {{ font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: {TEXT_MUTED};
            letter-spacing: 1.6px; text-transform: uppercase; margin-bottom: 14px;
            display: flex; align-items: center; gap: 8px; }}
        .jw-panel-label::before {{ content: ''; width: 3px; height: 12px; background: {AMBER};
            display: inline-block; border-radius: 1px; }}

        .jw-verdict {{ font-family: 'JetBrains Mono', monospace; font-size: 21px; font-weight: 800;
            letter-spacing: 1px; padding: 14px 0; text-align: center; border-radius: 5px; margin-bottom: 14px; }}
        .jw-verdict-normal {{ color: {TEAL}; background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.35); }}
        .jw-verdict-anomaly {{ color: {RED}; background: rgba(255,92,108,0.08); border: 1px solid rgba(255,92,108,0.35);
            animation: alertGlow 1.4s ease-in-out infinite; }}
        @keyframes alertGlow {{
            0%, 100% {{ box-shadow: 0 0 0 rgba(255,92,108,0); }}
            50% {{ box-shadow: 0 0 20px rgba(255,92,108,0.25); }}
        }}

        .jw-metric-row {{ display: flex; justify-content: space-between; align-items: baseline;
            padding: 8px 0; border-bottom: 1px solid {BORDER}; font-family: 'JetBrains Mono', monospace; }}
        .jw-metric-row:last-child {{ border-bottom: none; }}
        .jw-metric-label {{ font-size: 11.5px; color: {TEXT_MUTED}; letter-spacing: 0.5px; }}
        .jw-metric-value {{ font-size: 14px; color: {TEXT_PRIMARY}; font-weight: 500; }}

        div[data-testid="stButton"] > button {{
            font-family: 'JetBrains Mono', monospace; font-size: 12.5px; font-weight: 700;
            letter-spacing: 1px; text-transform: uppercase; border-radius: 5px; padding: 10px 0;
            width: 100%; transition: all 0.15s ease; border: none;
        }}
        div[data-testid="column"]:nth-of-type(3) div[data-testid="stButton"] > button {{
            background: {AMBER}; color: #1A1200;
        }}
        div[data-testid="column"]:nth-of-type(3) div[data-testid="stButton"] > button:hover {{
            background: #FFC26E; box-shadow: 0 0 16px rgba(255,169,64,0.4);
        }}
        div[data-testid="column"]:nth-of-type(4) div[data-testid="stButton"] > button {{
            background: transparent; color: {TEXT_MUTED}; border: 1px solid {BORDER_BRIGHT};
        }}
        div[data-testid="column"]:nth-of-type(4) div[data-testid="stButton"] > button:hover {{
            color: {TEXT_PRIMARY}; border-color: {AMBER};
        }}

        div[data-testid="stSelectbox"] label {{ font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
            color: {TEXT_MUTED}; letter-spacing: 1.2px; text-transform: uppercase; }}

        .jw-log {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; }}
        .jw-log-header {{ display: grid; grid-template-columns: 90px 170px 100px 110px 1fr;
            color: {TEXT_DIM}; letter-spacing: 1px; padding: 6px 4px; border-bottom: 1px solid {BORDER}; }}
        .jw-log-row {{ display: grid; grid-template-columns: 90px 170px 100px 110px 1fr;
            padding: 8px 4px; border-bottom: 1px solid {BORDER}; color: {TEXT_PRIMARY}; }}
        .jw-log-row:hover {{ background: rgba(255,169,64,0.03); }}

        .jw-footer {{ font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: {TEXT_DIM};
            text-align: center; padding-top: 24px; letter-spacing: 0.8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_and_threshold():
    model = SpectrogramAutoencoder()
    model.load_state_dict(torch.load("jamwatch_autoencoder.pt", map_location="cpu"))
    model.eval()
    with open("threshold.txt") as f:
        threshold = float(f.read().strip())
    n_params = sum(p.numel() for p in model.parameters())
    return model, threshold, n_params


def run_scan(model, generator_fn):
    iq_signal = generator_fn(seed=np.random.randint(0, 1_000_000))
    spec, freqs, times = iq_to_spectrogram(iq_signal, SAMPLE_RATE)
    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spec_tensor = pad_to_even_dims(spec_tensor)
    with torch.no_grad():
        reconstructed = model(spec_tensor)
        error = torch.mean((reconstructed - spec_tensor) ** 2).item()
    return iq_signal, spec, freqs, times, error


def make_gauge(error, threshold):
    max_range = max(threshold * 6, error * 1.15, threshold * 3)
    color = RED if error > threshold else TEAL
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=error,
        number={"valueformat": ".5f", "font": {"family": "JetBrains Mono", "color": TEXT_PRIMARY, "size": 26}},
        gauge={
            "axis": {"range": [0, max_range], "tickcolor": TEXT_DIM, "tickfont": {"size": 9, "color": TEXT_DIM}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": BG_PANEL_RAISED,
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold], "color": "rgba(45,212,191,0.12)"},
                {"range": [threshold, max_range], "color": "rgba(255,92,108,0.10)"},
            ],
            "threshold": {"line": {"color": AMBER, "width": 2}, "thickness": 0.9, "value": threshold},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=170, margin=dict(l=25, r=25, t=15, b=5),
        font=dict(family="JetBrains Mono", color=TEXT_MUTED),
    )
    return fig


def make_spectrogram_fig(spec, freqs, times):
    fig = go.Figure(data=go.Heatmap(
        z=spec, x=times * 1000, y=freqs,
        colorscale=[[0.0, BG_VOID], [0.35, "#3D2408"], [0.65, AMBER_SOFT], [0.85, AMBER], [1.0, "#FFE0A8"]],
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(family="JetBrains Mono", color=TEXT_MUTED, size=11),
        margin=dict(l=50, r=15, t=10, b=40), height=290,
        xaxis=dict(title="Time (ms)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Frequency (Hz)", gridcolor=BORDER, zeroline=False),
    )
    return fig


def make_constellation_fig(iq_signal):
    sample = iq_signal[:250]
    magnitude = np.abs(sample)
    fig = go.Figure(data=go.Scatter(
        x=sample.real, y=sample.imag, mode="markers",
        marker=dict(size=5, color=magnitude, colorscale=[[0, AMBER_SOFT], [1, AMBER]], opacity=0.75, line=dict(width=0)),
    ))
    fig.update_layout(
        paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(family="JetBrains Mono", color=TEXT_MUTED, size=10),
        margin=dict(l=40, r=15, t=10, b=35), height=200,
        xaxis=dict(title="I", gridcolor=BORDER, zeroline=True, zerolinecolor=BORDER_BRIGHT),
        yaxis=dict(title="Q", gridcolor=BORDER, zeroline=True, zerolinecolor=BORDER_BRIGHT, scaleanchor="x"),
    )
    return fig


def make_trend_fig(log, threshold):
    entries = list(reversed(log))
    xs = list(range(1, len(entries) + 1))
    errors = [e["error"] for e in entries]
    colors = [RED if e["verdict"] == "ANOMALY" else TEAL for e in entries]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=errors, mode="lines+markers",
        line=dict(color=BORDER_BRIGHT, width=1.5),
        marker=dict(size=9, color=colors, line=dict(width=1, color=BG_PANEL)),
    ))
    fig.add_hline(y=threshold, line=dict(color=AMBER, width=1.5, dash="dash"))
    fig.update_layout(
        paper_bgcolor=BG_PANEL, plot_bgcolor=BG_PANEL,
        font=dict(family="JetBrains Mono", color=TEXT_MUTED, size=10),
        margin=dict(l=45, r=15, t=10, b=30), height=170, showlegend=False,
        xaxis=dict(title="Scan #", gridcolor=BORDER, zeroline=False, dtick=1),
        yaxis=dict(title="Error", gridcolor=BORDER, zeroline=False),
    )
    return fig


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="jw-header">
        <div><div class="jw-logo">JAM<span>//</span>WATCH</div>
        <div class="jw-subtitle">RF spectrum anomaly detection console</div></div>
        <div class="jw-status"><span class="jw-dot"></span>MODEL ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, threshold, n_params = load_model_and_threshold()

if "scan_log" not in st.session_state:
    st.session_state.scan_log = []

# --------------------------------------------------------------------------
# Stats strip
# --------------------------------------------------------------------------
log = st.session_state.scan_log
total_scans = len(log)
anomaly_count = sum(1 for e in log if e["verdict"] == "ANOMALY")
detection_rate = (anomaly_count / total_scans * 100) if total_scans else 0
avg_error = (sum(e["error"] for e in log) / total_scans) if total_scans else 0.0

s1, s2, s3, s4 = st.columns(4)
for col, label, value, cls in [
    (s1, "Total scans", f"{total_scans}", ""),
    (s2, "Anomalies flagged", f"{anomaly_count}", "red" if anomaly_count else ""),
    (s3, "Anomaly rate", f"{detection_rate:.0f}%", "amber" if detection_rate else "teal"),
    (s4, "Avg. error", f"{avg_error:.5f}", ""),
]:
    with col:
        st.markdown(
            f'<div class="jw-stat"><div class="jw-stat-label">{label}</div>'
            f'<div class="jw-stat-value {cls}">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
ctrl_col, spacer, btn_col, btn2_col = st.columns([2.3, 0.1, 1, 1])
with ctrl_col:
    signal_choice = st.selectbox("Signal source", list(SIGNAL_GENERATORS.keys()), label_visibility="visible")
with btn_col:
    st.write("")
    do_scan = st.button("Run scan")
with btn2_col:
    st.write("")
    do_batch = st.button("Batch x10")

if do_scan or do_batch:
    label, generator_fn = SIGNAL_GENERATORS[signal_choice]
    n_runs = 10 if do_batch else 1
    last = None
    for _ in range(n_runs):
        iq_signal, spec, freqs, times, error = run_scan(model, generator_fn)
        is_anomaly = error > threshold
        st.session_state.scan_log.insert(0, {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "label": label, "error": error,
            "verdict": "ANOMALY" if is_anomaly else "NORMAL",
        })
        last = (iq_signal, spec, freqs, times, error, is_anomaly, label)
    st.session_state.scan_log = st.session_state.scan_log[:30]
    iq_signal, spec, freqs, times, error, is_anomaly, label = last
    st.session_state.last_result = {
        "iq": iq_signal, "spec": spec, "freqs": freqs, "times": times,
        "error": error, "is_anomaly": is_anomaly, "label": label,
    }

# --------------------------------------------------------------------------
# Main row: spectrogram + constellation | verdict + gauge
# --------------------------------------------------------------------------
st.write("")
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown('<div class="jw-panel">', unsafe_allow_html=True)
    st.markdown('<div class="jw-panel-label">Live spectrogram</div>', unsafe_allow_html=True)
    if "last_result" in st.session_state:
        r = st.session_state.last_result
        st.plotly_chart(make_spectrogram_fig(r["spec"], r["freqs"], r["times"]),
                         use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            f'<div style="height:290px; display:flex; align-items:center; justify-content:center; '
            f'color:{TEXT_MUTED}; font-family:JetBrains Mono; font-size:13px;">RUN A SCAN TO VIEW SPECTRUM</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="jw-panel">', unsafe_allow_html=True)
    st.markdown('<div class="jw-panel-label">IQ constellation</div>', unsafe_allow_html=True)
    if "last_result" in st.session_state:
        r = st.session_state.last_result
        st.plotly_chart(make_constellation_fig(r["iq"]), use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            f'<div style="height:200px; display:flex; align-items:center; justify-content:center; '
            f'color:{TEXT_MUTED}; font-family:JetBrains Mono; font-size:13px;">NO DATA</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="jw-panel">', unsafe_allow_html=True)
    st.markdown('<div class="jw-panel-label">Detection result</div>', unsafe_allow_html=True)
    if "last_result" in st.session_state:
        r = st.session_state.last_result
        verdict_class = "jw-verdict-anomaly" if r["is_anomaly"] else "jw-verdict-normal"
        verdict_text = "\u26A0 ANOMALY" if r["is_anomaly"] else "\u2713 NORMAL"
        st.markdown(f'<div class="jw-verdict {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(r["error"], threshold), use_container_width=True, config={"displayModeBar": False})
        margin_pct = (r["error"] / threshold - 1) * 100
        st.markdown(
            f"""
            <div class="jw-metric-row"><span class="jw-metric-label">SOURCE</span>
                <span class="jw-metric-value">{r['label'].upper()}</span></div>
            <div class="jw-metric-row"><span class="jw-metric-label">THRESHOLD</span>
                <span class="jw-metric-value">{threshold:.5f}</span></div>
            <div class="jw-metric-row"><span class="jw-metric-label">DEVIATION</span>
                <span class="jw-metric-value">{margin_pct:+.0f}%</span></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="color:{TEXT_MUTED}; font-family:JetBrains Mono; font-size:12px; '
            f'text-align:center; padding:50px 0;">AWAITING FIRST SCAN</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Trend chart
# --------------------------------------------------------------------------
st.write("")
st.markdown('<div class="jw-panel">', unsafe_allow_html=True)
st.markdown('<div class="jw-panel-label">Error trend (recent scans)</div>', unsafe_allow_html=True)
if len(st.session_state.scan_log) >= 2:
    st.plotly_chart(make_trend_fig(st.session_state.scan_log[:20], threshold),
                     use_container_width=True, config={"displayModeBar": False})
else:
    st.markdown(
        f'<div style="color:{TEXT_MUTED}; font-family:JetBrains Mono; font-size:12px; padding:12px 0;">'
        f"RUN AT LEAST 2 SCANS TO SEE A TREND</div>",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Scan history log + export
# --------------------------------------------------------------------------
st.write("")
st.markdown('<div class="jw-panel">', unsafe_allow_html=True)
hdr_col, dl_col = st.columns([5, 1])
with hdr_col:
    st.markdown('<div class="jw-panel-label">Scan history</div>', unsafe_allow_html=True)
with dl_col:
    if st.session_state.scan_log:
        csv_lines = ["time,source,error,verdict"]
        for e in st.session_state.scan_log:
            csv_lines.append(f'{e["time"]},{e["label"]},{e["error"]:.6f},{e["verdict"]}')
        st.download_button("Export CSV", "\n".join(csv_lines), file_name="jamwatch_scan_log.csv",
                            use_container_width=True)

if st.session_state.scan_log:
    rows_html = '<div class="jw-log">'
    rows_html += ('<div class="jw-log-header"><span>TIME</span><span>SOURCE</span>'
                  '<span>ERROR</span><span>VERDICT</span><span></span></div>')
    for entry in st.session_state.scan_log[:12]:
        color = RED if entry["verdict"] == "ANOMALY" else TEAL
        rows_html += (
            f'<div class="jw-log-row"><span>{entry["time"]}</span><span>{entry["label"]}</span>'
            f'<span>{entry["error"]:.5f}</span>'
            f'<span style="color:{color}; font-weight:700;">{entry["verdict"]}</span><span></span></div>'
        )
    rows_html += "</div>"
    st.markdown(rows_html, unsafe_allow_html=True)
else:
    st.markdown(
        f'<div style="color:{TEXT_MUTED}; font-family:JetBrains Mono; font-size:12px; padding:12px 0;">'
        f"NO SCANS YET</div>",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="jw-footer">JAM//WATCH v0.2 &nbsp;&middot;&nbsp; convolutional autoencoder '
    f"&nbsp;&middot;&nbsp; {n_params:,} parameters &nbsp;&middot;&nbsp; threshold {threshold:.5f}</div>",
    unsafe_allow_html=True,
)
