"""
Optimization Benchmark Laboratory — Home Page
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# Trigger registry population
import benchmark_functions.unimodal  # noqa: F401
import benchmark_functions.multimodal  # noqa: F401
from benchmark_functions import BENCHMARK_REGISTRY
from algorithms import ALGORITHM_REGISTRY

st.set_page_config(
    page_title="OptimBench Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.2rem;
    color: #aab4c8;
    margin-top: 0.5rem;
    font-weight: 300;
}
.metric-card {
    background: linear-gradient(135deg, #1A1D2E, #252849);
    border: 1px solid #2E3260;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-num { font-size: 2.4rem; font-weight: 700; color: #6C63FF; }
.metric-label { font-size: 0.85rem; color: #aab4c8; text-transform: uppercase; letter-spacing: 0.08em; }
.feature-card {
    background: #1A1D2E;
    border: 1px solid #2E3260;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.feature-icon { font-size: 1.5rem; margin-right: 0.5rem; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-mm { background: #3b1f6e; color: #b794f4; }
.badge-um { background: #1a3a6e; color: #90cdf4; }
.badge-hard { background: #4a1527; color: #fc8181; }
.badge-med { background: #3d3209; color: #f6e05e; }
.badge-easy { background: #1a3e2a; color: #68d391; }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">⚡ OptimBench Lab</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">A professional optimization benchmark laboratory for metaheuristic algorithms.<br>'
    'Compare, analyse, and understand algorithm behaviour across classic benchmark functions.</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Quick stats ───────────────────────────────────────────────────────────────
cols = st.columns(4)
stats = [
    (len(ALGORITHM_REGISTRY), "Algorithms"),
    (len(BENCHMARK_REGISTRY), "Benchmark Functions"),
    ("∞", "Experiment Configurations"),
    ("5", "Analysis Modules"),
]
for col, (num, label) in zip(cols, stats):
    col.markdown(
        f'<div class="metric-card"><div class="metric-num">{num}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("### ")

# ── Feature overview ──────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown("#### 🔬 Core Features")
    features = [
        ("📊", "Benchmark Explorer", "View & filter 9 benchmark functions with 2D landscape plots."),
        ("🚀", "Run Experiments", "Single or multi-run experiments with mean ± std convergence curves."),
        ("🎛️", "Sensitivity Analysis", "1D/2D parameter sweeps with interactive heatmaps."),
        ("⏱️", "Runtime Analysis", "Dimension & budget scaling, efficiency ranking."),
        ("📋", "Comparative Report", "Algorithm rankings, stat tables, and Friedman-style analysis."),
    ]
    for icon, name, desc in features:
        st.markdown(
            f'<div class="feature-card"><span class="feature-icon">{icon}</span>'
            f'<strong>{name}</strong><br><span style="color:#aab4c8;font-size:0.85rem;">{desc}</span></div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown("#### 🧮 Available Algorithms")
    for name in ALGORITHM_REGISTRY:
        st.markdown(f"- **{name}**")

    st.markdown("#### 📐 Benchmark Library")
    from benchmark_functions import list_functions
    all_funcs = list_functions()
    for f in all_funcs:
        mm_cls = "badge-mm" if f.metadata.modality == "multimodal" else "badge-um"
        mm_txt = "MM" if f.metadata.modality == "multimodal" else "UM"
        diff_cls = f"badge-{f.metadata.difficulty[:3]}"
        st.markdown(
            f'<span class="badge {mm_cls}">{mm_txt}</span>'
            f'<span class="badge {diff_cls}">{f.metadata.difficulty}</span> '
            f'**{f.name}** — {f.metadata.bounds[0]} ≤ xᵢ ≤ {f.metadata.bounds[1]}',
            unsafe_allow_html=True,
        )

st.divider()

# ── Quick Start ───────────────────────────────────────────────────────────────
st.markdown("#### ⚡ Quick Start")
c1, c2, c3 = st.columns(3)
c1.info("1️⃣ **Select** a benchmark function in _Benchmark Explorer_")
c2.info("2️⃣ **Configure** algorithms & parameters in _Run Experiment_")
c3.info("3️⃣ **Analyse** results with Sensitivity / Runtime / Reports")

st.caption(
    "OptimBench Lab • Built for researchers, students, and optimization enthusiasts • "
    "MIT License"
)
