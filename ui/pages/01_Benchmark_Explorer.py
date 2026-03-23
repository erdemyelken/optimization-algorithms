"""
Page 1 — Benchmark Explorer
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import benchmark_functions.unimodal  # noqa
import benchmark_functions.multimodal  # noqa
from benchmark_functions import BENCHMARK_REGISTRY, list_functions
from ui.components.charts import landscape_plot

st.set_page_config(page_title="Benchmark Explorer | OptimBench", page_icon="📐", layout="wide")

st.markdown("## 📐 Benchmark Explorer")
st.caption("Browse, filter, and visualise the benchmark function library.")
st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
modality_filter = col1.selectbox("Modality", ["All", "unimodal", "multimodal"])
difficulty_filter = col2.selectbox("Difficulty", ["All", "easy", "medium", "hard"])
search = col3.text_input("Search by name", "")

funcs = list_functions(
    modality=None if modality_filter == "All" else modality_filter,
    difficulty=None if difficulty_filter == "All" else difficulty_filter,
)
if search:
    funcs = [f for f in funcs if search.lower() in f.name.lower()]

st.markdown(f"**{len(funcs)} function(s) found**")
st.divider()

# ── Function cards ────────────────────────────────────────────────────────────
DIFF_COLOR = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
MOD_LABEL = {"unimodal": "📈 Unimodal", "multimodal": "🌊 Multimodal"}

for func in funcs:
    with st.expander(
        f"{DIFF_COLOR.get(func.metadata.difficulty, '⚪')} **{func.name}** "
        f"— {MOD_LABEL.get(func.metadata.modality, func.metadata.modality)} | "
        f"Difficulty: {func.metadata.difficulty.title()}",
        expanded=False,
    ):
        mc1, mc2 = st.columns([2, 3])
        with mc1:
            st.markdown(f"**Formula:** `{func.metadata.formula}`")
            st.markdown(f"**Global Optimum:** `{func.metadata.global_optimum}` at `{func.metadata.global_optimum_location}`")
            st.markdown(f"**Default Bounds:** `[{func.metadata.bounds[0]}, {func.metadata.bounds[1]}]` per dimension")
            st.markdown(f"**Separability:** {func.metadata.separability.title()}")
            st.markdown(f"**Recommended Dims:** {', '.join(str(d) for d in func.metadata.recommended_dims)}")
            st.info(func.metadata.description)

        with mc2:
            st.markdown("**2D Landscape**")
            try:
                res = st.slider(
                    "Resolution", 40, 200, 80, step=20, key=f"res_{func.name}"
                )
                fig = landscape_plot(func, resolution=res)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render landscape: {e}")

st.divider()
st.caption("Tip: Select a function here, then head to **Run Experiment** to benchmark algorithms against it.")
