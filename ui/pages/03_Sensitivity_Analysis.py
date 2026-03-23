"""
Page 3 — Sensitivity Analysis
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import io, json

import benchmark_functions.unimodal  # noqa
import benchmark_functions.multimodal  # noqa
from benchmark_functions import BENCHMARK_REGISTRY
from algorithms import ALGORITHM_REGISTRY
from analysis.sensitivity_analysis import SensitivityAnalyzer
from ui.components.charts import heatmap_figure

st.set_page_config(page_title="Sensitivity Analysis | OptimBench", page_icon="🎛️", layout="wide")

st.markdown("## 🎛️ Sensitivity Analysis")
st.caption("Explore how algorithm parameters affect performance through systematic parameter sweeps.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Sweep Configuration")
    algo_name = st.selectbox("Algorithm", list(ALGORITHM_REGISTRY.keys()))
    cls = ALGORITHM_REGISTRY[algo_name]
    schema = cls.get_param_schema()

    func_name = st.selectbox("Benchmark Function", list(BENCHMARK_REGISTRY.keys()))
    func = BENCHMARK_REGISTRY[func_name]
    dim = st.select_slider("Dimension", options=[2, 5, 10, 20], value=10)
    n_runs = st.select_slider("Runs per combo", options=[1, 3, 5, 10], value=3)
    base_seed = st.number_input("Base Seed", value=42, min_value=0)

    st.markdown("### 🎯 Sweep Mode")
    sweep_mode = st.radio("Mode", ["1D Sweep (one param)", "2D Sweep (two params — heatmap)"])

    numeric_params = {k: v for k, v in schema.items() if v["type"] in ("int", "float")}
    param_names = list(numeric_params.keys())

    if sweep_mode.startswith("1D"):
        p1_name = st.selectbox("Parameter to sweep", param_names)
        p1_spec = numeric_params[p1_name]
        if p1_spec["type"] == "int":
            p1_min = st.number_input("Min", value=int(p1_spec["min"]), step=1)
            p1_max = st.number_input("Max", value=int(p1_spec["max"]), step=1)
            p1_steps = st.number_input("Steps", value=8, min_value=2, max_value=30, step=1)
            p1_vals = list(np.linspace(p1_min, p1_max, int(p1_steps), dtype=int).tolist())
            p1_vals = list(dict.fromkeys(p1_vals))  # deduplicate
        else:
            p1_min = st.number_input("Min", value=float(p1_spec["min"]))
            p1_max = st.number_input("Max", value=float(p1_spec["max"]))
            p1_steps = st.number_input("Steps", value=8, min_value=2, max_value=30, step=1)
            p1_vals = np.linspace(p1_min, p1_max, int(p1_steps)).tolist()
        metric_1d = st.selectbox("Metric", ["mean_fitness", "best_fitness", "std_fitness", "mean_runtime"])
        p2_name = None
    else:
        p1_name = st.selectbox("Parameter 1 (x-axis)", param_names, index=0)
        p2_name = st.selectbox("Parameter 2 (y-axis)", param_names, index=min(1, len(param_names) - 1))
        p1_spec = numeric_params[p1_name]
        p2_spec = numeric_params[p2_name]

        def _range(spec, n=5):
            t = spec["type"]
            vals = np.linspace(spec["min"], spec["max"], n)
            return list(vals.astype(int).tolist()) if t == "int" else vals.tolist()

        p1_n = st.slider(f"# values for {p1_name}", 3, 10, 5)
        p2_n = st.slider(f"# values for {p2_name}", 3, 10, 5)
        p1_vals = _range(p1_spec, p1_n)
        p2_vals = _range(p2_spec, p2_n)
        hm_metric = st.selectbox("Heatmap Metric", ["mean_fitness", "std_fitness", "mean_runtime", "best_fitness"])

    run_btn = st.button("▶ Run Sweep", type="primary", use_container_width=True)

# ── Execution ─────────────────────────────────────────────────────────────────
if "sa_result" not in st.session_state:
    st.session_state.sa_result = None

if run_btn:
    prog = st.progress(0.0, "Starting sweep…")
    status = st.empty()

    def cb(done, total, msg):
        prog.progress(done / total, msg)
        status.caption(f"🔄 {msg}")

    analyzer = SensitivityAnalyzer(progress_callback=cb)

    if sweep_mode.startswith("1D"):
        varied = {p1_name: p1_vals}
    else:
        varied = {p1_name: p1_vals, p2_name: p2_vals}

    result = analyzer.sweep(cls, func, dim, varied_params=varied, n_runs=n_runs, base_seed=base_seed)
    prog.progress(1.0, "✅ Done!")
    status.empty()
    st.session_state.sa_result = result
    st.success(f"Sweep complete: {len(result.results)} combo(s)")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.sa_result:
    res = st.session_state.sa_result

    if sweep_mode.startswith("1D"):
        tab1, tab2 = st.tabs(["📈 1D Sweep Chart", "📄 Data Table"])
        with tab1:
            import plotly.graph_objects as go
            x_vals = [r.param_values[p1_name] for r in res.results]
            y_vals = [getattr(r, metric_1d) for r in res.results]
            std_vals = [r.std_fitness for r in res.results]

            fig = go.Figure()
            if metric_1d in ("mean_fitness", "best_fitness"):
                fig.add_trace(go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=[m + s for m, s in zip(y_vals, std_vals)] + [m - s for m, s in zip(y_vals, std_vals)][::-1],
                    fill="toself", fillcolor="rgba(108,99,255,0.15)", line=dict(width=0),
                    showlegend=False, hoverinfo="skip"
                ))
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="lines+markers",
                marker=dict(size=8, color="#6C63FF"), line=dict(color="#6C63FF", width=2),
                name=metric_1d
            ))
            fig.update_layout(
                title=f"{algo_name} — {metric_1d} vs {p1_name}",
                xaxis_title=p1_name, yaxis_title=metric_1d,
                template="plotly_dark", height=420
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            df = pd.DataFrame(res.to_records())
            st.dataframe(df, use_container_width=True)
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("⬇ CSV", csv_buf.getvalue(), "sensitivity_1d.csv", "text/csv")

    else:
        tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "📈 Line View", "📄 Data Table"])
        with tab1:
            col_metric = hm_metric
            mat, px_vals, py_vals = res.to_matrix(p1_name, p2_name, col_metric)
            colorscale_map = {
                "mean_fitness": "Viridis", "best_fitness": "Viridis",
                "std_fitness": "YlOrRd", "mean_runtime": "Blues"
            }
            reverse_map = {"mean_fitness": False, "best_fitness": False, "std_fitness": False, "mean_runtime": False}
            fig_hm = heatmap_figure(
                mat, px_vals, py_vals,
                x_title=p1_name, y_title=p2_name, metric_title=col_metric,
                colorscale=colorscale_map.get(col_metric, "Viridis"),
            )
            fig_hm.update_layout(title=f"{col_metric} Heatmap — {algo_name} on {func_name}")
            st.plotly_chart(fig_hm, use_container_width=True)

        with tab2:
            import plotly.graph_objects as go
            fig_line = go.Figure()
            for p2v in sorted(set(r.param_values[p2_name] for r in res.results)):
                subset = sorted([r for r in res.results if r.param_values[p2_name] == p2v],
                                 key=lambda r: r.param_values[p1_name])
                xs = [r.param_values[p1_name] for r in subset]
                ys = [getattr(r, hm_metric) for r in subset]
                fig_line.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"{p2_name}={p2v:.3g}"))
            fig_line.update_layout(template="plotly_dark", height=400,
                                    xaxis_title=p1_name, yaxis_title=hm_metric)
            st.plotly_chart(fig_line, use_container_width=True)

        with tab3:
            df = pd.DataFrame(res.to_records())
            st.dataframe(df, use_container_width=True)
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("⬇ CSV", csv_buf.getvalue(), "sensitivity_2d.csv", "text/csv")
else:
    st.info("Configure the sweep in the sidebar and click **Run Sweep**.")
