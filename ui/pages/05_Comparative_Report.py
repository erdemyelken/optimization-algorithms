"""
Page 5 — Comparative Report
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
from core.benchmark_runner import BenchmarkRunner
from core.metrics import compute_rank_matrix, summary_table
from ui.components.charts import heatmap_figure, convergence_plot

st.set_page_config(page_title="Comparative Report | OptimBench", page_icon="📋", layout="wide")

st.markdown("## 📋 Comparative Report")
st.caption("Run multiple algorithms across multiple benchmark functions and compare with Friedman-style rankings.")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Batch Configuration")
    selected_funcs = st.multiselect(
        "Benchmark Functions",
        list(BENCHMARK_REGISTRY.keys()),
        default=list(BENCHMARK_REGISTRY.keys())[:4],
    )
    selected_algos = st.multiselect(
        "Algorithms",
        list(ALGORITHM_REGISTRY.keys()),
        default=list(ALGORITHM_REGISTRY.keys())[:4],
    )
    dim = st.select_slider("Dimension", options=[2, 5, 10, 20], value=10)
    n_runs = st.select_slider("Runs per combo", options=[1, 3, 5, 10], value=3)
    base_seed = st.number_input("Base Seed", value=42)
    run_btn = st.button("▶ Run Full Comparison", type="primary", use_container_width=True)

if "cmp_results" not in st.session_state:
    st.session_state.cmp_results = None

if run_btn:
    if not selected_funcs or not selected_algos:
        st.error("Select at least one function and algorithm.")
    else:
        total = len(selected_funcs) * len(selected_algos)
        prog = st.progress(0.0)
        status = st.empty()
        done_c = [0]

        def cb(d, t, msg):
            prog.progress(d / t, msg)
            status.caption(f"🔄 {msg}")

        runner = BenchmarkRunner(progress_callback=cb)
        all_agg = []
        done_outer = 0

        for func_name in selected_funcs:
            func = BENCHMARK_REGISTRY[func_name]
            for algo_name in selected_algos:
                cls = ALGORITHM_REGISTRY[algo_name]
                opt = cls(**cls.get_default_params())
                status.caption(f"Running {algo_name} on {func_name}…")
                agg = runner.run_multiple(opt, func, dim, n_runs=n_runs, base_seed=base_seed)
                all_agg.append(agg)
                done_outer += 1
                prog.progress(done_outer / total, f"{algo_name} × {func_name}")

        prog.progress(1.0, "✅ Done!")
        status.empty()
        st.session_state.cmp_results = all_agg
        st.success(f"Comparison complete: {len(all_agg)} experiments")

if st.session_state.cmp_results:
    all_agg = st.session_state.cmp_results
    all_algos = list(dict.fromkeys(a.algorithm_name for a in all_agg))
    all_funcs = list(dict.fromkeys(a.function_name for a in all_agg))

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "🔥 Heatmap", "📈 Convergence", "📄 Full Table"])

    with tab1:
        ranks = compute_rank_matrix(all_agg)
        df_ranks = pd.DataFrame(
            sorted(ranks.items(), key=lambda x: x[1]),
            columns=["Algorithm", "Average Rank"],
        )
        df_ranks.index = df_ranks.index + 1
        df_ranks.index.name = "Rank"
        df_ranks["Average Rank"] = df_ranks["Average Rank"].map("{:.2f}".format)
        st.markdown("#### Friedman-Style Average Rankings")
        st.caption("Lower rank = better. Averaged across all functions.")
        st.dataframe(df_ranks, use_container_width=True)

        import plotly.graph_objects as go
        sorted_algos = [r[0] for r in sorted(ranks.items(), key=lambda x: x[1])]
        sorted_ranks = [ranks[a] for a in sorted_algos]
        fig_rank = go.Figure(go.Bar(
            x=sorted_algos, y=sorted_ranks,
            marker_color=["#6C63FF" if i == 0 else "#4a4a7a" for i in range(len(sorted_algos))]
        ))
        fig_rank.update_layout(template="plotly_dark", height=350,
                                xaxis_title="Algorithm", yaxis_title="Mean Rank (lower=better)",
                                title="Algorithm Rankings")
        st.plotly_chart(fig_rank, use_container_width=True)

    with tab2:
        metric_hm = st.selectbox("Metric", ["mean_fitness", "std_fitness", "mean_runtime", "median_fitness"])
        # Build algo × function matrix
        mat = np.full((len(all_algos), len(all_funcs)), np.nan)
        for agg in all_agg:
            i = all_algos.index(agg.algorithm_name)
            j = all_funcs.index(agg.function_name)
            mat[i, j] = getattr(agg, metric_hm)

        fig_hm = heatmap_figure(
            mat, all_funcs, all_algos,
            x_title="Function", y_title="Algorithm", metric_title=metric_hm,
            colorscale="Viridis"
        )
        fig_hm.update_layout(title=f"{metric_hm} — Algorithm × Function")
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab3:
        fn_sel = st.selectbox("Show convergence for function", all_funcs)
        subset = [a for a in all_agg if a.function_name == fn_sel]
        if subset:
            fig_conv = convergence_plot(subset, log_scale=True, show_std=True,
                                        title=f"Convergence on {fn_sel} (D={dim})")
            st.plotly_chart(fig_conv, use_container_width=True)

    with tab4:
        rows = summary_table(all_agg)
        df_full = pd.DataFrame(rows).sort_values(["Function", "Mean Fitness"])
        st.dataframe(df_full.style.format({
            "Mean Fitness": "{:.6f}", "Std Fitness": "{:.6f}",
            "Min Fitness": "{:.6f}", "Max Fitness": "{:.6f}",
            "Median": "{:.6f}", "Mean Runtime (s)": "{:.4f}",
        }), use_container_width=True)
        csv_buf = io.StringIO()
        df_full.to_csv(csv_buf, index=False)
        st.download_button("⬇ Download CSV", csv_buf.getvalue(), "comparative_report.csv", "text/csv")
else:
    st.info("Configure the comparison in the sidebar and click **Run Full Comparison**.")
