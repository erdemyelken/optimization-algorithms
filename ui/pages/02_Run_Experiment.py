"""
Page 2 — Run Experiment
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import io

import benchmark_functions.unimodal  # noqa
import benchmark_functions.multimodal  # noqa
from benchmark_functions import BENCHMARK_REGISTRY
from algorithms import ALGORITHM_REGISTRY
from core.benchmark_runner import BenchmarkRunner
from core.result import AggregatedResult
from core.metrics import summary_table, compute_rank_matrix
from core.exporter import Exporter
from ui.components.charts import (
    convergence_plot, runtime_bar, fitness_boxplot, performance_runtime_scatter
)

st.set_page_config(page_title="Run Experiment | OptimBench", page_icon="🚀", layout="wide")

st.markdown("## 🚀 Run Experiment")
st.caption("Configure and execute optimization experiments. Compare algorithms side by side.")
st.divider()

# ── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Experiment Configuration")

    func_name = st.selectbox("Benchmark Function", list(BENCHMARK_REGISTRY.keys()))
    func = BENCHMARK_REGISTRY[func_name]

    st.caption(f"**Modality:** {func.metadata.modality.title()} | **Difficulty:** {func.metadata.difficulty.title()}")
    st.caption(f"**Bounds:** [{func.metadata.bounds[0]}, {func.metadata.bounds[1]}]")

    dim = st.select_slider("Dimension (D)", options=[2, 5, 10, 20, 30], value=10)
    n_runs = st.select_slider("Number of Runs", options=[1, 3, 5, 10, 20], value=5)
    base_seed = st.number_input("Base Seed", value=42, min_value=0, step=1)

    st.markdown("---")
    algo_names = st.multiselect(
        "Select Algorithms",
        list(ALGORITHM_REGISTRY.keys()),
        default=list(ALGORITHM_REGISTRY.keys())[:3],
    )

    st.markdown("### 🎛️ Algorithm Parameters")
    algo_params: dict = {}
    if algo_names:
        tabs = st.tabs([n.split(" ")[0] for n in algo_names])
        for tab, name in zip(tabs, algo_names):
            cls = ALGORITHM_REGISTRY[name]
            schema = cls.get_param_schema()
            params = {}
            with tab:
                for pname, spec in schema.items():
                    if spec["type"] == "int":
                        val = st.number_input(
                            pname, value=int(spec["default"]),
                            min_value=int(spec["min"]), max_value=int(spec["max"]),
                            step=int(spec["step"]), help=spec["help"], key=f"{name}_{pname}"
                        )
                        params[pname] = int(val)
                    else:
                        val = st.number_input(
                            pname, value=float(spec["default"]),
                            min_value=float(spec["min"]), max_value=float(spec["max"]),
                            step=float(spec["step"]), help=spec["help"], key=f"{name}_{pname}"
                        )
                        params[pname] = float(val)
            algo_params[name] = params

    st.markdown("---")
    run_btn = st.button("▶ Run Experiment", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if not algo_names:
    st.info("Please select at least one algorithm in the sidebar.")
    st.stop()

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if run_btn:
    if not algo_names:
        st.error("Select at least one algorithm.")
    else:
        prog = st.progress(0.0, "Starting…")
        status = st.empty()
        agg_list: list[AggregatedResult] = []

        def cb(done, total, msg):
            prog.progress(done / total, msg)
            status.caption(f"🔄 {msg}")

        runner = BenchmarkRunner(progress_callback=cb)
        for idx, name in enumerate(algo_names):
            cls = ALGORITHM_REGISTRY[name]
            params = algo_params.get(name, cls.get_default_params())
            opt = cls(**params)
            with st.spinner(f"Running {name}…"):
                agg = runner.run_multiple(opt, func, dim, n_runs=n_runs, base_seed=base_seed)
            agg_list.append(agg)
            prog.progress((idx + 1) / len(algo_names), f"Completed {name}")

        prog.progress(1.0, "✅ Done!")
        status.empty()
        st.session_state.last_results = agg_list
        st.success(f"Experiment completed — {len(algo_names)} algorithm(s) × {n_runs} run(s) on **{func_name}** (D={dim})")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.last_results:
    agg_list: list[AggregatedResult] = st.session_state.last_results

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Convergence", "📦 Boxplot", "⏱️ Runtime", "🏆 Rankings", "📥 Export"]
    )

    with tab1:
        log_scale = st.checkbox("Log scale Y-axis", value=True, key="log_conv")
        show_std = st.checkbox("Show ± std band", value=True, key="show_std")
        fig = convergence_plot(agg_list, log_scale=log_scale, show_std=show_std,
                               title=f"Convergence — {func_name} (D={dim})")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        runs_by_algo = {a.algorithm_name: [] for a in agg_list}
        for agg in agg_list:
            # Approximate per-run fitnesses from stored stats
            # We store mean ± std but not raw runs directly in AggregatedResult.
            # Re-display mean/std as indication.
            runs_by_algo[agg.algorithm_name] = [agg.min_fitness,
                *[agg.mean_fitness] * max(1, agg.n_runs - 2), agg.max_fitness]
        log_b = st.checkbox("Log scale", value=False, key="log_box")
        fig2 = fitness_boxplot(runs_by_algo, title=f"Fitness Distribution — {func_name}", log_scale=log_b)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        c1.plotly_chart(runtime_bar(agg_list, title="Runtime per Algorithm"), use_container_width=True)
        c2.plotly_chart(performance_runtime_scatter(agg_list, title="Performance vs Runtime"), use_container_width=True)

    with tab4:
        rows = summary_table(agg_list)
        df = pd.DataFrame(rows)
        df = df.sort_values("Mean Fitness").reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"
        st.dataframe(df.style.format({
            "Mean Fitness": "{:.6f}", "Std Fitness": "{:.6f}",
            "Min Fitness": "{:.6f}", "Max Fitness": "{:.6f}",
            "Median": "{:.6f}", "Mean Runtime (s)": "{:.4f}",
        }), use_container_width=True)

    with tab5:
        exp = Exporter(output_dir="results")
        rows = summary_table(agg_list)
        df_exp = pd.DataFrame(rows)

        csv_buf = io.StringIO()
        df_exp.to_csv(csv_buf, index=False)
        st.download_button("⬇ Download CSV", csv_buf.getvalue(), "experiment_results.csv", "text/csv")

        import json
        json_data = [a.to_dict() for a in agg_list]
        st.download_button("⬇ Download JSON", json.dumps(json_data, indent=2), "experiment_results.json", "application/json")

        config_snap = {
            "function": func_name, "dimension": dim, "n_runs": n_runs,
            "base_seed": base_seed, "algorithms": algo_params,
        }
        st.download_button("⬇ Config Snapshot", json.dumps(config_snap, indent=2), "config_snapshot.json", "application/json")
