"""
Page 4 — Runtime Analysis
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import io

import benchmark_functions.unimodal  # noqa
import benchmark_functions.multimodal  # noqa
from benchmark_functions import BENCHMARK_REGISTRY
from algorithms import ALGORITHM_REGISTRY
from analysis.runtime_analysis import RuntimeAnalyzer
from ui.components.charts import dimension_scaling_plot, performance_runtime_scatter

st.set_page_config(page_title="Runtime Analysis | OptimBench", page_icon="⏱️", layout="wide")

st.markdown("## ⏱️ Runtime Analysis")
st.caption("Analyse how algorithms scale with dimension and iteration budget.")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    func_name = st.selectbox("Benchmark Function", list(BENCHMARK_REGISTRY.keys()))
    func = BENCHMARK_REGISTRY[func_name]

    algo_names = st.multiselect("Algorithms", list(ALGORITHM_REGISTRY.keys()),
                                  default=list(ALGORITHM_REGISTRY.keys())[:3])
    n_runs = st.select_slider("Runs per point", options=[1, 3, 5], value=3)
    base_seed = st.number_input("Base Seed", value=42, min_value=0)

    st.markdown("### 📏 Analysis Type")
    analysis_type = st.radio("Type", ["Dimension Scaling", "Budget (Iteration) Scaling"])

    if analysis_type == "Dimension Scaling":
        dims = st.multiselect("Dimensions", [2, 5, 10, 20, 30, 50], default=[2, 5, 10, 20])
        fixed_iter = st.number_input("Fixed max_iter", value=100, min_value=10, step=10)
    else:
        fixed_dim = st.select_slider("Fixed Dimension", options=[2, 5, 10, 20], value=10)
        budgets = st.multiselect("Iteration Budgets", [50, 100, 200, 500, 1000], default=[50, 100, 200, 500])

    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

if "rt_records" not in st.session_state:
    st.session_state.rt_records = None

if run_btn and algo_names:
    prog = st.progress(0.0, "Running…")

    def cb(done, total, msg):
        prog.progress(done / total, msg)

    analyzer = RuntimeAnalyzer(progress_callback=cb)
    optimizers_with_params = [
        (ALGORITHM_REGISTRY[n], ALGORITHM_REGISTRY[n].get_default_params())
        for n in algo_names
    ]

    if analysis_type == "Dimension Scaling":
        # Override max_iter to fixed_iter
        optimizers_with_params_adj = [
            (cls, {**params, "max_iter": int(fixed_iter)})
            for cls, params in optimizers_with_params
        ]
        records = analyzer.dimension_scaling(
            optimizers_with_params_adj, func, sorted(dims), n_runs=n_runs, base_seed=base_seed
        )
    else:
        records = analyzer.budget_scaling(
            optimizers_with_params, func, int(fixed_dim), sorted(budgets), n_runs=n_runs, base_seed=base_seed
        )

    prog.progress(1.0, "✅ Done!")
    st.session_state.rt_records = (records, analysis_type)
    st.success("Analysis complete!")

elif run_btn:
    st.error("Select at least one algorithm.")

if st.session_state.rt_records:
    records, a_type = st.session_state.rt_records
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Runtime Plot", "🎯 Fitness Plot", "🏅 Efficiency", "📄 Data"])

    with tab1:
        x_label = "Dimension" if a_type == "Dimension Scaling" else "Max Iterations"
        x_attr = "dimension" if a_type == "Dimension Scaling" else "max_iter"
        if a_type == "Dimension Scaling":
            fig = dimension_scaling_plot(records, y="mean_runtime", y_label="Mean Runtime (s)",
                                          title=f"Runtime vs Dimension — {func_name}")
        else:
            import plotly.graph_objects as go
            from collections import defaultdict
            by_algo = defaultdict(list)
            for r in records:
                by_algo[r.algorithm_name].append(r)
            fig = go.Figure()
            for algo, recs in by_algo.items():
                recs = sorted(recs, key=lambda r: r.max_iter)
                fig.add_trace(go.Scatter(
                    x=[r.max_iter for r in recs], y=[r.mean_runtime for r in recs],
                    mode="lines+markers", name=algo,
                    error_y=dict(type="data", array=[r.std_runtime for r in recs], visible=True)
                ))
            fig.update_layout(template="plotly_dark", height=420,
                               xaxis_title="Max Iterations", yaxis_title="Mean Runtime (s)",
                               title=f"Budget Scaling — {func_name}")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if a_type == "Dimension Scaling":
            fig2 = dimension_scaling_plot(records, y="mean_fitness", y_label="Mean Fitness",
                                           title=f"Fitness vs Dimension — {func_name}")
        else:
            import plotly.graph_objects as go
            from collections import defaultdict
            by_algo = defaultdict(list)
            for r in records:
                by_algo[r.algorithm_name].append(r)
            fig2 = go.Figure()
            for algo, recs in by_algo.items():
                recs = sorted(recs, key=lambda r: r.max_iter)
                fig2.add_trace(go.Scatter(
                    x=[r.max_iter for r in recs], y=[r.mean_fitness for r in recs],
                    mode="lines+markers", name=algo))
            fig2.update_layout(template="plotly_dark", height=420,
                                xaxis_title="Max Iterations", yaxis_title="Mean Fitness")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        eff = RuntimeAnalyzer().efficiency_ranking(records)
        df_eff = pd.DataFrame(eff)
        df_eff.index = df_eff.index + 1
        df_eff.index.name = "Rank"
        st.dataframe(df_eff.style.format({
            "Avg Mean Fitness": "{:.6f}", "Avg Mean Runtime (s)": "{:.4f}",
            "Efficiency Score": "{:.6f}"
        }), use_container_width=True)
        st.caption("Lower Efficiency Score = better trade-off (fitness × runtime).")

    with tab4:
        rows = [r.to_dict() for r in records]
        df_data = pd.DataFrame(rows)
        st.dataframe(df_data, use_container_width=True)
        csv_buf = io.StringIO()
        df_data.to_csv(csv_buf, index=False)
        st.download_button("⬇ CSV", csv_buf.getvalue(), "runtime_analysis.csv", "text/csv")
else:
    st.info("Configure the analysis in the sidebar and click **Run Analysis**.")
