# OptimBench Lab

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Algorithms](https://img.shields.io/badge/Algorithms-6-purple)
![Functions](https://img.shields.io/badge/Benchmark%20Functions-9-teal)

**A professional, interactive optimization benchmark laboratory for metaheuristic algorithms.**

*Compare algorithms · Analyse sensitivity · Profile runtime · Generate publication-quality plots*

</div>

---

## Overview

**OptimBench Lab** transforms a classic algorithm collection into a full **optimization experimentation platform**. It is designed for researchers, students, and developers who want to:

- Benchmark metaheuristic algorithms against classic test functions
- Run systematic parameter sensitivity studies
- Understand runtime–performance trade-offs
- Generate publication-quality convergence curves and comparison charts
- Export results to CSV, JSON, and PNG for further analysis

---

## Features

| Feature | Description |
|---|---|
| 📐 **Benchmark Explorer** | Browse 9 functions with metadata, 2D landscape plots, and smart filtering |
| 🚀 **Run Experiments** | Multi-algorithm, multi-run experiments with mean ± std convergence curves |
| 🎛️ **Sensitivity Analysis** | 1D parameter sweeps and 2D heatmaps (fitness, runtime, std) |
| ⏱️ **Runtime Analysis** | Dimension scaling and budget scaling plots with efficiency ranking |
| 📋 **Comparative Report** | Friedman-style rankings across algorithms × functions |
| 📥 **Export System** | CSV, JSON, PNG — single-click from every results view |

---

## Architecture

```
optimization-algorithms/
├── algorithms/              # 6 metaheuristic algorithms (unified interface)
│   ├── base_optimizer.py    # Abstract BaseOptimizer
│   ├── genetic_algorithm/   # GA — real-valued, tournament selection
│   ├── particle_swarm/      # PSO — inertia weight decay
│   ├── grey_wolf/           # GWO — Mirjalili et al. (2014)
│   ├── whale_optimization/  # WOA — Mirjalili & Lewis (2016)
│   ├── cuckoo_search/       # CS — Yang & Deb (2009)
│   └── firefly/             # FA — Yang (2009)
├── benchmark_functions/     # 9 benchmark functions + registry
│   ├── unimodal.py          # Sphere, Rosenbrock, Zakharov
│   └── multimodal.py        # Rastrigin, Ackley, Griewank, Schwefel, Levy, Styblinski-Tang
├── core/                    # Orchestration & utilities
│   ├── result.py            # OptimizationResult / AggregatedResult dataclasses
│   ├── benchmark_runner.py  # Single/multi-run experiment runner
│   ├── metrics.py           # Rankings, efficiency, summary tables
│   └── exporter.py          # CSV / JSON / PNG export
├── analysis/                # Analytical modules
│   ├── sensitivity_analysis.py  # 1D/2D parameter sweeps
│   └── runtime_analysis.py      # Dimension & budget scaling
├── ui/                      # Streamlit application
│   ├── app.py               # Home page
│   ├── components/charts.py # Reusable Plotly chart functions
│   └── pages/               # 5 interactive pages
├── configs/                 # YAML experiment presets
├── tests/                   # pytest test suite
└── results/                 # Auto-generated outputs
```

---

## Supported Algorithms

| Algorithm | Reference | Key Parameters |
|---|---|---|
| **Genetic Algorithm** | Holland (1975) | pop_size, mutation_rate, crossover_rate |
| **Particle Swarm Optimization** | Kennedy & Eberhart (1995) | w, c1, c2, pop_size |
| **Grey Wolf Optimizer** | Mirjalili et al. (2014) | pop_size, max_iter |
| **Whale Optimization Algorithm** | Mirjalili & Lewis (2016) | pop_size, b, max_iter |
| **Cuckoo Search** | Yang & Deb (2009) | pa, beta, pop_size |
| **Firefly Algorithm** | Yang (2009) | alpha, beta0, gamma |

---

## Benchmark Functions

| Function | Type | Difficulty | Bounds |
|---|---|---|---|
| Sphere | Unimodal · Separable | Easy | [-5.12, 5.12] |
| Rosenbrock | Unimodal · Non-sep | Medium | [-5, 10] |
| Zakharov | Unimodal · Non-sep | Medium | [-5, 10] |
| Rastrigin | Multimodal · Separable | Hard | [-5.12, 5.12] |
| Ackley | Multimodal · Non-sep | Hard | [-32.768, 32.768] |
| Griewank | Multimodal · Non-sep | Medium | [-600, 600] |
| Schwefel | Multimodal · Separable | Hard | [-500, 500] |
| Levy | Multimodal · Non-sep | Medium | [-10, 10] |
| Styblinski-Tang | Multimodal · Separable | Medium | [-5, 5] |

Each function includes: formula, global optimum, recommended dimensions, modality, separability, and difficulty level.

---

## Installation

```bash
git clone https://github.com/your-username/optimization-algorithms.git
cd optimization-algorithms
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Launch the interactive UI
streamlit run ui/app.py

# Run the test suite
pytest tests/ -v
```

---

## Usage Examples

### Python API — Run a single experiment

```python
from benchmark_functions.multimodal import Rastrigin
from algorithms.particle_swarm.pso import ParticleSwarmOptimization
from core.benchmark_runner import BenchmarkRunner

func = Rastrigin()
opt = ParticleSwarmOptimization(pop_size=30, max_iter=200, w=0.7, c1=1.5, c2=1.5)
runner = BenchmarkRunner()

# Single run
result = runner.run_single(opt, func, dim=10, seed=42)
print(f"Best fitness: {result.best_fitness:.6f} in {result.runtime_seconds:.3f}s")

# Multi-run with statistics
agg = runner.run_multiple(opt, func, dim=10, n_runs=10)
print(f"Mean: {agg.mean_fitness:.4f} ± {agg.std_fitness:.4f}")
```

### Sensitivity Analysis

```python
from analysis.sensitivity_analysis import SensitivityAnalyzer
from algorithms.particle_swarm.pso import ParticleSwarmOptimization
from benchmark_functions.multimodal import Ackley
import numpy as np

analyzer = SensitivityAnalyzer()
result = analyzer.sweep(
    ParticleSwarmOptimization,
    Ackley(),
    dim=10,
    varied_params={
        "w":  np.linspace(0.3, 0.9, 7).tolist(),
        "c1": np.linspace(0.5, 2.5, 5).tolist(),
    },
    n_runs=5,
)

# Extract 2D matrix for heatmap
matrix, x_vals, y_vals = result.to_matrix("w", "c1", metric="mean_fitness")
```

### Runtime Analysis

```python
from analysis.runtime_analysis import RuntimeAnalyzer
from benchmark_functions.unimodal import Sphere
from algorithms.grey_wolf.gwo import GreyWolfOptimizer

analyzer = RuntimeAnalyzer()
records = analyzer.dimension_scaling(
    [(GreyWolfOptimizer, GreyWolfOptimizer.get_default_params())],
    Sphere(),
    dimensions=[2, 5, 10, 20, 30],
    n_runs=5,
)
```

---

## Sensitivity Analysis Example

Configure in the UI (**Page 3 — Sensitivity Analysis**):
1. Select **PSO** + **Ackley** function + D=10
2. Choose **2D Sweep** mode
3. Set: `w ∈ [0.3, 0.9]` (7 steps), `c1 ∈ [0.5, 2.5]` (5 steps)
4. 3 runs per combo
5. View heatmap: mean fitness as function of w and c1

---

## Runtime Analysis Example

Configure in the UI (**Page 4 — Runtime Analysis**):
1. Select all 6 algorithms
2. Use **Dimension Scaling** mode
3. Dimensions: [2, 5, 10, 20, 30]
4. See which algorithms scale best

---

## Roadmap

- [ ] Add Differential Evolution, Simulated Annealing, Ant Colony Optimization
- [ ] Wilcoxon statistical significance testing in Comparative Report
- [ ] LaTeX-compatible table export
- [ ] CEC benchmark suite integration
- [ ] Constrained optimization support
- [ ] Algorithm plugin system for external contributions
- [ ] Automated experiment reports (Markdown / PDF)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New algorithms and benchmark functions are especially welcome — the registry system makes integration easy.

---

## License

MIT © 2024 — See [LICENSE](LICENSE) for details.
