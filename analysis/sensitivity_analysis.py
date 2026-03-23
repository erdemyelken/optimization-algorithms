"""
Sensitivity Analysis — 1D and 2D parameter sweeps.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import itertools

import numpy as np

from algorithms.base_optimizer import BaseOptimizer
from benchmark_functions import BenchmarkFunction
from core.result import AggregatedResult, OptimizationResult


@dataclass
class SweepResult:
    """Result of a single parameter combination in a sweep."""

    param_values: Dict[str, Any]
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    mean_runtime: float
    std_runtime: float
    all_fitnesses: List[float] = field(default_factory=list)
    all_runtimes: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.param_values)
        d.update(
            {
                "best_fitness": self.best_fitness,
                "mean_fitness": self.mean_fitness,
                "std_fitness": self.std_fitness,
                "min_fitness": self.min_fitness,
                "max_fitness": self.max_fitness,
                "mean_runtime": self.mean_runtime,
                "std_runtime": self.std_runtime,
            }
        )
        return d


@dataclass
class SensitivityResult:
    """Collection of sweep results with metadata."""

    algorithm_name: str
    function_name: str
    dimension: int
    n_runs: int
    fixed_params: Dict[str, Any]
    varied_params: List[str]
    results: List[SweepResult] = field(default_factory=list)

    def to_matrix(
        self, param1: str, param2: str, metric: str = "mean_fitness"
    ) -> Tuple[np.ndarray, List[Any], List[Any]]:
        """
        Build a 2D matrix for heatmap visualisation.

        Returns (matrix, p1_values, p2_values).
        """
        p1_vals = sorted(set(r.param_values[param1] for r in self.results))
        p2_vals = sorted(set(r.param_values[param2] for r in self.results))
        mat = np.full((len(p2_vals), len(p1_vals)), np.nan)
        for r in self.results:
            i = p2_vals.index(r.param_values[param2])
            j = p1_vals.index(r.param_values[param1])
            mat[i, j] = getattr(r, metric)
        return mat, p1_vals, p2_vals

    def to_records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]


class SensitivityAnalyzer:
    """
    Runs parameter sweeps and returns structured results.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with ``(done, total, message)`` during runs.
    """

    def __init__(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> None:
        self.progress_callback = progress_callback

    def sweep(
        self,
        optimizer_cls: type,
        func: BenchmarkFunction,
        dim: int,
        varied_params: Dict[str, List[Any]],
        fixed_params: Optional[Dict[str, Any]] = None,
        n_runs: int = 5,
        base_seed: int = 42,
    ) -> SensitivityResult:
        """
        Run a grid sweep over ``varied_params``.

        Parameters
        ----------
        optimizer_cls : Type[BaseOptimizer]
            The optimizer class to sweep.
        func : BenchmarkFunction
            The benchmark function.
        dim : int
            Dimensionality.
        varied_params : dict
            {param_name: [value1, value2, ...]}
        fixed_params : dict, optional
            Parameters held constant; defaults filled from schema.
        n_runs : int
            Runs per parameter combination.
        base_seed : int
            Base seed; actual seed = base_seed + run_index.
        """
        base_params = optimizer_cls.get_default_params()
        if fixed_params:
            base_params.update(fixed_params)

        # Build all combinations
        param_names = list(varied_params.keys())
        param_grids = list(varied_params.values())
        combos = list(itertools.product(*param_grids))
        total = len(combos) * n_runs
        done = 0

        bounds = [func.bounds] * dim
        sweep_results = []

        for combo in combos:
            params = dict(base_params)
            combo_dict = dict(zip(param_names, combo))
            params.update(combo_dict)

            raw: List[OptimizationResult] = []
            for run_idx in range(n_runs):
                seed = base_seed + run_idx
                opt = optimizer_cls(**params)
                result = opt.optimize(func.evaluate, bounds, dim, seed=seed, run_id=run_idx)
                raw.append(result)
                done += 1
                if self.progress_callback:
                    self.progress_callback(done, total, f"{optimizer_cls.name} | {combo_dict}")

            fitnesses = [r.best_fitness for r in raw]
            runtimes = [r.runtime_seconds for r in raw]
            sweep_results.append(
                SweepResult(
                    param_values=combo_dict,
                    best_fitness=float(np.min(fitnesses)),
                    mean_fitness=float(np.mean(fitnesses)),
                    std_fitness=float(np.std(fitnesses)),
                    min_fitness=float(np.min(fitnesses)),
                    max_fitness=float(np.max(fitnesses)),
                    mean_runtime=float(np.mean(runtimes)),
                    std_runtime=float(np.std(runtimes)),
                    all_fitnesses=fitnesses,
                    all_runtimes=runtimes,
                )
            )

        return SensitivityResult(
            algorithm_name=optimizer_cls.name,
            function_name=func.name,
            dimension=dim,
            n_runs=n_runs,
            fixed_params={k: v for k, v in base_params.items() if k not in param_names},
            varied_params=param_names,
            results=sweep_results,
        )
