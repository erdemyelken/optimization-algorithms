"""
Benchmark runner — orchestrates single and multi-run experiments.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np

from core.result import OptimizationResult, AggregatedResult
from algorithms.base_optimizer import BaseOptimizer
from benchmark_functions import BenchmarkFunction


class BenchmarkRunner:
    """
    Runs optimization experiments and returns structured results.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with ``(current_run, total_runs, message)`` during execution.
        Useful for Streamlit progress bars.
    """

    def __init__(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> None:
        self.progress_callback = progress_callback

    def run_single(
        self,
        optimizer: BaseOptimizer,
        func: BenchmarkFunction,
        dim: int,
        seed: Optional[int] = 42,
        run_id: int = 0,
    ) -> OptimizationResult:
        """Run one experiment and return the raw result."""
        bounds = [func.bounds] * dim if not isinstance(func.bounds[0], (list, tuple)) else func.bounds[:dim]
        return optimizer.optimize(func.evaluate, bounds, dim, seed=seed, run_id=run_id)

    def run_multiple(
        self,
        optimizer: BaseOptimizer,
        func: BenchmarkFunction,
        dim: int,
        n_runs: int = 10,
        base_seed: Optional[int] = 42,
    ) -> AggregatedResult:
        """
        Run the experiment ``n_runs`` times with different seeds and aggregate.

        Returns
        -------
        AggregatedResult
        """
        raw_results: List[OptimizationResult] = []

        for i in range(n_runs):
            seed = (base_seed + i) if base_seed is not None else None
            if self.progress_callback:
                self.progress_callback(i + 1, n_runs, f"Run {i+1}/{n_runs} — {optimizer.name} on {func.name}")

            result = self.run_single(optimizer, func, dim, seed=seed, run_id=i)
            raw_results.append(result)

        return AggregatedResult.from_runs(raw_results, func.name, dim)

    def compare_algorithms(
        self,
        optimizers: List[BaseOptimizer],
        func: BenchmarkFunction,
        dim: int,
        n_runs: int = 10,
        base_seed: int = 42,
    ) -> List[AggregatedResult]:
        """
        Run all ``optimizers`` on the same ``func`` and return aggregated results.
        """
        results: List[AggregatedResult] = []
        for opt in optimizers:
            agg = self.run_multiple(opt, func, dim, n_runs=n_runs, base_seed=base_seed)
            results.append(agg)
        return results

    def run_experiment_config(self, config: Dict[str, Any]) -> List[AggregatedResult]:
        """
        High-level entry point accepting a config dict (from YAML or UI).

        Expected keys:
          function_name, algorithm_names, dim, n_runs, base_seed,
          algorithm_params (optional overrides per algorithm).
        """
        from benchmark_functions import BENCHMARK_REGISTRY
        from algorithms import ALGORITHM_REGISTRY

        func = BENCHMARK_REGISTRY[config["function_name"]]
        dim = config.get("dim", 10)
        n_runs = config.get("n_runs", 10)
        base_seed = config.get("base_seed", 42)
        algo_params = config.get("algorithm_params", {})

        results: List[AggregatedResult] = []
        algo_names = config["algorithm_names"]
        total = len(algo_names)

        for idx, name in enumerate(algo_names):
            cls = ALGORITHM_REGISTRY[name]
            params = algo_params.get(name, cls.get_default_params())
            optimizer = cls(**params)
            if self.progress_callback:
                self.progress_callback(idx, total, f"Running {name}…")
            agg = self.run_multiple(optimizer, func, dim, n_runs=n_runs, base_seed=base_seed)
            results.append(agg)

        return results
