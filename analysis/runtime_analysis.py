"""
Runtime Analysis module — dimension scaling, budget scaling, efficiency ranking.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import time

import numpy as np

from algorithms.base_optimizer import BaseOptimizer
from benchmark_functions import BenchmarkFunction


@dataclass
class RuntimeRecord:
    """Single measurement point."""

    algorithm_name: str
    function_name: str
    dimension: int
    max_iter: int
    mean_runtime: float
    std_runtime: float
    mean_fitness: float
    std_fitness: float
    n_runs: int
    efficiency: float  # fitness_improvement_per_second (smaller is better)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Algorithm": self.algorithm_name,
            "Function": self.function_name,
            "Dimension": self.dimension,
            "Max Iter": self.max_iter,
            "Mean Runtime (s)": self.mean_runtime,
            "Std Runtime (s)": self.std_runtime,
            "Mean Fitness": self.mean_fitness,
            "Std Fitness": self.std_fitness,
            "Efficiency": self.efficiency,
        }


class RuntimeAnalyzer:
    """
    Analyses runtime behaviour of optimizers under various conditions.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with ``(done, total, message)``.
    """

    def __init__(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> None:
        self.progress_callback = progress_callback

    def _run_once(
        self,
        cls: Type[BaseOptimizer],
        params: Dict[str, Any],
        func: BenchmarkFunction,
        dim: int,
        seed: int = 0,
    ) -> Tuple[float, float]:
        """Returns (runtime, best_fitness)."""
        bounds = [func.bounds] * dim
        opt = cls(**params)
        t0 = time.perf_counter()
        result = opt.optimize(func.evaluate, bounds, dim, seed=seed)
        return time.perf_counter() - t0, result.best_fitness

    def dimension_scaling(
        self,
        optimizers: List[Tuple[Type[BaseOptimizer], Dict[str, Any]]],
        func: BenchmarkFunction,
        dimensions: List[int],
        n_runs: int = 5,
        base_seed: int = 42,
    ) -> List[RuntimeRecord]:
        """
        Measure runtime and fitness across increasing dimensions.

        Parameters
        ----------
        optimizers : list of (optimizer_class, params_dict)
        func : BenchmarkFunction
        dimensions : list of int
        """
        records = []
        total = len(optimizers) * len(dimensions) * n_runs
        done = 0

        for cls, params in optimizers:
            for dim in dimensions:
                runtimes, fitnesses = [], []
                for run in range(n_runs):
                    seed = base_seed + run
                    rt, fit = self._run_once(cls, params, func, dim, seed)
                    runtimes.append(rt)
                    fitnesses.append(fit)
                    done += 1
                    if self.progress_callback:
                        self.progress_callback(done, total, f"{cls.name} | D={dim}")

                mean_rt = float(np.mean(runtimes))
                mean_fit = float(np.mean(fitnesses))
                eff = mean_fit / mean_rt if mean_rt > 0 else 0.0
                records.append(
                    RuntimeRecord(
                        algorithm_name=cls.name,
                        function_name=func.name,
                        dimension=dim,
                        max_iter=params.get("max_iter", 200),
                        mean_runtime=mean_rt,
                        std_runtime=float(np.std(runtimes)),
                        mean_fitness=mean_fit,
                        std_fitness=float(np.std(fitnesses)),
                        n_runs=n_runs,
                        efficiency=eff,
                    )
                )

        return records

    def budget_scaling(
        self,
        optimizers: List[Tuple[Type[BaseOptimizer], Dict[str, Any]]],
        func: BenchmarkFunction,
        dim: int,
        budgets: List[int],
        n_runs: int = 5,
        base_seed: int = 42,
    ) -> List[RuntimeRecord]:
        """
        Measure runtime and fitness as max_iter (budget) increases.
        """
        records = []
        total = len(optimizers) * len(budgets) * n_runs
        done = 0

        for cls, base_params in optimizers:
            for budget in budgets:
                params = dict(base_params)
                params["max_iter"] = budget
                runtimes, fitnesses = [], []
                for run in range(n_runs):
                    seed = base_seed + run
                    rt, fit = self._run_once(cls, params, func, dim, seed)
                    runtimes.append(rt)
                    fitnesses.append(fit)
                    done += 1
                    if self.progress_callback:
                        self.progress_callback(done, total, f"{cls.name} | budget={budget}")

                mean_rt = float(np.mean(runtimes))
                mean_fit = float(np.mean(fitnesses))
                eff = mean_fit / mean_rt if mean_rt > 0 else 0.0
                records.append(
                    RuntimeRecord(
                        algorithm_name=cls.name,
                        function_name=func.name,
                        dimension=dim,
                        max_iter=budget,
                        mean_runtime=mean_rt,
                        std_runtime=float(np.std(runtimes)),
                        mean_fitness=mean_fit,
                        std_fitness=float(np.std(fitnesses)),
                        n_runs=n_runs,
                        efficiency=eff,
                    )
                )

        return records

    def efficiency_ranking(self, records: List[RuntimeRecord]) -> List[Dict[str, Any]]:
        """
        Rank algorithms by trade-off between runtime and fitness.
        Lower efficiency score = better (smaller fitness × smaller runtime).
        """
        by_algo: Dict[str, List[RuntimeRecord]] = {}
        for r in records:
            by_algo.setdefault(r.algorithm_name, []).append(r)

        rows = []
        for algo, recs in by_algo.items():
            mean_fit = float(np.mean([r.mean_fitness for r in recs]))
            mean_rt = float(np.mean([r.mean_runtime for r in recs]))
            score = mean_fit * mean_rt if mean_rt > 0 else float("inf")
            rows.append(
                {
                    "Algorithm": algo,
                    "Avg Mean Fitness": mean_fit,
                    "Avg Mean Runtime (s)": mean_rt,
                    "Efficiency Score": score,
                }
            )
        return sorted(rows, key=lambda x: x["Efficiency Score"])
