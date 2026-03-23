"""
Standard result container for all optimization runs.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np
import json


@dataclass
class OptimizationResult:
    """Standardised output of a single optimization run."""

    algorithm_name: str
    best_solution: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    runtime_seconds: float
    eval_count: int
    params_used: Dict[str, Any]
    seed: Optional[int]
    run_id: int = 0

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["best_solution"] = self.best_solution.tolist()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def iterations(self) -> int:
        return len(self.convergence_history)


@dataclass
class AggregatedResult:
    """
    Statistics over multiple runs of the same algorithm × function combination.
    """

    algorithm_name: str
    function_name: str
    dimension: int
    n_runs: int

    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    median_fitness: float

    mean_runtime: float
    std_runtime: float

    mean_eval_count: float
    convergence_mean: List[float] = field(default_factory=list)
    convergence_std: List[float] = field(default_factory=list)

    params_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_runs(
        runs: List[OptimizationResult],
        function_name: str,
        dimension: int,
    ) -> "AggregatedResult":
        fitnesses = np.array([r.best_fitness for r in runs])
        runtimes = np.array([r.runtime_seconds for r in runs])
        evals = np.array([r.eval_count for r in runs])

        # Pad convergence histories to same length
        max_len = max(len(r.convergence_history) for r in runs)
        histories = np.array(
            [
                r.convergence_history
                + [r.convergence_history[-1]] * (max_len - len(r.convergence_history))
                for r in runs
            ]
        )

        return AggregatedResult(
            algorithm_name=runs[0].algorithm_name,
            function_name=function_name,
            dimension=dimension,
            n_runs=len(runs),
            mean_fitness=float(np.mean(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            min_fitness=float(np.min(fitnesses)),
            max_fitness=float(np.max(fitnesses)),
            median_fitness=float(np.median(fitnesses)),
            mean_runtime=float(np.mean(runtimes)),
            std_runtime=float(np.std(runtimes)),
            mean_eval_count=float(np.mean(evals)),
            convergence_mean=np.mean(histories, axis=0).tolist(),
            convergence_std=np.std(histories, axis=0).tolist(),
            params_used=runs[0].params_used,
        )
