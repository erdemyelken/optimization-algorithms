"""
Base optimizer interface for all metaheuristic algorithms.
All concrete optimizers must inherit from BaseOptimizer.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time

from core.result import OptimizationResult


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.

    Every concrete optimizer must implement the ``_run`` method which performs
    the actual optimization and returns ``(best_solution, best_fitness,
    convergence_history, eval_count)``.
    """

    #: Short display name shown in the UI and reports.
    name: str = "BaseOptimizer"

    #: Parameter schema used by the UI to auto-generate controls.
    #: Each key maps to a dict with keys: type, default, min, max, step, help.
    param_schema: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        self.params = self._validate_params(params)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        dim: int,
        seed: Optional[int] = None,
        run_id: int = 0,
    ) -> OptimizationResult:
        """
        Run the optimization and return a standardised result object.

        Parameters
        ----------
        func:
            Objective function to *minimise*.  Signature ``f(x: np.ndarray) -> float``.
        bounds:
            List of ``(low, high)`` pairs, one per dimension.
        dim:
            Number of dimensions.
        seed:
            Random seed for reproducibility.
        run_id:
            Index of this run in a multi-run experiment.

        Returns
        -------
        OptimizationResult
        """
        if seed is not None:
            np.random.seed(seed)

        eval_counter = [0]

        def counted_func(x: np.ndarray) -> float:
            eval_counter[0] += 1
            return func(x)

        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        t0 = time.perf_counter()
        best_solution, best_fitness, convergence_history = self._run(
            counted_func, lb, ub, dim
        )
        runtime = time.perf_counter() - t0

        return OptimizationResult(
            algorithm_name=self.name,
            best_solution=np.asarray(best_solution, dtype=float),
            best_fitness=float(best_fitness),
            convergence_history=list(convergence_history),
            runtime_seconds=runtime,
            eval_count=eval_counter[0],
            params_used=dict(self.params),
            seed=seed,
            run_id=run_id,
        )

    @abstractmethod
    def _run(
        self,
        func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Core optimization loop.

        Returns
        -------
        best_solution : np.ndarray
        best_fitness  : float
        convergence_history : list[float]  — best fitness at each iteration
        """
        ...

    @classmethod
    def get_param_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return the parameter schema for this optimizer."""
        return cls.param_schema

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Return a dict of {param_name: default_value}."""
        return {k: v["default"] for k, v in cls.param_schema.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing params with defaults and cast to declared types."""
        merged: Dict[str, Any] = {}
        for key, spec in self.param_schema.items():
            raw = params.get(key, spec["default"])
            dtype = spec.get("type", "float")
            if dtype == "int":
                merged[key] = int(raw)
            elif dtype == "float":
                merged[key] = float(raw)
            else:
                merged[key] = raw
        return merged

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}({self.params})"
