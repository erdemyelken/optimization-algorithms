"""
Firefly Algorithm — refactored to unified BaseOptimizer interface.
Based on Yang (2009).
"""

from typing import Callable, List, Tuple

import numpy as np

from algorithms.base_optimizer import BaseOptimizer


class FireflyAlgorithm(BaseOptimizer):
    """Firefly Algorithm (FA)."""

    name = "Firefly Algorithm"

    param_schema = {
        "pop_size": {
            "type": "int",
            "default": 25,
            "min": 5,
            "max": 300,
            "step": 5,
            "help": "Number of fireflies.",
        },
        "max_iter": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "Maximum number of iterations.",
        },
        "alpha": {
            "type": "float",
            "default": 0.2,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "Randomness (step size) parameter α.",
        },
        "beta0": {
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 2.0,
            "step": 0.1,
            "help": "Attractiveness at zero distance β₀.",
        },
        "gamma": {
            "type": "float",
            "default": 1.0,
            "min": 0.01,
            "max": 10.0,
            "step": 0.1,
            "help": "Light absorption coefficient γ.",
        },
    }

    def _run(
        self,
        func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
    ) -> Tuple[np.ndarray, float, List[float]]:
        n = self.params["pop_size"]
        max_iter = self.params["max_iter"]
        alpha = self.params["alpha"]
        beta0 = self.params["beta0"]
        gamma = self.params["gamma"]

        pos = lb + np.random.rand(n, dim) * (ub - lb)
        fitness = np.array([func(p) for p in pos])
        history: List[float] = []
        best_fit = float(np.min(fitness))
        best_pos = pos[int(np.argmin(fitness))].copy()

        for _ in range(max_iter):
            for i in range(n):
                for j in range(n):
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(pos[i] - pos[j])
                        beta = beta0 * np.exp(-gamma * r ** 2)
                        pos[i] = (
                            pos[i]
                            + beta * (pos[j] - pos[i])
                            + alpha * (ub - lb) * (np.random.rand(dim) - 0.5)
                        )
                        pos[i] = np.clip(pos[i], lb, ub)
                        fitness[i] = func(pos[i])

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_pos = pos[gen_best_idx].copy()

            history.append(best_fit)

        return best_pos, best_fit, history
