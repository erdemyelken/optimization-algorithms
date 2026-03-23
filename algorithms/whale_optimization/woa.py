"""
Whale Optimization Algorithm — refactored to unified BaseOptimizer interface.
Based on Mirjalili & Lewis (2016).
"""

from typing import Callable, List, Tuple

import numpy as np

from algorithms.base_optimizer import BaseOptimizer


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """Whale Optimization Algorithm (WOA)."""

    name = "Whale Optimization Algorithm"

    param_schema = {
        "pop_size": {
            "type": "int",
            "default": 30,
            "min": 5,
            "max": 300,
            "step": 5,
            "help": "Number of whales.",
        },
        "max_iter": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "Maximum number of iterations.",
        },
        "b": {
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 5.0,
            "step": 0.1,
            "help": "Spiral constant b.",
        },
    }

    def _run(
        self,
        func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
    ) -> Tuple[np.ndarray, float, List[float]]:
        pop_size = self.params["pop_size"]
        max_iter = self.params["max_iter"]
        b = self.params["b"]

        pos = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.array([func(p) for p in pos])
        best_idx = int(np.argmin(fitness))
        best_pos = pos[best_idx].copy()
        best_fit = fitness[best_idx]
        history: List[float] = []

        for t in range(max_iter):
            a = 2 - t * (2 / max_iter)   # linearly decreases from 2 to 0
            a2 = -1 + t * (-1 / max_iter)  # for random walk

            for i in range(pop_size):
                r = np.random.rand()
                A = 2 * a * np.random.rand(dim) - a
                C = 2 * np.random.rand(dim)
                p = np.random.rand()
                l = (a2 - 1) * np.random.rand() + 1   # noqa: E741

                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        # Shrinking encircling
                        D = np.abs(C * best_pos - pos[i])
                        pos[i] = best_pos - A * D
                    else:
                        # Random search (exploration)
                        rand_idx = np.random.randint(0, pop_size)
                        X_rand = pos[rand_idx]
                        D = np.abs(C * X_rand - pos[i])
                        pos[i] = X_rand - A * D
                else:
                    # Spiral (bubble-net attack)
                    D_prime = np.abs(best_pos - pos[i])
                    pos[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

                pos[i] = np.clip(pos[i], lb, ub)
                fit = func(pos[i])
                if fit < fitness[i]:
                    fitness[i] = fit
                if fit < best_fit:
                    best_fit = fit
                    best_pos = pos[i].copy()

            history.append(best_fit)

        return best_pos, best_fit, history
