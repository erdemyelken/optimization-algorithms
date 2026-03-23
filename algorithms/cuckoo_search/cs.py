"""
Cuckoo Search — refactored to unified BaseOptimizer interface.
Based on Yang & Deb (2009).
"""

from typing import Callable, List, Tuple

import numpy as np
from scipy.special import gamma as scipy_gamma

from algorithms.base_optimizer import BaseOptimizer


def _levy_flight(beta: float, size: tuple) -> np.ndarray:
    """Generate Lévy-distributed steps."""
    sigma_num = scipy_gamma(1 + beta) * np.sin(np.pi * beta / 2)
    sigma_den = scipy_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (sigma_num / sigma_den) ** (1 / beta)
    u = np.random.randn(*size) * sigma
    v = np.random.randn(*size)
    step = u / (np.abs(v) ** (1 / beta))
    return step


class CuckooSearch(BaseOptimizer):
    """Cuckoo Search with Lévy flights."""

    name = "Cuckoo Search"

    param_schema = {
        "pop_size": {
            "type": "int",
            "default": 25,
            "min": 5,
            "max": 300,
            "step": 5,
            "help": "Number of nests.",
        },
        "max_iter": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "Maximum number of iterations.",
        },
        "pa": {
            "type": "float",
            "default": 0.25,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "Fraction of worst nests abandoned per iteration.",
        },
        "beta": {
            "type": "float",
            "default": 1.5,
            "min": 1.0,
            "max": 2.0,
            "step": 0.1,
            "help": "Lévy exponent (1.0 = Brownian, 2.0 = Gaussian).",
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
        pa = self.params["pa"]
        beta = self.params["beta"]

        nests = lb + np.random.rand(n, dim) * (ub - lb)
        fitness = np.array([func(nest) for nest in nests])
        best_idx = int(np.argmin(fitness))
        best_nest = nests[best_idx].copy()
        best_fit = fitness[best_idx]
        history: List[float] = []

        for _ in range(max_iter):
            # Lévy flight step
            step_size = 0.01 * _levy_flight(beta, (n, dim))
            new_nests = nests + step_size * (nests - best_nest)
            new_nests = np.clip(new_nests, lb, ub)

            for i in range(n):
                f_new = func(new_nests[i])
                if f_new < fitness[i]:
                    nests[i] = new_nests[i]
                    fitness[i] = f_new

            # Abandon worst nests
            n_abandon = max(1, int(pa * n))
            worst_idxs = np.argsort(fitness)[-n_abandon:]
            nests[worst_idxs] = lb + np.random.rand(n_abandon, dim) * (ub - lb)
            fitness[worst_idxs] = np.array([func(nests[j]) for j in worst_idxs])

            # Track best
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_nest = nests[gen_best_idx].copy()

            history.append(best_fit)

        return best_nest, best_fit, history
