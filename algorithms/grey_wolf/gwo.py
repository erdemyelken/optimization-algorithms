"""
Grey Wolf Optimizer — refactored to unified BaseOptimizer interface.
"""

from typing import Callable, List, Tuple

import numpy as np

from algorithms.base_optimizer import BaseOptimizer


class GreyWolfOptimizer(BaseOptimizer):
    """
    GWO based on Mirjalili et al. (2014).
    """

    name = "Grey Wolf Optimizer"

    param_schema = {
        "pop_size": {
            "type": "int",
            "default": 30,
            "min": 5,
            "max": 300,
            "step": 5,
            "help": "Number of search agents (wolves).",
        },
        "max_iter": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "Maximum number of iterations.",
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

        # Initialise positions
        pos = lb + np.random.rand(pop_size, dim) * (ub - lb)

        alpha_pos = np.zeros(dim)
        beta_pos = np.zeros(dim)
        delta_pos = np.zeros(dim)
        alpha_score = beta_score = delta_score = np.inf

        history: List[float] = []

        for iteration in range(max_iter):
            pos = np.clip(pos, lb, ub)
            fitness = np.array([func(pos[i]) for i in range(pop_size)])

            # Update alpha, beta, delta
            for i, fit in enumerate(fitness):
                if fit < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fit, pos[i].copy()
                elif fit < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fit, pos[i].copy()
                elif fit < delta_score:
                    delta_score, delta_pos = fit, pos[i].copy()

            # Linearly decreasing a
            a = 2 - iteration * (2 / max_iter)

            # Update positions
            r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * np.random.rand(pop_size, dim)
            D_alpha = np.abs(C1 * alpha_pos - pos)
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * np.random.rand(pop_size, dim)
            D_beta = np.abs(C2 * beta_pos - pos)
            X2 = beta_pos - A2 * D_beta

            r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * np.random.rand(pop_size, dim)
            D_delta = np.abs(C3 * delta_pos - pos)
            X3 = delta_pos - A3 * D_delta

            pos = (X1 + X2 + X3) / 3
            history.append(alpha_score)

        return alpha_pos, alpha_score, history
