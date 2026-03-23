"""
Particle Swarm Optimization — refactored to unified BaseOptimizer interface.
"""

from typing import Callable, List, Tuple

import numpy as np

from algorithms.base_optimizer import BaseOptimizer


class ParticleSwarmOptimization(BaseOptimizer):
    """
    Standard PSO with inertia weight decay.
    """

    name = "Particle Swarm Optimization"

    param_schema = {
        "pop_size": {
            "type": "int",
            "default": 30,
            "min": 5,
            "max": 300,
            "step": 5,
            "help": "Number of particles.",
        },
        "max_iter": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "Maximum number of iterations.",
        },
        "w": {
            "type": "float",
            "default": 0.7,
            "min": 0.1,
            "max": 1.5,
            "step": 0.05,
            "help": "Inertia weight.",
        },
        "c1": {
            "type": "float",
            "default": 1.5,
            "min": 0.1,
            "max": 4.0,
            "step": 0.1,
            "help": "Cognitive coefficient (personal best attraction).",
        },
        "c2": {
            "type": "float",
            "default": 1.5,
            "min": 0.1,
            "max": 4.0,
            "step": 0.1,
            "help": "Social coefficient (global best attraction).",
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
        w = self.params["w"]
        c1 = self.params["c1"]
        c2 = self.params["c2"]

        # Initialise positions and velocities
        pos = lb + np.random.rand(pop_size, dim) * (ub - lb)
        vel = np.zeros_like(pos)
        pbest = pos.copy()
        pbest_fit = np.array([func(p) for p in pbest])

        gbest_idx = int(np.argmin(pbest_fit))
        gbest = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        history: List[float] = []
        v_max = 0.2 * (ub - lb)

        for _ in range(max_iter):
            r1 = np.random.rand(pop_size, dim)
            r2 = np.random.rand(pop_size, dim)

            vel = (
                w * vel
                + c1 * r1 * (pbest - pos)
                + c2 * r2 * (gbest - pos)
            )
            vel = np.clip(vel, -v_max, v_max)
            pos = np.clip(pos + vel, lb, ub)

            fit = np.array([func(p) for p in pos])

            # Update personal bests
            improved = fit < pbest_fit
            pbest[improved] = pos[improved].copy()
            pbest_fit[improved] = fit[improved]

            # Update global best
            gen_best_idx = int(np.argmin(pbest_fit))
            if pbest_fit[gen_best_idx] < gbest_fit:
                gbest_fit = pbest_fit[gen_best_idx]
                gbest = pbest[gen_best_idx].copy()

            history.append(gbest_fit)

        return gbest, gbest_fit, history
