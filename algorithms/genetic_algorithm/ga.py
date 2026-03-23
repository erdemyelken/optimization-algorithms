"""
Genetic Algorithm — refactored to unified BaseOptimizer interface.

This implementation uses real-valued (floating-point) encoding to work
natively with continuous benchmark functions.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from algorithms.base_optimizer import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    """
    Real-valued Genetic Algorithm with tournament selection,
    SBX crossover, and polynomial mutation.
    """

    name = "Genetic Algorithm"

    param_schema = {
        "pop_size": {
            "type": "int",
            "default": 50,
            "min": 10,
            "max": 500,
            "step": 10,
            "help": "Number of individuals in the population.",
        },
        "max_iter": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "Maximum number of generations.",
        },
        "mutation_rate": {
            "type": "float",
            "default": 0.1,
            "min": 0.001,
            "max": 1.0,
            "step": 0.01,
            "help": "Probability of mutating a gene.",
        },
        "crossover_rate": {
            "type": "float",
            "default": 0.9,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "Probability of applying crossover.",
        },
        "tournament_size": {
            "type": "int",
            "default": 3,
            "min": 2,
            "max": 10,
            "step": 1,
            "help": "Number of candidates in tournament selection.",
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
        mutation_rate = self.params["mutation_rate"]
        crossover_rate = self.params["crossover_rate"]
        t_size = self.params["tournament_size"]

        # Initialise population
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.array([func(ind) for ind in pop])

        best_idx = int(np.argmin(fitness))
        best_sol = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        history: List[float] = []

        for _ in range(max_iter):
            new_pop = np.empty_like(pop)

            for i in range(pop_size):
                # Tournament selection — parent 1
                idxs = np.random.randint(0, pop_size, t_size)
                p1 = pop[idxs[np.argmin(fitness[idxs])]]

                # Tournament selection — parent 2
                idxs = np.random.randint(0, pop_size, t_size)
                p2 = pop[idxs[np.argmin(fitness[idxs])]]

                # Crossover (uniform)
                if np.random.rand() < crossover_rate:
                    mask = np.random.rand(dim) < 0.5
                    child = np.where(mask, p1, p2)
                else:
                    child = p1.copy()

                # Gaussian mutation
                for g in range(dim):
                    if np.random.rand() < mutation_rate:
                        sigma = (ub[g] - lb[g]) * 0.1
                        child[g] += np.random.randn() * sigma

                # Clip to bounds
                child = np.clip(child, lb, ub)
                new_pop[i] = child

            pop = new_pop
            fitness = np.array([func(ind) for ind in pop])

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_sol = pop[gen_best_idx].copy()

            history.append(best_fit)

        return best_sol, best_fit, history
