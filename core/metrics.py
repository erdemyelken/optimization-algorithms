"""
Statistical utilities for experiment analysis.
"""

from typing import Dict, List
import numpy as np
from core.result import AggregatedResult


def compute_rank_matrix(
    agg_results: List[AggregatedResult],
) -> Dict[str, float]:
    """
    Compute average ranks across functions for each algorithm (Friedman-style).

    Parameters
    ----------
    agg_results : list of AggregatedResult
        Flat list of aggregated results; multiple function/algorithm combinations.

    Returns
    -------
    dict mapping algorithm_name → average rank (lower is better).
    """
    # Group by function_name
    from collections import defaultdict
    by_func: Dict[str, List[AggregatedResult]] = defaultdict(list)
    for r in agg_results:
        by_func[r.function_name].append(r)

    algo_ranks: Dict[str, List[float]] = defaultdict(list)

    for func_name, results in by_func.items():
        # Sort by mean fitness ascending (minimisation)
        sorted_r = sorted(results, key=lambda r: r.mean_fitness)
        for rank, r in enumerate(sorted_r, start=1):
            algo_ranks[r.algorithm_name].append(float(rank))

    return {algo: float(np.mean(ranks)) for algo, ranks in algo_ranks.items()}


def efficiency_score(mean_fitness: float, mean_runtime: float) -> float:
    """
    Simple efficiency: inverse of (normalised_fitness × runtime).
    Higher is better.  Returns 0 if runtime or fitness is 0/invalid.
    """
    if mean_runtime <= 0 or mean_fitness <= 0:
        return 0.0
    return 1.0 / (mean_fitness * mean_runtime)


def summary_table(agg_results: List[AggregatedResult]) -> List[Dict]:
    """Return a list of dicts suitable for a pandas DataFrame."""
    rows = []
    for r in agg_results:
        rows.append(
            {
                "Algorithm": r.algorithm_name,
                "Function": r.function_name,
                "Dim": r.dimension,
                "Runs": r.n_runs,
                "Mean Fitness": r.mean_fitness,
                "Std Fitness": r.std_fitness,
                "Min Fitness": r.min_fitness,
                "Max Fitness": r.max_fitness,
                "Median": r.median_fitness,
                "Mean Runtime (s)": r.mean_runtime,
                "Std Runtime (s)": r.std_runtime,
                "Mean Evals": r.mean_eval_count,
            }
        )
    return rows
