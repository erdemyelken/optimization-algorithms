"""
Tests for all optimizer implementations.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark_functions.unimodal  # noqa
import benchmark_functions.multimodal  # noqa
from benchmark_functions import get_function
from algorithms import ALGORITHM_REGISTRY
from core.result import OptimizationResult


SPHERE = get_function("Sphere")
DIM = 5
BOUNDS = [SPHERE.bounds] * DIM


# ── All algorithms — basic smoke tests ───────────────────────────────────────

@pytest.mark.parametrize("algo_name", list(ALGORITHM_REGISTRY.keys()))
def test_optimize_returns_result(algo_name):
    """Each algorithm must run and return an OptimizationResult."""
    cls = ALGORITHM_REGISTRY[algo_name]
    params = {**cls.get_default_params(), "max_iter": 20, "pop_size": 10}
    opt = cls(**params)
    result = opt.optimize(SPHERE.evaluate, BOUNDS, DIM, seed=42)
    assert isinstance(result, OptimizationResult)


@pytest.mark.parametrize("algo_name", list(ALGORITHM_REGISTRY.keys()))
def test_convergence_history_length(algo_name):
    """Convergence history must have exactly max_iter entries."""
    max_iter = 25
    cls = ALGORITHM_REGISTRY[algo_name]
    params = {**cls.get_default_params(), "max_iter": max_iter, "pop_size": 10}
    opt = cls(**params)
    result = opt.optimize(SPHERE.evaluate, BOUNDS, DIM, seed=0)
    assert len(result.convergence_history) == max_iter


@pytest.mark.parametrize("algo_name", list(ALGORITHM_REGISTRY.keys()))
def test_seed_reproducibility(algo_name):
    """Same seed must produce identical results."""
    cls = ALGORITHM_REGISTRY[algo_name]
    params = {**cls.get_default_params(), "max_iter": 20, "pop_size": 10}

    opt1 = cls(**params)
    r1 = opt1.optimize(SPHERE.evaluate, BOUNDS, DIM, seed=17)

    opt2 = cls(**params)
    r2 = opt2.optimize(SPHERE.evaluate, BOUNDS, DIM, seed=17)

    assert r1.best_fitness == pytest.approx(r2.best_fitness, rel=1e-9)


@pytest.mark.parametrize("algo_name", list(ALGORITHM_REGISTRY.keys()))
def test_result_fields(algo_name):
    """Check required fields exist and are sensible."""
    cls = ALGORITHM_REGISTRY[algo_name]
    params = {**cls.get_default_params(), "max_iter": 15, "pop_size": 8}
    opt = cls(**params)
    r = opt.optimize(SPHERE.evaluate, BOUNDS, DIM, seed=1)

    assert r.algorithm_name == cls.name
    assert isinstance(r.best_solution, np.ndarray)
    assert r.best_solution.shape == (DIM,)
    assert np.isfinite(r.best_fitness)
    assert r.runtime_seconds > 0
    assert r.eval_count > 0
    assert r.seed == 1


@pytest.mark.parametrize("algo_name", list(ALGORITHM_REGISTRY.keys()))
def test_monotone_convergence(algo_name):
    """Best fitness must be non-increasing over iterations."""
    cls = ALGORITHM_REGISTRY[algo_name]
    params = {**cls.get_default_params(), "max_iter": 50, "pop_size": 15}
    opt = cls(**params)
    r = opt.optimize(SPHERE.evaluate, BOUNDS, DIM, seed=5)

    for i in range(1, len(r.convergence_history)):
        assert r.convergence_history[i] <= r.convergence_history[i - 1] + 1e-12, \
            f"{algo_name}: convergence not monotone at iteration {i}"


def test_param_schema_populated():
    """Each algorithm must define a non-empty param_schema."""
    for name, cls in ALGORITHM_REGISTRY.items():
        schema = cls.get_param_schema()
        assert isinstance(schema, dict) and len(schema) > 0, f"{name} has empty param_schema"


def test_default_params_match_schema():
    """Default params must exist for every schema key."""
    for name, cls in ALGORITHM_REGISTRY.items():
        defaults = cls.get_default_params()
        for key in cls.get_param_schema():
            assert key in defaults, f"{name}: missing default for '{key}'"
