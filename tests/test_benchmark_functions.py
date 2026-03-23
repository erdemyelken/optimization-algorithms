"""
Tests for benchmark functions.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark_functions.unimodal  # noqa — populate registry
import benchmark_functions.multimodal  # noqa
from benchmark_functions import BENCHMARK_REGISTRY, get_function, list_functions


# ── Registry tests ────────────────────────────────────────────────────────────

def test_registry_not_empty():
    assert len(BENCHMARK_REGISTRY) >= 9


def test_all_functions_registered():
    expected = ["Sphere", "Rosenbrock", "Zakharov",
                "Rastrigin", "Ackley", "Griewank",
                "Schwefel", "Levy", "Styblinski-Tang"]
    for name in expected:
        assert name in BENCHMARK_REGISTRY, f"Missing: {name}"


def test_get_function_raises_on_unknown():
    with pytest.raises(KeyError):
        get_function("NonExistentFunction")


def test_list_functions_filter_modality():
    unimodal = list_functions(modality="unimodal")
    for f in unimodal:
        assert f.metadata.modality == "unimodal"

    multimodal = list_functions(modality="multimodal")
    for f in multimodal:
        assert f.metadata.modality == "multimodal"


# ── Evaluation tests ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("func_name,point,expected", [
    ("Sphere", np.zeros(5), 0.0),
    ("Sphere", np.array([1.0, 1.0, 1.0]), 3.0),
    ("Rosenbrock", np.ones(5), 0.0),
    ("Zakharov", np.zeros(5), 0.0),
    ("Rastrigin", np.zeros(5), 0.0),
    ("Ackley", np.zeros(5), 0.0),
    ("Griewank", np.zeros(5), 0.0),
])
def test_global_optimum(func_name, point, expected):
    func = get_function(func_name)
    val = func.evaluate(point)
    assert abs(val - expected) < 1e-6, f"{func_name}: expected {expected}, got {val}"


@pytest.mark.parametrize("func_name", list(BENCHMARK_REGISTRY.keys()) if BENCHMARK_REGISTRY else ["Sphere"])
def test_evaluate_returns_float(func_name):
    func = get_function(func_name)
    x = np.random.uniform(*func.bounds, size=5)
    result = func.evaluate(x)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_callable_interface():
    func = get_function("Sphere")
    x = np.array([1.0, 2.0])
    assert func(x) == func.evaluate(x)


def test_metadata_fields():
    for func in BENCHMARK_REGISTRY.values():
        m = func.metadata
        assert m.name
        assert m.formula
        assert isinstance(m.bounds, tuple) and len(m.bounds) == 2
        assert m.modality in ("unimodal", "multimodal")
        assert m.difficulty in ("easy", "medium", "hard")
        assert isinstance(m.recommended_dims, list)
