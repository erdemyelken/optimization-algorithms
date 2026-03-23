"""
Unimodal benchmark functions: Sphere, Rosenbrock, Zakharov.
"""

import numpy as np

from benchmark_functions import BenchmarkFunction, FunctionMetadata, register


@register
class Sphere(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Sphere",
        formula="f(x) = Σ xᵢ²",
        global_optimum=0.0,
        global_optimum_location="x* = (0, …, 0)",
        bounds=(-5.12, 5.12),
        recommended_dims=[2, 5, 10, 20, 30],
        modality="unimodal",
        separability="separable",
        difficulty="easy",
        description=(
            "The classic baseline function. Perfectly smooth, convex, and "
            "separable. Global minimum is at the origin. All algorithms should "
            "solve this easily."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))


@register
class Rosenbrock(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Rosenbrock",
        formula="f(x) = Σ [100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]",
        global_optimum=0.0,
        global_optimum_location="x* = (1, …, 1)",
        bounds=(-5.0, 10.0),
        recommended_dims=[2, 5, 10, 20],
        modality="unimodal",
        separability="non-separable",
        difficulty="medium",
        description=(
            "Banana-shaped narrow valley. The global minimum is inside a "
            "long, narrow, parabolic flat valley making convergence slow. "
            "Tests ability to follow curved ridges."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        xi = x[:-1]
        xi1 = x[1:]
        return float(np.sum(100 * (xi1 - xi ** 2) ** 2 + (1 - xi) ** 2))


@register
class Zakharov(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Zakharov",
        formula="f(x) = Σxᵢ² + (0.5Σixᵢ)² + (0.5Σixᵢ)⁴",
        global_optimum=0.0,
        global_optimum_location="x* = (0, …, 0)",
        bounds=(-5.0, 10.0),
        recommended_dims=[2, 5, 10],
        modality="unimodal",
        separability="non-separable",
        difficulty="medium",
        description=(
            "Unimodal and non-separable, but with higher-order terms that "
            "create a warped basin. Tests accuracy near the global minimum."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        i = np.arange(1, n + 1, dtype=float)
        s1 = np.sum(x ** 2)
        s2 = np.sum(0.5 * i * x)
        return float(s1 + s2 ** 2 + s2 ** 4)
