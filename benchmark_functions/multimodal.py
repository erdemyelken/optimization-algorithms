"""
Multimodal benchmark functions:
Rastrigin, Ackley, Griewank, Schwefel, Levy, Styblinski-Tang.
"""

import numpy as np

from benchmark_functions import BenchmarkFunction, FunctionMetadata, register


@register
class Rastrigin(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Rastrigin",
        formula="f(x) = 10n + Σ[xᵢ² - 10cos(2πxᵢ)]",
        global_optimum=0.0,
        global_optimum_location="x* = (0, …, 0)",
        bounds=(-5.12, 5.12),
        recommended_dims=[2, 5, 10, 20, 30],
        modality="multimodal",
        separability="separable",
        difficulty="hard",
        description=(
            "Highly multimodal with regularly distributed local minima. "
            "One of the hardest standard benchmarks. Separable but very "
            "deceptive due to the cosine landscape."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


@register
class Ackley(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Ackley",
        formula="f(x) = -20exp(-0.2√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e",
        global_optimum=0.0,
        global_optimum_location="x* = (0, …, 0)",
        bounds=(-32.768, 32.768),
        recommended_dims=[2, 5, 10, 20, 30],
        modality="multimodal",
        separability="non-separable",
        difficulty="hard",
        description=(
            "Exponential landscape with many local minima and a single "
            "narrow global minimum. The near-flat outer region can confuse "
            "algorithms, while inner oscillations create local traps."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        s1 = -0.2 * np.sqrt(np.sum(x ** 2) / n)
        s2 = np.sum(np.cos(2 * np.pi * x)) / n
        return float(-20 * np.exp(s1) - np.exp(s2) + 20 + np.e)


@register
class Griewank(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Griewank",
        formula="f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√i) + 1",
        global_optimum=0.0,
        global_optimum_location="x* = (0, …, 0)",
        bounds=(-600.0, 600.0),
        recommended_dims=[2, 5, 10, 20],
        modality="multimodal",
        separability="non-separable",
        difficulty="medium",
        description=(
            "Large-scale multimodal function with evenly distributed local "
            "minima. The product term creates interactions between dimensions "
            "despite having a structurally separable look."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        i = np.arange(1, n + 1, dtype=float)
        return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)


@register
class Schwefel(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Schwefel",
        formula="f(x) = 418.9829n - Σ[xᵢ sin(√|xᵢ|)]",
        global_optimum=0.0,
        global_optimum_location="x* = (420.9687, …, 420.9687)",
        bounds=(-500.0, 500.0),
        recommended_dims=[2, 5, 10],
        modality="multimodal",
        separability="separable",
        difficulty="hard",
        description=(
            "The global optimum is geometrically distant from the second-best "
            "local optima. Many algorithms converge to suboptimal solutions "
            "far from the global minimum."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


@register
class Levy(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Levy",
        formula="sin²(πw₁) + Σ(wᵢ-1)²[1+10sin²(πwᵢ₊₁)] + (wₙ-1)²[1+sin²(2πwₙ)]",
        global_optimum=0.0,
        global_optimum_location="x* = (1, …, 1)",
        bounds=(-10.0, 10.0),
        recommended_dims=[2, 5, 10, 20],
        modality="multimodal",
        separability="non-separable",
        difficulty="medium",
        description=(
            "Multimodal function with complex sine wave landscape. Tests "
            "fine-grained search near the global optimum. The w-transformation "
            "adds non-trivial interactions between dimensions."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        wi = w[:-1]
        wi1 = w[1:]
        sum_terms = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi1) ** 2))
        return float(term1 + sum_terms + term3)


@register
class StyblinskiTang(BenchmarkFunction):
    metadata = FunctionMetadata(
        name="Styblinski-Tang",
        formula="f(x) = Σ(xᵢ⁴ - 16xᵢ² + 5xᵢ) / 2",
        global_optimum=-39.16599 * 1,   # per dimension; updated in evaluate
        global_optimum_location="x* = (-2.903534, …, -2.903534)",
        bounds=(-5.0, 5.0),
        recommended_dims=[2, 5, 10],
        modality="multimodal",
        separability="separable",
        difficulty="medium",
        description=(
            "Separable but highly multimodal. Global optimum scales with "
            "dimension (f* ≈ -39.166n). Good for testing separability "
            "exploitation and local minima avoidance."
        ),
    )

    def _evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2)
