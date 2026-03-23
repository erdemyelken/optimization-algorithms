"""
Benchmark function base class + registry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np


@dataclass
class FunctionMetadata:
    """Rich metadata for every benchmark function."""

    name: str
    formula: str
    global_optimum: float
    global_optimum_location: str          # e.g. "x* = (0, …, 0)"
    bounds: Tuple[float, float]           # symmetric per-dimension bounds
    recommended_dims: List[int]
    modality: str                         # "unimodal" | "multimodal"
    separability: str                     # "separable" | "non-separable"
    difficulty: str                       # "easy" | "medium" | "hard"
    description: str


class BenchmarkFunction(ABC):
    """Abstract base for all benchmark functions."""

    metadata: FunctionMetadata

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def bounds(self) -> Tuple[float, float]:
        return self.metadata.bounds

    @property
    def global_optimum(self) -> float:
        return self.metadata.global_optimum

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the function at point ``x``."""
        return float(self._evaluate(x))

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> float: ...

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def plot_landscape(self, ax, resolution: int = 100) -> None:
        """
        Draw a 2-D colour-map of the function (first two dimensions only).
        ``ax`` should be a matplotlib Axes object.
        """
        lb, ub = self.bounds
        xx = np.linspace(lb, ub, resolution)
        yy = np.linspace(lb, ub, resolution)
        X, Y = np.meshgrid(xx, yy)
        Z = np.array(
            [[self.evaluate(np.array([X[i, j], Y[i, j]])) for j in range(resolution)]
             for i in range(resolution)]
        )
        cp = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title(f"{self.name} (2D slice)")
        return cp


# ---------------------------------------------------------------------- #
# Global registry
# ---------------------------------------------------------------------- #

BENCHMARK_REGISTRY: Dict[str, BenchmarkFunction] = {}


def register(cls: Type[BenchmarkFunction]) -> Type[BenchmarkFunction]:
    """Class decorator that adds an instance to the registry."""
    instance = cls()
    BENCHMARK_REGISTRY[instance.name] = instance
    return cls


def get_function(name: str) -> BenchmarkFunction:
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark function: '{name}'. "
                       f"Available: {list(BENCHMARK_REGISTRY)}")
    return BENCHMARK_REGISTRY[name]


def list_functions(
    modality: Optional[str] = None,
    difficulty: Optional[str] = None,
    max_dim: Optional[int] = None,
) -> List[BenchmarkFunction]:
    """Return filtered list of benchmark functions."""
    funcs = list(BENCHMARK_REGISTRY.values())
    if modality:
        funcs = [f for f in funcs if f.metadata.modality == modality]
    if difficulty:
        funcs = [f for f in funcs if f.metadata.difficulty == difficulty]
    if max_dim:
        funcs = [f for f in funcs if any(d <= max_dim for d in f.metadata.recommended_dims)]
    return funcs
