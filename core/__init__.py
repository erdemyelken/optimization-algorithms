"""
core package

Note: BenchmarkRunner and Exporter are NOT imported here to avoid circular
imports with the algorithms package. Import them directly:
    from core.benchmark_runner import BenchmarkRunner
    from core.exporter import Exporter
"""
from core.result import OptimizationResult, AggregatedResult
from core.metrics import compute_rank_matrix, summary_table

__all__ = [
    "OptimizationResult",
    "AggregatedResult",
    "compute_rank_matrix",
    "summary_table",
]
