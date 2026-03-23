"""
Tests for BenchmarkRunner and AggregatedResult.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark_functions.unimodal  # noqa
import benchmark_functions.multimodal  # noqa
from benchmark_functions import get_function
from algorithms.particle_swarm.pso import ParticleSwarmOptimization
from algorithms.grey_wolf.gwo import GreyWolfOptimizer
from core.benchmark_runner import BenchmarkRunner
from core.result import AggregatedResult
from core.metrics import compute_rank_matrix, summary_table


DIM = 5


def test_run_single_returns_result():
    func = get_function("Sphere")
    opt = ParticleSwarmOptimization(pop_size=10, max_iter=20, w=0.7, c1=1.5, c2=1.5)
    runner = BenchmarkRunner()
    result = runner.run_single(opt, func, DIM)
    from core.result import OptimizationResult
    assert isinstance(result, OptimizationResult)


def test_run_multiple_aggregated():
    func = get_function("Sphere")
    opt = GreyWolfOptimizer(pop_size=10, max_iter=20)
    runner = BenchmarkRunner()
    agg = runner.run_multiple(opt, func, DIM, n_runs=3, base_seed=42)
    assert isinstance(agg, AggregatedResult)
    assert agg.n_runs == 3
    assert agg.mean_fitness <= agg.max_fitness
    assert agg.mean_fitness >= agg.min_fitness


def test_compare_algorithms():
    func = get_function("Sphere")
    opts = [
        ParticleSwarmOptimization(pop_size=10, max_iter=15, w=0.7, c1=1.5, c2=1.5),
        GreyWolfOptimizer(pop_size=10, max_iter=15),
    ]
    runner = BenchmarkRunner()
    results = runner.compare_algorithms(opts, func, DIM, n_runs=3)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, AggregatedResult)


def test_rank_matrix():
    func = get_function("Sphere")
    opts = [
        ParticleSwarmOptimization(pop_size=10, max_iter=20, w=0.7, c1=1.5, c2=1.5),
        GreyWolfOptimizer(pop_size=10, max_iter=20),
    ]
    runner = BenchmarkRunner()
    results = runner.compare_algorithms(opts, func, DIM, n_runs=3)
    ranks = compute_rank_matrix(results)
    assert all(algo in ranks for algo in [r.algorithm_name for r in results])


def test_summary_table():
    func = get_function("Rastrigin")
    opt = ParticleSwarmOptimization(pop_size=10, max_iter=15, w=0.7, c1=1.5, c2=1.5)
    runner = BenchmarkRunner()
    agg = runner.run_multiple(opt, func, DIM, n_runs=3)
    rows = summary_table([agg])
    assert len(rows) == 1
    row = rows[0]
    assert "Mean Fitness" in row
    assert "Mean Runtime (s)" in row
