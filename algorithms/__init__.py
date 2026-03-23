"""
Algorithm package — registry of all available optimizers.

Add new algorithms by importing them here and registering in ALGORITHM_REGISTRY.
"""

from typing import Dict, Type
from algorithms.base_optimizer import BaseOptimizer

# Import all concrete optimizers
from algorithms.genetic_algorithm.ga import GeneticAlgorithm
from algorithms.particle_swarm.pso import ParticleSwarmOptimization
from algorithms.grey_wolf.gwo import GreyWolfOptimizer
from algorithms.whale_optimization.woa import WhaleOptimizationAlgorithm
from algorithms.cuckoo_search.cs import CuckooSearch
from algorithms.firefly.fa import FireflyAlgorithm

#: Central registry mapping display name → optimizer class.
ALGORITHM_REGISTRY: Dict[str, Type[BaseOptimizer]] = {
    GeneticAlgorithm.name: GeneticAlgorithm,
    ParticleSwarmOptimization.name: ParticleSwarmOptimization,
    GreyWolfOptimizer.name: GreyWolfOptimizer,
    WhaleOptimizationAlgorithm.name: WhaleOptimizationAlgorithm,
    CuckooSearch.name: CuckooSearch,
    FireflyAlgorithm.name: FireflyAlgorithm,
}

__all__ = [
    "BaseOptimizer",
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
    "GreyWolfOptimizer",
    "WhaleOptimizationAlgorithm",
    "CuckooSearch",
    "FireflyAlgorithm",
    "ALGORITHM_REGISTRY",
]
