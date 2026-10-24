"""Random-search baseline for a fair comparison with the Genetic Algorithm.

Draws independent random chromosomes with the SAME generator, chromosome
representation, FitnessEvaluator and constraint rules as the GA, but uses no
selection, crossover or mutation. Given the same number of fitness evaluations
as a GA run, it keeps the best candidate found, preferring feasible candidates
(feasible always beats infeasible; then higher fitness).

Fairness notes: the candidate generator is the one used for the GA's initial
population (waypoints drawn on flyable cells, with bounded re-draws until the
route is feasible), so both methods get the same cheap feasibility-aware
sampling. Those feasibility checks are not fitness evaluations, for either method.
With the same seed the first ``population_size`` random-search candidates equal
the GA's initial population; random search then keeps drawing fresh candidates
until its budget is spent.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from uav_ga.chromosome import Chromosome
from uav_ga.config import FitnessConfig, PopulationConfig, UAVConfig
from uav_ga.fitness import FitnessBreakdown, FitnessEvaluator
from uav_ga.grid import DisasterGrid
from uav_ga.population import PopulationGenerator


@dataclass
class RandomSearchResult:
    seed: int
    budget: int
    best_chromosome: Chromosome
    best_evaluation: FitnessBreakdown
    best_so_far: List[float] = field(default_factory=list)  # one entry per evaluation
    evaluations: int = 0
    feasible_count: int = 0
    mean_fitness: float = 0.0
    runtime_seconds: float = 0.0


class RandomSearch:
    def __init__(self, grid: DisasterGrid, uav: Optional[UAVConfig] = None,
                 fitness_config: Optional[FitnessConfig] = None,
                 population_config: Optional[PopulationConfig] = None):
        self.grid = grid
        self.uav = uav or UAVConfig()
        self.evaluator = FitnessEvaluator(grid, self.uav, fitness_config)
        self.sampler = PopulationGenerator(grid, self.uav, population_config)

    def run(self, budget: int, seed: int) -> RandomSearchResult:
        """Evaluate exactly ``budget`` random chromosomes (seeded) and keep the best."""
        if budget < 1:
            raise ValueError("budget must be >= 1")
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        best_chrom, best_eval = None, None
        history: List[float] = []
        total, feasible = 0.0, 0
        for _ in range(budget):
            chrom = self.sampler.random_chromosome(rng)
            ev = self.evaluator.evaluate(chrom)
            total += ev.fitness
            feasible += ev.feasible
            if best_eval is None or (ev.feasible, ev.fitness) > (best_eval.feasible, best_eval.fitness):
                best_chrom, best_eval = chrom, ev
            history.append(best_eval.fitness)
        return RandomSearchResult(seed=seed, budget=budget, best_chromosome=best_chrom,
                                  best_evaluation=best_eval, best_so_far=history, evaluations=budget,
                                  feasible_count=feasible, mean_fitness=total / budget,
                                  runtime_seconds=time.perf_counter() - start)
