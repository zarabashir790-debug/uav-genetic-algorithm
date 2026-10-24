"""Genetic Algorithm main loop.

Generation cycle
----------------
1. Evaluate the population with FitnessEvaluator.
2. Copy the best ``elite_count`` chromosomes unchanged into the next generation (elitism).
3. Fill the rest: tournament-select two parents, crossover with probability
   ``crossover_prob`` (else copy), mutate each child, evaluate.
4. The new generation replaces the old one (generational replacement).
5. Stop at ``max_generations``, or when the optional ``max_evaluations``
   fitness-evaluation budget is used up, or, if configured, after
   ``early_stopping_patience`` generations without best-so-far improving by
   more than ``min_improvement``.

Evaluation budget (used for fair comparisons): only calls to the fitness
evaluator count; constraint-feasibility checks inside the regeneration guard
do not. When the budget runs out in the middle of a generation, the remaining
population slots are filled with the best not-yet-copied parents of the
previous generation (already evaluated, so they cost nothing); the run then
stops with exactly ``max_evaluations`` evaluations. With ``max_evaluations=None``
behaviour is unchanged.

Constraint handling in reproduction (bounded regeneration, never assumed safe)
-----------------------------------------------------------------------------
A crossover/mutation result is accepted only if it passes structural validation
AND its routes are feasible (every segment, return leg, range) according to
ConstraintChecker, or if the parent(s) it came from were already infeasible
(there is nothing to protect). Otherwise the operator is retried with fresh
randomness up to ``max_attempts`` times; if every attempt fails, an UNCHANGED
copy of the parent is used. The fitness function re-checks every chromosome
independently, so an infeasible individual can never be classified feasible.
Fallback counts are reported in ``GAResult``.

Terminology
-----------
generation-best fitness : best fitness inside one generation's population.
best-so-far fitness     : best fitness seen in any generation up to now. It never
                          decreases (elitism + explicit tracking).
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from uav_ga.chromosome import Chromosome
from uav_ga.config import FitnessConfig, GAConfig, PopulationConfig, UAVConfig
from uav_ga.constraints import ConstraintChecker
from uav_ga.fitness import FitnessBreakdown, FitnessEvaluator
from uav_ga.grid import DisasterGrid
from uav_ga.operators import (Mutator, drone_crossover, selection_key,
                              tournament_selection, waypoint_crossover)
from uav_ga.population import PopulationGenerator


@dataclass
class GAResult:
    config: GAConfig
    best_chromosome: Chromosome
    best_evaluation: FitnessBreakdown
    # per-generation history; index 0 is the initial population
    generations: List[int] = field(default_factory=list)
    generation_best: List[float] = field(default_factory=list)
    generation_average: List[float] = field(default_factory=list)
    generation_worst: List[float] = field(default_factory=list)
    best_so_far: List[float] = field(default_factory=list)
    feasible_count: List[int] = field(default_factory=list)
    diversity: List[float] = field(default_factory=list)
    # best-so-far chromosome / evaluation at each generation (independent copies)
    best_so_far_chromosomes: List[Chromosome] = field(default_factory=list)
    best_so_far_evaluations: List[FitnessBreakdown] = field(default_factory=list)
    cumulative_evaluations: List[int] = field(default_factory=list)  # evaluations used up to each generation
    initial_population_best: float = 0.0
    generations_run: int = 0
    stop_reason: str = ""
    evaluations: int = 0
    crossover_fallbacks: int = 0
    mutation_fallbacks: int = 0
    runtime_seconds: float = 0.0

    @property
    def middle_generation(self) -> int:
        return self.generations_run // 2


class GeneticAlgorithm:
    def __init__(self, grid: DisasterGrid, uav: Optional[UAVConfig] = None,
                 fitness_config: Optional[FitnessConfig] = None, config: Optional[GAConfig] = None):
        self.grid = grid
        self.uav = uav or UAVConfig()
        self.config = config or GAConfig()
        self.evaluator = FitnessEvaluator(grid, self.uav, fitness_config)
        self.checker = ConstraintChecker(grid, self.uav)
        c = self.config
        self.mutator = Mutator(grid, self.uav, c.waypoint_sigma, c.waypoint_replace_prob,
                               c.speed_sigma_fraction)
        self.crossover_fallbacks = 0
        self.mutation_fallbacks = 0

    # ------------------------------------------------------------------
    def run(self, population: Optional[List[Chromosome]] = None,
            progress: Optional[Callable[[int, GAResult], None]] = None) -> GAResult:
        """Evolve and return the full history. Same config + seed + grid => same result."""
        cfg = self.config
        start = time.perf_counter()
        rng = np.random.default_rng(cfg.seed)
        self.crossover_fallbacks = self.mutation_fallbacks = 0

        if population is None:
            population = PopulationGenerator(
                self.grid, self.uav, PopulationConfig(size=cfg.population_size, seed=cfg.seed)).generate()
        population = [c.copy() for c in population]
        evals = self.evaluator.evaluate_population(population)
        n_evals = len(evals)

        result = GAResult(cfg, population[0], evals[0])
        best_chrom, best_eval = None, None
        stale = 0
        gen = 0
        while True:
            # --- record this generation -------------------------------------------------
            top = max(range(len(evals)), key=lambda i: selection_key(evals[i]))
            fits = np.array([e.fitness for e in evals])
            improved = best_eval is None or \
                selection_key(evals[top])[1] > best_eval.fitness + cfg.min_improvement
            if best_eval is None or selection_key(evals[top]) > selection_key(best_eval):
                best_chrom, best_eval = population[top].copy(), evals[top]
            stale = 0 if improved else stale + 1
            result.generations.append(gen)
            result.generation_best.append(float(fits.max()))
            result.generation_average.append(float(fits.mean()))
            result.generation_worst.append(float(fits.min()))
            result.best_so_far.append(best_eval.fitness)
            result.feasible_count.append(sum(e.feasible for e in evals))
            result.diversity.append(PopulationGenerator.diversity(population))
            result.best_so_far_chromosomes.append(best_chrom.copy())
            result.best_so_far_evaluations.append(best_eval)
            result.cumulative_evaluations.append(n_evals)
            if gen == 0:
                result.initial_population_best = float(fits.max())
            if progress:
                progress(gen, result)

            # --- termination -------------------------------------------------------------
            if gen >= cfg.max_generations:
                result.stop_reason = f"reached max_generations ({cfg.max_generations})"
                break
            if cfg.max_evaluations is not None and n_evals >= cfg.max_evaluations:
                result.stop_reason = f"evaluation budget exhausted ({n_evals} evaluations)"
                break
            if cfg.early_stopping_patience is not None and stale >= cfg.early_stopping_patience:
                result.stop_reason = (f"early stopping: no improvement > {cfg.min_improvement:g} "
                                      f"for {cfg.early_stopping_patience} generations")
                break

            # --- next generation ----------------------------------------------------------
            order = sorted(range(len(evals)), key=lambda i: selection_key(evals[i]), reverse=True)
            new_pop = [population[i].copy() for i in order[:cfg.elite_count]]
            new_evals = [evals[i] for i in order[:cfg.elite_count]]
            budget = cfg.max_evaluations
            while len(new_pop) < cfg.population_size:
                if budget is not None and n_evals >= budget:
                    break
                i = tournament_selection(evals, cfg.tournament_size, rng)
                j = tournament_selection(evals, cfg.tournament_size, rng)
                kids = self._crossover(population[i], population[j], evals[i], evals[j], rng) \
                    if rng.random() < cfg.crossover_prob else (population[i].copy(), population[j].copy())
                for kid in kids:
                    if len(new_pop) == cfg.population_size or (budget is not None and n_evals >= budget):
                        break
                    kid = self._mutate(kid, rng)
                    new_pop.append(kid)
                    new_evals.append(self.evaluator.evaluate(kid))
                    n_evals += 1
            for i in order[cfg.elite_count:]:  # budget ran out mid-generation: carry over evaluated parents
                if len(new_pop) == cfg.population_size:
                    break
                new_pop.append(population[i].copy())
                new_evals.append(evals[i])
            population, evals = new_pop, new_evals
            gen += 1

        result.best_chromosome, result.best_evaluation = best_chrom, best_eval
        result.generations_run = gen
        result.evaluations = n_evals
        result.crossover_fallbacks = self.crossover_fallbacks
        result.mutation_fallbacks = self.mutation_fallbacks
        result.runtime_seconds = time.perf_counter() - start
        return result

    # ------------------------------------------------------------------
    def crossover_pair(self, a: Chromosome, b: Chromosome,
                       rng: np.random.Generator) -> Tuple[Chromosome, Chromosome]:
        """Apply the configured crossover method once (no feasibility guard)."""
        method = self.config.crossover_method
        if method == "mixed":
            method = "drone" if rng.random() < 0.5 else "waypoint"
        return (drone_crossover if method == "drone" else waypoint_crossover)(a, b, rng)

    def _acceptable(self, child: Chromosome, parents_feasible: bool) -> bool:
        if child.validate(self.grid, self.uav):
            return False
        return self.checker.is_feasible(child) or not parents_feasible

    def _crossover(self, a, b, ea, eb, rng) -> Tuple[Chromosome, Chromosome]:
        protect = ea.feasible or eb.feasible
        for _ in range(self.config.max_attempts):
            c1, c2 = self.crossover_pair(a, b, rng)
            if self._acceptable(c1, protect) and self._acceptable(c2, protect):
                return c1, c2
        self.crossover_fallbacks += 1
        return a.copy(), b.copy()

    def _mutate(self, child: Chromosome, rng) -> Chromosome:
        protect = self.checker.is_feasible(child)  # an infeasible child has nothing to protect
        for _ in range(self.config.max_attempts):
            mutant = self.mutator.mutate(child, self.config.mutation_prob, rng)
            if self._acceptable(mutant, protect):
                return mutant
        self.mutation_fallbacks += 1
        return child
