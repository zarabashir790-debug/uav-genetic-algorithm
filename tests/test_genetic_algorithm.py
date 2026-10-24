"""GA loop: elitism, termination, history, determinism, constraint guard (Step 4)."""

import numpy as np
import pytest

from tests.helpers import UAV, blank_grid
from uav_ga.config import GAConfig, GridConfig, PopulationConfig
from uav_ga.fitness import FitnessEvaluator
from uav_ga.genetic_algorithm import GeneticAlgorithm
from uav_ga.grid import DisasterGrid
from uav_ga.population import PopulationGenerator

SMALL = dict(population_size=10, max_generations=6)


@pytest.fixture(scope="module")
def grid():
    return DisasterGrid(GridConfig(seed=42))


def run(grid, **kw):
    return GeneticAlgorithm(grid, config=GAConfig(**{**SMALL, **kw})).run()


def test_config_defaults_and_validation():
    c = GAConfig()
    assert (c.population_size, c.max_generations, c.crossover_prob, c.mutation_prob) == (30, 60, 0.8, 0.15)
    assert (c.tournament_size, c.elite_count, c.seed) == (3, 2, 42)
    for bad in (dict(population_size=1), dict(elite_count=30), dict(tournament_size=0),
                dict(crossover_prob=1.5), dict(mutation_prob=-0.1), dict(crossover_method="x"),
                dict(early_stopping_patience=0), dict(max_attempts=0), dict(max_generations=-1)):
        with pytest.raises(ValueError):
            GAConfig(**bad)


def test_history_lengths_and_content(grid):
    r = run(grid)
    n = r.generations_run + 1
    assert r.generations == list(range(n))
    for series in (r.generation_best, r.generation_average, r.generation_worst, r.best_so_far,
                   r.feasible_count, r.diversity, r.best_so_far_chromosomes, r.best_so_far_evaluations):
        assert len(series) == n
    assert all(w <= a <= b for w, a, b in zip(r.generation_worst, r.generation_average, r.generation_best))
    assert all(0 <= f <= 10 for f in r.feasible_count)
    assert r.evaluations == 10 + r.generations_run * (10 - 2)  # elites are not re-evaluated
    assert r.initial_population_best == r.generation_best[0]


def test_best_so_far_is_running_maximum_and_monotone(grid):
    r = run(grid, max_generations=12)
    assert all(b >= a for a, b in zip(r.best_so_far, r.best_so_far[1:]))
    for g in range(len(r.generation_best)):
        assert r.best_so_far[g] == pytest.approx(max(r.generation_best[: g + 1]))
    assert r.best_so_far[-1] == r.best_evaluation.fitness


def test_best_so_far_never_below_generation_best_even_without_elitism(grid):
    r = run(grid, elite_count=0, max_generations=15, mutation_prob=0.4)
    assert all(bs >= gb for bs, gb in zip(r.best_so_far, r.generation_best))
    assert all(b >= a for a, b in zip(r.best_so_far, r.best_so_far[1:]))


def test_elitism_keeps_generation_best_from_dropping(grid):
    r = run(grid, elite_count=2, max_generations=10, mutation_prob=0.5)
    assert all(b >= a - 1e-12 for a, b in zip(r.generation_best, r.generation_best[1:]))


def test_elites_survive_unchanged(grid):
    ga = GeneticAlgorithm(grid, config=GAConfig(**SMALL, elite_count=2))
    pop = PopulationGenerator(grid, UAV, PopulationConfig(size=10, seed=42)).generate()
    ev = ga.evaluator.evaluate_population(pop)
    top2 = sorted(range(10), key=lambda i: ev[i].fitness, reverse=True)[:2]
    r = GeneticAlgorithm(grid, config=GAConfig(population_size=10, max_generations=1, elite_count=2)).run(pop)
    assert r.generation_best[1] >= max(ev[i].fitness for i in top2) - 1e-12


def test_stops_at_max_generations(grid):
    r = run(grid, max_generations=4)
    assert r.generations_run == 4 and "max_generations" in r.stop_reason


def test_zero_generations_returns_initial_population_result(grid):
    r = run(grid, max_generations=0)
    assert r.generations_run == 0 and len(r.generation_best) == 1
    assert r.best_evaluation.fitness == r.initial_population_best


def test_early_stopping_triggers_without_meaningful_improvement(grid):
    r = run(grid, max_generations=50, early_stopping_patience=2, min_improvement=10.0)
    assert r.generations_run == 2 and "early stopping" in r.stop_reason


def test_early_stopping_is_off_by_default(grid):
    r = run(grid, max_generations=8)
    assert r.generations_run == 8 and r.config.early_stopping_patience is None


def test_deterministic_for_a_fixed_seed(grid):
    a, b = run(grid, seed=7), run(grid, seed=7)
    assert a.generation_best == b.generation_best and a.generation_average == b.generation_average
    assert a.best_chromosome.signature() == b.best_chromosome.signature()
    assert a.diversity == b.diversity
    c = run(grid, seed=8)
    assert c.generation_average != a.generation_average


@pytest.mark.parametrize("method", ["drone", "waypoint", "mixed"])
def test_all_crossover_methods_run_and_give_valid_best(grid, method):
    r = run(grid, crossover_method=method)
    assert r.best_chromosome.validate(grid, UAV) == []
    assert r.best_evaluation.feasible


def test_final_best_is_independently_feasible_and_matches_fresh_evaluation(grid):
    r = run(grid, max_generations=10)
    fresh = FitnessEvaluator(grid, UAV).evaluate(r.best_chromosome)
    assert fresh.fitness == pytest.approx(r.best_evaluation.fitness)
    assert fresh.feasible and r.best_evaluation.feasible


def test_supplied_population_is_not_modified(grid):
    pop = PopulationGenerator(grid, UAV, PopulationConfig(size=10, seed=1)).generate()
    before = [c.signature() for c in pop]
    GeneticAlgorithm(grid, config=GAConfig(**SMALL)).run(pop)
    assert [c.signature() for c in pop] == before


def test_snapshots_are_independent_copies(grid):
    r = run(grid)
    sigs = [c.signature() for c in r.best_so_far_chromosomes]
    r.best_chromosome.plans[0].waypoints[0, 0] += 1
    assert [c.signature() for c in r.best_so_far_chromosomes] == sigs


def test_progress_callback_called_every_generation(grid):
    seen = []
    GeneticAlgorithm(grid, config=GAConfig(**SMALL)).run(progress=lambda g, r: seen.append(g))
    assert seen == list(range(7))


# ---- constraint guard: failed regeneration never yields a falsely feasible classification ----
def blocked_copy(c, grid):
    bad = c.copy()
    x, y = np.argwhere(grid.obstacles)[0][::-1]
    bad.plans[0].waypoints[0] = (x, y)  # waypoint in an obstacle
    return bad


def test_mutation_guard_falls_back_to_unchanged_parent(grid):
    ga = GeneticAlgorithm(grid, config=GAConfig(**SMALL, max_attempts=3))
    pop = PopulationGenerator(grid, UAV, PopulationConfig(size=3, seed=5)).generate()
    parent = pop[0]
    assert ga.checker.is_feasible(parent)
    ga.mutator.mutate = lambda c, p, rng: blocked_copy(c, grid)  # every attempt is infeasible
    out = ga._mutate(parent.copy(), np.random.default_rng(0))
    assert out.signature() == parent.signature() and ga.mutation_fallbacks == 1
    assert ga.evaluator.evaluate(out).feasible  # still honestly classified


def test_crossover_guard_falls_back_to_parent_copies(grid):
    ga = GeneticAlgorithm(grid, config=GAConfig(**SMALL, max_attempts=3))
    pop = PopulationGenerator(grid, UAV, PopulationConfig(size=3, seed=5)).generate()
    a, b = pop[0], pop[1]
    ea, eb = ga.evaluator.evaluate(a), ga.evaluator.evaluate(b)
    ga.crossover_pair = lambda x, y, rng: (blocked_copy(x, grid), blocked_copy(y, grid))
    c1, c2 = ga._crossover(a, b, ea, eb, np.random.default_rng(0))
    assert c1.signature() == a.signature() and c2.signature() == b.signature()
    assert ga.crossover_fallbacks == 1 and c1 is not a and not np.shares_memory(c1.plans[0].waypoints, a.plans[0].waypoints)


def test_infeasible_offspring_are_never_classified_feasible(grid):
    ga = GeneticAlgorithm(grid, config=GAConfig(**SMALL))
    pop = PopulationGenerator(grid, UAV, PopulationConfig(size=3, seed=5)).generate()
    bad = blocked_copy(pop[0], grid)
    ev = ga.evaluator.evaluate(bad)
    assert not ev.feasible and ev.fitness < -1.0
    assert not ga._acceptable(bad, parents_feasible=True)  # validate() rejects the blocked waypoint
    assert ev.fitness < ga.evaluator.evaluate(pop[1]).fitness


def test_final_population_individuals_pass_validation_after_many_operations(grid):
    r = run(grid, max_generations=15, mutation_prob=0.4, crossover_prob=1.0)
    for c in r.best_so_far_chromosomes:
        assert c.validate(grid, UAV) == []
    assert r.best_evaluation.feasible and r.best_evaluation.penalty == 0.0
