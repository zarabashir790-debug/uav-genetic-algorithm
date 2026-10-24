"""Random-search baseline (Step 5)."""

import numpy as np
import pytest

from tests.helpers import UAV
from uav_ga.config import GridConfig
from uav_ga.fitness import FitnessEvaluator
from uav_ga.grid import DisasterGrid
from uav_ga.population import PopulationGenerator
from uav_ga.config import PopulationConfig
from uav_ga.random_search import RandomSearch


@pytest.fixture(scope="module")
def grid():
    return DisasterGrid(GridConfig(seed=42))


def test_uses_exactly_the_requested_budget(grid):
    r = RandomSearch(grid).run(budget=40, seed=3)
    assert r.evaluations == 40 and len(r.best_so_far) == 40


def test_best_so_far_monotone_and_ends_at_best(grid):
    r = RandomSearch(grid).run(budget=60, seed=4)
    assert all(b >= a for a, b in zip(r.best_so_far, r.best_so_far[1:]))
    assert r.best_so_far[-1] == r.best_evaluation.fitness


def test_reproducible_for_a_seed_and_different_across_seeds(grid):
    a, b, c = (RandomSearch(grid).run(40, s) for s in (5, 5, 6))
    assert a.best_so_far == b.best_so_far
    assert a.best_chromosome.signature() == b.best_chromosome.signature()
    assert a.best_so_far != c.best_so_far


def test_best_is_valid_and_matches_a_fresh_evaluation(grid):
    r = RandomSearch(grid).run(30, 7)
    assert r.best_chromosome.validate(grid, UAV) == []
    fresh = FitnessEvaluator(grid, UAV).evaluate(r.best_chromosome)
    assert fresh.fitness == pytest.approx(r.best_evaluation.fitness)
    assert fresh.feasible == r.best_evaluation.feasible


def test_prefers_feasible_candidates(grid):
    r = RandomSearch(grid).run(50, 8)
    assert r.feasible_count > 0 and r.best_evaluation.feasible


def test_first_draws_equal_the_ga_initial_population_for_the_same_seed(grid):
    pop = PopulationGenerator(grid, UAV, PopulationConfig(size=5, seed=9)).generate()
    sampler = RandomSearch(grid).sampler
    rng = np.random.default_rng(9)
    drawn = [sampler.random_chromosome(rng) for _ in range(5)]
    assert [c.signature() for c in pop] == [c.signature() for c in drawn]


def test_rejects_non_positive_budget(grid):
    with pytest.raises(ValueError):
        RandomSearch(grid).run(0, 1)
