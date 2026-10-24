"""Population initialisation (Step 3)."""

import numpy as np

from tests.helpers import UAV
from uav_ga.config import GridConfig, PopulationConfig
from uav_ga.constraints import ConstraintChecker
from uav_ga.grid import DisasterGrid
from uav_ga.population import PopulationGenerator


def make(seed=42, size=30, grid_seed=42):
    g = DisasterGrid(GridConfig(seed=grid_seed))
    return g, PopulationGenerator(g, UAV, PopulationConfig(size=size, seed=seed)).generate()


def test_default_size_is_30():
    assert PopulationConfig().size == 30
    assert len(make()[1]) == 30


def test_configurable_size():
    assert len(make(size=7)[1]) == 7


def test_every_chromosome_valid_with_three_drones():
    g, pop = make()
    for c in pop:
        assert c.n_drones == 3
        assert c.validate(g, UAV) == []


def test_waypoints_never_in_blocked_cells():
    g, pop = make()
    for c in pop:
        for p in c.plans:
            assert all(g.is_flyable(int(x), int(y)) for x, y in p.waypoints)


def test_speed_and_sensor_ranges():
    _, pop = make()
    for c in pop:
        for p in c.plans:
            assert p.speeds.min() >= UAV.speed_min and p.speeds.max() <= UAV.speed_max
            assert set(p.sensors.tolist()) <= {0, 1, 2}


def test_same_seed_reproducible_and_different_seed_differs():
    _, a = make(seed=5)
    _, b = make(seed=5)
    _, c = make(seed=6)
    assert [x.signature() for x in a] == [x.signature() for x in b]
    assert [x.signature() for x in a] != [x.signature() for x in c]


def test_population_is_diverse_without_duplicates():
    _, pop = make()
    assert len({c.signature() for c in pop}) == len(pop)
    assert PopulationGenerator.diversity(pop) > 5.0
    assert PopulationGenerator.diversity(pop[:1]) == 0.0


def test_most_initial_routes_are_feasible():
    g, pop = make()
    checker = ConstraintChecker(g, UAV)
    assert sum(checker.is_feasible(c) for c in pop) >= len(pop) // 2


def test_valid_across_grid_seeds():
    for gs in (1, 7, 123):
        g, pop = make(size=5, grid_seed=gs)
        assert all(c.validate(g, UAV) == [] for c in pop)
