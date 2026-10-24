"""Fitness function behaviour (Step 3)."""

import numpy as np
import pytest

from tests.helpers import UAV, blank_grid, chromosome, near_base_waypoints, plan
from uav_ga.chromosome import Chromosome
from uav_ga.config import FitnessConfig, GridConfig, PopulationConfig
from uav_ga.fitness import FitnessEvaluator
from uav_ga.grid import DisasterGrid
from uav_ga.population import PopulationGenerator

# one drone sweeps the line y=10 from x=10 to x=30; the other two stay near their bases
LINE = [(10, 10), (15, 10), (20, 10), (25, 10), (30, 10)]


def scenario(sensor=2, speed=3.0):
    g = blank_grid()
    wps = near_base_waypoints(g)
    wps[0] = LINE
    plans = [plan(b, w, speed, 0) for b, w in zip(g.bases, wps)]
    plans[0] = plan(g.bases[0], LINE, speed, sensor)
    return g, Chromosome(plans)


def test_sensor_off_gives_no_coverage_or_detection():
    g, c = scenario(sensor=0)
    g.priority[8:13, 8:32] = 1.0
    g.targets[10, 20] = True
    b = FitnessEvaluator(g, UAV).evaluate(c)
    assert b.coverage == 0.0 and b.detection_rate == 0.0 and b.detected_targets == 0


def test_sensor_on_beats_off_and_high_beats_low():
    g, c = scenario(sensor=0)
    g.priority[8:13, 8:32] = 1.0
    g.targets[10, 20] = True
    ev = FitnessEvaluator(g, UAV)
    results = {}
    for mode in (0, 1, 2):
        _, cm = scenario(sensor=mode)
        results[mode] = ev.evaluate(cm)
    assert results[2].coverage > results[1].coverage > results[0].coverage == 0.0
    assert results[2].energy > results[1].energy > results[0].energy  # more sensor power
    assert results[2].fitness > results[0].fitness


def test_target_detection_depends_on_distance_and_mode():
    g, _ = scenario()
    g.targets[13, 20] = True  # 3 cells from the line: HIGH (r=4) sees it, LOW (r=2) does not
    ev = FitnessEvaluator(g, UAV)
    assert ev.evaluate(scenario(sensor=2)[1]).detection_rate == 1.0
    assert ev.evaluate(scenario(sensor=1)[1]).detection_rate == 0.0
    g.targets[40, 5] = True  # far away: never seen
    assert ev.evaluate(scenario(sensor=2)[1]).detection_rate == 0.5


def test_no_double_counting_across_drones():
    g = blank_grid()
    g.priority[:] = 1.0  # every cell counts once, so coverage is exactly covered cells / 2500
    g.targets[10, 20] = True  # lies on the line all three drones sweep
    three = Chromosome([plan(b, LINE, 3.0, 2) for b in g.bases])
    ev = FitnessEvaluator(g, UAV)
    b = ev.evaluate(three)
    assert b.detected_targets == 1 and b.detection_rate == 1.0  # counted once, not three times
    cov = ev.coverage_map(three)
    assert b.coverage == pytest.approx(cov.sum() / cov.size)  # union of footprints
    assert 0.0 <= b.coverage <= 1.0
    # three overlapping drones cover far less than three times one drone's area
    single = Chromosome([plan(g.bases[0], LINE, 3.0, 2)] +
                        [plan(x, w, 3.0, 0) for x, w in zip(g.bases[1:], near_base_waypoints(g)[1:])])
    assert b.coverage < 3 * ev.evaluate(single).coverage


def test_zero_coverage_edge_cases_do_not_crash():
    g = blank_grid()  # no priority, no targets
    b = FitnessEvaluator(g, UAV).evaluate(scenario(sensor=2)[1])
    assert b.coverage == 0.0 and b.detection_rate == 0.0 and b.feasible
    assert 0.0 <= b.fitness <= 1.0


def test_time_energy_hand_calculation():
    g, _ = scenario()
    p = plan(g.bases[0], [(12, 11), (12, 1), (12, 11), (12, 1), (12, 11)], speed=2.0, sensor=0)
    c = chromosome(g, [[(12, 11), (12, 1), (12, 11), (12, 1), (12, 11)]] * 3, speed=2.0, sensor=0)
    b = FitnessEvaluator(g, UAV).evaluate(c)
    d = b.drones[0]
    assert d.distance == pytest.approx(p.route_length()) == pytest.approx(60.0)
    assert d.flight_time == pytest.approx(30.0)  # 60 / 2
    power = UAV.hover_power + UAV.drag_coeff * 2.0 ** 3
    assert d.energy == pytest.approx(power * 30.0)
    assert b.flight_time == pytest.approx(sum(x.flight_time for x in b.drones))


def test_sensor_energy_adds_to_flight_energy():
    g, _ = scenario()
    base = chromosome(g, near_base_waypoints(g), sensor=0)
    high = chromosome(g, near_base_waypoints(g), sensor=2)
    ev = FitnessEvaluator(g, UAV)
    e0, e1 = ev.evaluate(base), ev.evaluate(high)
    assert e1.energy - e0.energy == pytest.approx(UAV.high_sensor.power * e0.flight_time)


def test_faster_flight_is_shorter_time():
    g = blank_grid()
    ev = FitnessEvaluator(g, UAV)
    slow = ev.evaluate(chromosome(g, near_base_waypoints(g), speed=1.5))
    fast = ev.evaluate(chromosome(g, near_base_waypoints(g), speed=4.5))
    assert fast.flight_time < slow.flight_time


def test_hazard_exposure_measured_along_segments():
    g = blank_grid()
    g.hazards[5, :] = 0.5  # a hazard band at y=5, no waypoint inside it
    wps = [[(b[0], 2), (b[0], 8), (b[0], 2), (b[0], 8), (b[0], 2)] for b in g.bases]
    b = FitnessEvaluator(g, UAV).evaluate(chromosome(g, wps))
    # each drone crosses y in [4.5, 5.5] 4 times (up/down legs), 1 cell each, intensity 0.5
    assert b.drones[0].hazard_exposure == pytest.approx(4 * 0.5)
    assert b.hazard_exposure == pytest.approx(3 * 4 * 0.5)
    clear = FitnessEvaluator(g, UAV).evaluate(chromosome(g, near_base_waypoints(g)))
    assert clear.hazard_exposure == 0.0


def test_hazard_does_not_make_route_infeasible_but_lowers_score():
    g = blank_grid()
    c = chromosome(g, near_base_waypoints(g))
    ev = FitnessEvaluator(g, UAV)
    before = ev.evaluate(c)
    g.hazards[:] = 1.0
    after = ev.evaluate(c)
    assert after.feasible and after.fitness < before.fitness


def test_infeasible_always_below_feasible_even_with_great_coverage():
    g = blank_grid()
    g.priority[:] = 1.0
    g.targets[10:40:3, 10:40:3] = True
    g.no_fly[10, 20] = True  # blocks the sweeping drone's straight line
    ev = FitnessEvaluator(g, UAV)
    bad = ev.evaluate(scenario(sensor=2, speed=3.0)[1])
    assert not bad.feasible and bad.no_fly_cells > 0
    assert bad.fitness < 0 and bad.raw_score > 0.3  # high coverage, still negative
    poor = ev.evaluate(chromosome(g, near_base_waypoints(g), sensor=0))  # feasible, no sensing
    assert poor.feasible and poor.fitness >= 0
    assert bad.fitness < poor.fitness


def test_more_violation_means_lower_fitness():
    g = blank_grid()
    ev = FitnessEvaluator(g, UAV)
    g.obstacles[10, 18] = True
    one = ev.evaluate(scenario()[1])
    g.obstacles[9:12, 18:22] = True
    many = ev.evaluate(scenario()[1])
    assert many.obstacle_cells > one.obstacle_cells and many.fitness < one.fitness < 0


def test_range_violation_penalised_and_flagged():
    g = blank_grid()
    far = [(12, 48), (12, 2), (12, 48), (12, 2), (12, 48)]
    wps = near_base_waypoints(g)
    wps[0] = far
    b = FitnessEvaluator(g, UAV).evaluate(chromosome(g, wps, sensor=2))
    assert not b.feasible and b.range_excess > 0 and b.fitness < -1.0
    assert b.drones[0].range_excess > 0 and b.drones[1].feasible


def test_out_of_bounds_waypoint_is_infeasible_not_a_crash():
    g = blank_grid()
    wps = near_base_waypoints(g)
    wps[2][2] = (70, 5)
    b = FitnessEvaluator(g, UAV).evaluate(chromosome(g, wps, sensor=2))
    assert not b.feasible and b.out_of_bounds_cells > 0 and b.fitness < 0


def test_waypoint_inside_obstacle_is_infeasible():
    g = blank_grid()
    wps = near_base_waypoints(g)
    g.obstacles[wps[1][0][1], wps[1][0][0]] = True
    assert not FitnessEvaluator(g, UAV).evaluate(chromosome(g, wps)).feasible


def test_breakdown_contents_and_scalar_ranges():
    g = DisasterGrid(GridConfig(seed=42))
    pop = PopulationGenerator(g, UAV, PopulationConfig(size=10, seed=1)).generate()
    ev = FitnessEvaluator(g, UAV)
    for b in ev.evaluate_population(pop):
        d = b.as_dict()
        for key in ("coverage", "detection_rate", "energy", "flight_time", "hazard_exposure",
                    "penalty", "feasible", "fitness", "drones"):
            assert key in d
        assert len(b.drones) == 3
        assert 0 <= b.coverage <= 1 and 0 <= b.detection_rate <= 1
        assert 0 <= b.energy_norm <= 1 and 0 <= b.time_norm <= 1 and 0 <= b.hazard_norm <= 1
        assert 0 <= b.raw_score <= 1
        assert (b.fitness >= 0) if b.feasible else (b.fitness < -1)
        assert b.summary()


def test_weights_normalised_and_configurable():
    g = blank_grid()
    ev = FitnessEvaluator(g, UAV, FitnessConfig(w_coverage=3, w_detection=2, w_energy=1, w_time=1, w_hazard=3))
    assert ev.weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        FitnessConfig(w_coverage=-1)
    with pytest.raises(ValueError):
        FitnessConfig(infeasible_base=0.5)


def test_same_chromosome_same_fitness_and_seeded_population_reproducible():
    g = DisasterGrid(GridConfig(seed=42))
    ev = FitnessEvaluator(g, UAV)
    run = lambda: [ev.fitness(c) for c in PopulationGenerator(g, UAV, PopulationConfig(seed=3)).generate()]
    assert run() == run()


def test_coverage_map_matches_evaluation():
    g, c = scenario(sensor=2)
    g.priority[8:13, 8:32] = 1.0
    ev = FitnessEvaluator(g, UAV)
    cov = ev.coverage_map(c)
    assert cov.shape == (50, 50) and cov.max() == UAV.high_sensor.effectiveness
    assert ev.evaluate(c).coverage == pytest.approx((g.priority * cov).sum() / g.priority.sum())
