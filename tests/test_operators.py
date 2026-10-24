"""Selection, crossover and mutation operators (Step 4)."""

import numpy as np
import pytest

from tests.helpers import UAV, blank_grid, chromosome, near_base_waypoints
from uav_ga.chromosome import Chromosome
from uav_ga.fitness import FitnessEvaluator
from uav_ga.grid import DisasterGrid
from uav_ga.config import GridConfig, PopulationConfig
from uav_ga.operators import (Mutator, drone_crossover, selection_key, tournament_selection,
                              waypoint_crossover)
from uav_ga.population import PopulationGenerator


def parents():
    g = DisasterGrid(GridConfig(seed=42))
    pop = PopulationGenerator(g, UAV, PopulationConfig(size=4, seed=3)).generate()
    return g, pop[0], pop[1]


def same_genes(a: Chromosome, b: Chromosome) -> bool:
    return a.signature() == b.signature()


def assert_structure(g, c):
    assert c.n_drones == 3
    assert [p.base for p in c.plans] == list(g.bases)
    for p in c.plans:
        assert p.waypoints.shape == (5, 2) and p.speeds.shape == (6,) and p.sensors.shape == (6,)
    assert c.validate(g, UAV) == []


def no_shared_memory(c, *others):
    for o in others:
        for p, q in zip(c.plans, o.plans):
            assert not np.shares_memory(p.waypoints, q.waypoints)
            assert not np.shares_memory(p.speeds, q.speeds)
            assert not np.shares_memory(p.sensors, q.sensors)


# ---------------- selection ----------------
def make_evals():
    g = blank_grid()
    ev = FitnessEvaluator(g, UAV)
    good = chromosome(g, near_base_waypoints(g), sensor=2)
    ok = chromosome(g, near_base_waypoints(g), sensor=0)
    bad = chromosome(g, near_base_waypoints(g), sensor=2)
    bad.plans[0].waypoints[0] = (70, 3)       # out of bounds -> infeasible
    worse = bad.copy()
    worse.plans[1].waypoints[0] = (70, 3)     # more violations -> lower fitness
    return [ev.evaluate(c) for c in (ok, bad, good, worse)]


def test_tournament_full_size_returns_the_best():
    evals = make_evals()
    rng = np.random.default_rng(0)
    for _ in range(20):
        i = tournament_selection(evals, len(evals), rng)
        assert evals[i].feasible
        assert evals[i].fitness == max(e.fitness for e in evals)


def test_tournament_prefers_feasible_over_infeasible():
    evals = make_evals()
    assert not evals[1].feasible and evals[0].feasible
    assert selection_key(evals[0]) > selection_key(evals[1])
    rng = np.random.default_rng(1)
    for _ in range(50):
        # contest of exactly one feasible vs infeasible individuals
        sub = [evals[0], evals[1]]
        assert tournament_selection(sub, 2, rng) == 0


def test_tournament_handles_all_negative_fitness():
    evals = make_evals()
    neg = [evals[1], evals[3]]
    assert all(e.fitness < 0 for e in neg)
    winner = tournament_selection(neg, 2, np.random.default_rng(0))
    assert neg[winner].fitness == max(e.fitness for e in neg)


def test_tournament_size_one_is_random_choice_and_covers_population():
    evals = make_evals()
    rng = np.random.default_rng(5)
    picks = {tournament_selection(evals, 1, rng) for _ in range(100)}
    assert picks == {0, 1, 2, 3}


def test_tournament_is_seed_reproducible_and_does_not_modify_inputs():
    evals = make_evals()
    before = [e.fitness for e in evals]
    a = [tournament_selection(evals, 3, np.random.default_rng(9)) for _ in range(5)]
    b = [tournament_selection(evals, 3, np.random.default_rng(9)) for _ in range(5)]
    assert a == b and [e.fitness for e in evals] == before


def test_tournament_larger_than_population_and_bad_arguments():
    evals = make_evals()
    assert 0 <= tournament_selection(evals, 10, np.random.default_rng(0)) < 4
    with pytest.raises(ValueError):
        tournament_selection(evals, 0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        tournament_selection([], 2, np.random.default_rng(0))


def test_higher_tournament_size_increases_selection_pressure():
    evals = make_evals()
    best = max(range(4), key=lambda i: evals[i].fitness)
    rng = np.random.default_rng(0)
    wins = {k: sum(tournament_selection(evals, k, rng) == best for _ in range(400)) for k in (1, 4)}
    assert wins[4] == 400 and wins[1] < 400


# ---------------- crossover ----------------
@pytest.mark.parametrize("seed", range(10))
def test_drone_crossover_swaps_whole_plans(seed):
    g, a, b = parents()
    a0, b0 = a.copy(), b.copy()
    c1, c2 = drone_crossover(a, b, np.random.default_rng(seed))
    assert same_genes(a, a0) and same_genes(b, b0)  # parents untouched
    for c in (c1, c2):
        assert_structure(g, c)
        no_shared_memory(c, a, b)
    swapped = [i for i in range(3) if not same_genes(Chromosome([c1.plans[i]]), Chromosome([a.plans[i]]))]
    assert 1 <= len(swapped) <= 2
    for i in range(3):
        assert same_genes(Chromosome([c1.plans[i]]), Chromosome([b.plans[i]])) == (i in swapped)
        assert same_genes(Chromosome([c2.plans[i]]), Chromosome([a.plans[i]])) == (i in swapped)


@pytest.mark.parametrize("seed", range(10))
def test_waypoint_crossover_keeps_genes_aligned(seed):
    g, a, b = parents()
    a0, b0 = a.copy(), b.copy()
    c1, c2 = waypoint_crossover(a, b, np.random.default_rng(seed))
    assert same_genes(a, a0) and same_genes(b, b0)
    for c in (c1, c2):
        assert_structure(g, c)
        no_shared_memory(c, a, b)
    changed = 0
    for d in range(3):
        pa, pb, p1 = a.plans[d], b.plans[d], c1.plans[d]
        from_b = []
        for w in range(5):
            triple = (tuple(p1.waypoints[w]), p1.speeds[w], p1.sensors[w])
            ta = (tuple(pa.waypoints[w]), pa.speeds[w], pa.sensors[w])
            tb = (tuple(pb.waypoints[w]), pb.speeds[w], pb.sensors[w])
            assert triple == ta or triple == tb       # waypoint, speed, sensor move together
            from_b.append(triple == tb and triple != ta)
        idx = [w for w, f in enumerate(from_b) if f]
        if idx:
            changed += 1
            assert idx == list(range(idx[0], idx[-1] + 1))  # one consecutive section
        assert p1.speeds[5] == pa.speeds[5] and p1.sensors[5] == pa.sensors[5]  # return leg untouched
    assert changed >= 1  # parents differ everywhere, so at least one drone exchanged a section


def test_crossover_of_identical_parents_gives_identical_children():
    g, a, _ = parents()
    for op in (drone_crossover, waypoint_crossover):
        c1, c2 = op(a, a.copy(), np.random.default_rng(0))
        assert same_genes(c1, a) and same_genes(c2, a)


def test_crossover_is_seed_reproducible():
    g, a, b = parents()
    for op in (drone_crossover, waypoint_crossover):
        x = op(a, b, np.random.default_rng(4))
        y = op(a, b, np.random.default_rng(4))
        assert same_genes(x[0], y[0]) and same_genes(x[1], y[1])


# ---------------- mutation ----------------
def mutator(g, **kw):
    return Mutator(g, UAV, **kw)


def test_zero_probability_mutation_changes_nothing_but_copies():
    g, a, _ = parents()
    child = mutator(g).mutate(a, 0.0, np.random.default_rng(0))
    assert same_genes(child, a) and child is not a
    no_shared_memory(child, a)


def test_spatial_mutation_moves_waypoints_to_valid_cells():
    g, a, _ = parents()
    a0 = a.copy()
    child = mutator(g, replace_prob=0.0).mutate_waypoints(a, 1.0, np.random.default_rng(0))
    assert same_genes(a, a0)
    assert_structure(g, child)
    moved = sum((p.waypoints != q.waypoints).any(axis=1).sum() for p, q in zip(child.plans, a.plans))
    assert moved >= 10  # 15 waypoints, nearly all moved
    for p, q in zip(child.plans, a.plans):
        assert np.abs(p.waypoints - q.waypoints).max() <= 15  # local move
        assert np.array_equal(p.speeds, q.speeds) and np.array_equal(p.sensors, q.sensors)


def test_spatial_mutation_never_enters_blocked_or_out_of_bounds_cells():
    g = DisasterGrid(GridConfig(seed=42))
    pop = PopulationGenerator(g, UAV, PopulationConfig(size=10, seed=2)).generate()
    m = mutator(g, waypoint_sigma=8.0, replace_prob=0.3)
    rng = np.random.default_rng(1)
    for c in pop:
        child = m.mutate_waypoints(c, 1.0, rng)
        for p in child.plans:
            for x, y in p.waypoints:
                assert g.in_bounds(int(x), int(y)) and g.is_flyable(int(x), int(y))


def test_waypoint_replacement_can_jump_anywhere_flyable():
    g, a, _ = parents()
    child = mutator(g, replace_prob=1.0).mutate_waypoints(a, 1.0, np.random.default_rng(0))
    far = max(np.abs(p.waypoints - q.waypoints).max() for p, q in zip(child.plans, a.plans))
    assert far > 10 and child.validate(g, UAV) == []


def test_speed_mutation_stays_in_bounds_and_leaves_other_genes():
    g, a, _ = parents()
    m = mutator(g)
    child = a
    for seed in range(20):
        child = m.mutate_speeds(child, 1.0, np.random.default_rng(seed))
        for p in child.plans:
            assert p.speeds.min() >= UAV.speed_min and p.speeds.max() <= UAV.speed_max
    assert any(not np.array_equal(p.speeds, q.speeds) for p, q in zip(child.plans, a.plans))
    for p, q in zip(child.plans, a.plans):
        assert np.array_equal(p.waypoints, q.waypoints) and np.array_equal(p.sensors, q.sensors)


def test_speed_mutation_clips_at_the_bounds():
    g = blank_grid()
    c = chromosome(g, near_base_waypoints(g), speed=UAV.speed_max)
    m = Mutator(g, UAV, speed_sigma_fraction=5.0)
    out = m.mutate_speeds(c, 1.0, np.random.default_rng(0))
    for p in out.plans:
        assert p.speeds.min() >= UAV.speed_min and p.speeds.max() <= UAV.speed_max


def test_sensor_mutation_changes_mode_to_a_valid_different_one():
    g, a, _ = parents()
    child = mutator(g).mutate_sensors(a, 1.0, np.random.default_rng(0))
    for p, q in zip(child.plans, a.plans):
        assert (p.sensors != q.sensors).all()
        assert set(p.sensors.tolist()) <= {0, 1, 2}
        assert np.array_equal(p.waypoints, q.waypoints) and np.array_equal(p.speeds, q.speeds)


def test_mutation_preserves_structure_and_parent_and_is_reproducible():
    g, a, _ = parents()
    a0 = a.copy()
    m = mutator(g)
    c1 = m.mutate(a, 0.3, np.random.default_rng(8))
    c2 = m.mutate(a, 0.3, np.random.default_rng(8))
    assert same_genes(a, a0) and same_genes(c1, c2) and not same_genes(c1, a)
    assert_structure(g, c1)
    no_shared_memory(c1, a)


def test_mutation_rate_controls_amount_of_change():
    g, a, _ = parents()
    m = mutator(g)
    def changed(p):
        c = m.mutate_sensors(a, p, np.random.default_rng(0))
        return sum((x.sensors != y.sensors).sum() for x, y in zip(c.plans, a.plans))
    assert changed(0.05) < changed(0.5) < changed(1.0) == 18
