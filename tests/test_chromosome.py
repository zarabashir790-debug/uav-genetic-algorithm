"""Chromosome encoding (Step 3)."""

import numpy as np

from tests.helpers import UAV, blank_grid, chromosome, near_base_waypoints, plan
from uav_ga.chromosome import Chromosome, DronePlan, SensorMode


def test_three_drone_structure_and_bases():
    g = blank_grid()
    c = chromosome(g, near_base_waypoints(g))
    assert c.n_drones == 3
    assert [p.base for p in c.plans] == list(g.bases)
    assert c.is_valid(g, UAV)


def test_plan_has_five_waypoints_and_matching_profiles():
    p = plan((12, 1), [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9)])
    assert p.waypoints.shape == (5, 2) and np.issubdtype(p.waypoints.dtype, np.integer)
    assert p.n_segments == 6 and p.speeds.shape == (6,) and p.sensors.shape == (6,)
    assert np.issubdtype(p.speeds.dtype, np.floating)


def test_deployment_position_is_first_waypoint():
    p = plan((12, 1), [(7, 9), (2, 6), (3, 7), (4, 8), (5, 9)])
    assert p.deployment_position == (7, 9)


def test_route_returns_to_base():
    p = plan((12, 1), [(12, 11), (22, 11), (22, 1), (12, 5), (12, 9)])
    pts = p.route_points
    assert tuple(pts[0]) == tuple(pts[-1]) == (12, 1)
    assert len(pts) == 7 and len(p.segments()) == 6
    expected = 10 + 10 + 10 + np.hypot(10, 4) + 4 + 8
    assert abs(p.route_length() - expected) < 1e-9


def test_waypoint_order_defines_route():
    a = plan((12, 1), [(12, 11), (30, 11), (30, 1), (12, 5), (12, 9)])
    b = plan((12, 1), [(12, 9), (12, 5), (30, 1), (30, 11), (12, 11)])
    assert a.segments() != b.segments()


def test_validation_catches_bad_genes():
    g = blank_grid()
    c = chromosome(g, near_base_waypoints(g))
    c.plans[0].speeds[0] = UAV.speed_max + 1
    c.plans[1].sensors[2] = 3
    c.plans[2].waypoints[0] = (60, 2)
    problems = c.validate(g, UAV)
    assert any("drone 0" in p and "speeds" in p for p in problems)
    assert any("drone 1" in p and "sensor" in p for p in problems)
    assert any("drone 2" in p and "out of bounds" in p for p in problems)


def test_validation_catches_blocked_waypoint_and_wrong_base_and_count():
    g = blank_grid()
    g.obstacles[10, 20] = True
    g.no_fly[11, 21] = True
    c = chromosome(g, near_base_waypoints(g))
    c.plans[0].waypoints[0] = (20, 10)
    c.plans[0].waypoints[1] = (21, 11)
    c.plans[1].base = (0, 0)
    problems = c.validate(g, UAV)
    assert sum("obstacle/no-fly" in p for p in problems) == 2
    assert any("base" in p for p in problems)
    assert Chromosome(c.plans[:2]).validate(g, UAV)  # wrong drone count


def test_copy_is_independent_and_signature_detects_change():
    g = blank_grid()
    c = chromosome(g, near_base_waypoints(g))
    d = c.copy()
    assert c.signature() == d.signature()
    d.plans[0].waypoints[0, 0] += 1
    assert c.signature() != d.signature()
    assert c.plans[0].waypoints[0, 0] != d.plans[0].waypoints[0, 0]


def test_sensor_mode_values():
    assert [m.name for m in SensorMode] == ["OFF", "LOW", "HIGH"]
    assert isinstance(DronePlan((0, 0), [(1, 1)], [1.0, 1.0], [0, 2]).sensors[1], np.integer)
