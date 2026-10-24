"""Segment traversal and hard-constraint detection (Step 3)."""

import numpy as np
import pytest

from tests.helpers import UAV, blank_grid, chromosome, near_base_waypoints, plan
from uav_ga.constraints import ConstraintChecker


def cells(g, p0, p1):
    return [(x, y) for x, y, _ in g.segment_cells(p0, p1)]


def test_traversal_horizontal_and_lengths():
    g = blank_grid()
    seg = g.segment_cells((2, 3), (6, 3))
    assert [(x, y) for x, y, _ in seg] == [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]
    assert sum(l for _, _, l in seg) == pytest.approx(4.0)
    assert seg[0][2] == pytest.approx(0.5) and seg[2][2] == pytest.approx(1.0)


def test_traversal_diagonal_and_sloped_length_sum():
    g = blank_grid()
    for p0, p1 in [((0, 0), (5, 5)), ((3, 2), (17, 9)), ((10, 10), (2, 4))]:
        total = sum(l for _, _, l in g.segment_cells(p0, p1))
        assert total == pytest.approx(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def test_traversal_visits_cells_between_endpoints():
    g = blank_grid()
    assert (5, 3) in cells(g, (0, 0), (10, 6))  # passes through the middle cell
    assert cells(g, (4, 4), (4, 4)) == [(4, 4)]


def test_traversal_reports_out_of_bounds_cells():
    g = blank_grid()
    assert any(not g.in_bounds(x, y) for x, y in cells(g, (45, 10), (55, 10)))


def test_endpoints_free_but_segment_crosses_obstacle():
    g = blank_grid()
    g.obstacles[10, 20] = True  # (x=20, y=10) in the middle of the route
    p = plan(g.bases[0], [(15, 10), (25, 10), (25, 12), (15, 12), (15, 14)])
    assert g.is_flyable(15, 10) and g.is_flyable(25, 10)  # endpoints are fine
    r = ConstraintChecker(g, UAV).check_drone(p)
    assert r.obstacle_cells >= 1 and not r.feasible


def test_segment_crossing_no_fly_zone_detected():
    g = blank_grid()
    g.no_fly[8:13, 18:23] = True
    p = plan(g.bases[1], [(15, 10), (28, 10), (28, 12), (15, 12), (15, 14)])
    r = ConstraintChecker(g, UAV).check_drone(p)
    assert r.no_fly_cells > 0 and r.obstacle_cells == 0 and not r.feasible


def test_return_leg_is_checked():
    g = blank_grid()
    g.obstacles[5, 12] = True  # on the straight line from (12, 10) down to base (12, 1)
    p = plan(g.bases[0], [(12, 10), (13, 10), (14, 10), (15, 10), (16, 10)])
    # last waypoint -> base crosses (12,5)? route returns from (16,10) to (12,1)
    g.obstacles[:] = False
    g.obstacles[5, 14] = True
    assert ConstraintChecker(g, UAV).check_drone(p).obstacle_cells >= 1


def test_out_of_bounds_route_is_infeasible():
    g = blank_grid()
    p = plan(g.bases[2], [(45, 5), (60, 5), (45, 8), (44, 9), (43, 10)])
    r = ConstraintChecker(g, UAV).check_drone(p)
    assert r.out_of_bounds_cells > 0 and not r.feasible


def test_clean_route_is_feasible():
    g = blank_grid()
    c = chromosome(g, near_base_waypoints(g))
    checker = ConstraintChecker(g, UAV)
    assert checker.is_feasible(c)
    assert all(r.blocked_cells == 0 and r.range_excess == 0 for r in checker.check(c))


def test_hazard_cells_do_not_block():
    g = blank_grid()
    g.hazards[:] = 1.0
    assert ConstraintChecker(g, UAV).is_feasible(chromosome(g, near_base_waypoints(g)))


def test_flight_range_violation():
    g = blank_grid()
    far = [(12, 48), (12, 2), (12, 48), (12, 2), (12, 48)]  # ~ 6 legs of ~47 cells
    r = ConstraintChecker(g, UAV).check_drone(plan(g.bases[0], far))
    assert r.distance > UAV.max_range
    assert r.range_excess == pytest.approx(r.distance - UAV.max_range)
    assert not r.feasible and r.blocked_cells == 0


def test_range_exactly_at_limit_is_feasible():
    g = blank_grid()
    p = plan(g.bases[0], [(12, 11), (12, 1), (12, 11), (12, 1), (12, 11)])
    r = ConstraintChecker(g, UAV).check_drone(p)
    assert r.range_excess == 0.0 and r.feasible
