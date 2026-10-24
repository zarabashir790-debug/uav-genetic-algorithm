"""Shared helpers: a controllable blank grid and a hand-built chromosome."""

import numpy as np

from uav_ga.chromosome import Chromosome, DronePlan
from uav_ga.config import GridConfig, UAVConfig
from uav_ga.grid import DisasterGrid

UAV = UAVConfig()


def blank_grid() -> DisasterGrid:
    """50x50 grid with the standard bases (12,1),(25,1),(38,1) and every layer emptied."""
    g = DisasterGrid(GridConfig(seed=42))
    g.obstacles[:] = False
    g.no_fly[:] = False
    g.hazards[:] = 0.0
    g.priority[:] = 0.0
    g.targets[:] = False
    return g


def plan(base, waypoints, speed=3.0, sensor=0) -> DronePlan:
    n = len(waypoints) + 1
    return DronePlan(base, waypoints, np.full(n, speed), np.full(n, sensor))


def chromosome(grid, waypoints_per_drone, speed=3.0, sensor=0) -> Chromosome:
    return Chromosome([plan(b, w, speed, sensor) for b, w in zip(grid.bases, waypoints_per_drone)])


def near_base_waypoints(grid):
    """Five short, in-bounds waypoints per drone, close to its base."""
    return [[(b[0] + i, b[1] + 3) for i in range(5)] for b in grid.bases]
