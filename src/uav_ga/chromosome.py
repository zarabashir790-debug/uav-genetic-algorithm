"""Chromosome encoding for a three-UAV mission strategy.

Route model (used everywhere: constraints, distance, time, energy, sensing)
--------------------------------------------------------------------------
Every drone starts at its fixed base, visits its operational waypoints in
index order, then RETURNS TO ITS BASE:

    base -> wp[0] -> wp[1] -> ... -> wp[n-1] -> base      (n + 1 segments)

Waypoint order matters: reversing it gives a different route. Each segment
``i`` has its own speed ``speeds[i]`` and sensor mode ``sensors[i]``; the
sensor is on (or off) for the whole segment. The deployment position of a
drone is its first waypoint, ``wp[0]``, where it begins its mission.

Hybrid encoding per drone: integer (x, y) waypoints, float speeds and
discrete sensor modes (OFF / LOW / HIGH).
"""

from enum import IntEnum
from typing import List, Tuple

import numpy as np

from uav_ga.config import UAVConfig
from uav_ga.grid import Cell, DisasterGrid


class SensorMode(IntEnum):
    OFF = 0
    LOW = 1
    HIGH = 2


class DronePlan:
    """One UAV's mission: fixed base, ordered waypoints, speeds and sensor modes."""

    def __init__(self, base: Cell, waypoints, speeds, sensors):
        self.base: Cell = (int(base[0]), int(base[1]))
        self.waypoints = np.asarray(waypoints, dtype=int).reshape(-1, 2)
        self.speeds = np.asarray(speeds, dtype=float)
        self.sensors = np.asarray(sensors, dtype=int)

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)

    @property
    def n_segments(self) -> int:
        return self.n_waypoints + 1  # includes the return leg

    @property
    def deployment_position(self) -> Cell:
        """Derived from the first operational waypoint."""
        return (int(self.waypoints[0, 0]), int(self.waypoints[0, 1]))

    @property
    def route_points(self) -> np.ndarray:
        """Full closed route, shape (n_waypoints + 2, 2): base, waypoints, base."""
        base = np.array([self.base], dtype=int)
        return np.vstack([base, self.waypoints, base])

    def segments(self) -> List[Tuple[Cell, Cell]]:
        pts = self.route_points
        return [((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in zip(pts[:-1], pts[1:])]

    def segment_lengths(self) -> np.ndarray:
        pts = self.route_points.astype(float)
        return np.hypot(*(pts[1:] - pts[:-1]).T)

    def route_length(self) -> float:
        return float(self.segment_lengths().sum())

    def validate(self, uav: UAVConfig) -> List[str]:
        """Structural checks on speeds and sensors (grid checks live in Chromosome)."""
        problems = []
        if self.waypoints.shape != (uav.n_waypoints, 2):
            problems.append(f"expected {uav.n_waypoints} waypoints, got shape {self.waypoints.shape}")
        if self.speeds.shape != (self.n_segments,):
            problems.append(f"expected {self.n_segments} speeds, got shape {self.speeds.shape}")
        elif (self.speeds < uav.speed_min).any() or (self.speeds > uav.speed_max).any():
            problems.append(f"speeds outside [{uav.speed_min}, {uav.speed_max}]")
        if self.sensors.shape != (self.n_segments,):
            problems.append(f"expected {self.n_segments} sensor modes, got shape {self.sensors.shape}")
        elif not np.isin(self.sensors, [m.value for m in SensorMode]).all():
            problems.append("sensor modes must be 0 (OFF), 1 (LOW) or 2 (HIGH)")
        return problems

    def copy(self) -> "DronePlan":
        return DronePlan(self.base, self.waypoints.copy(), self.speeds.copy(), self.sensors.copy())

    def describe(self) -> str:
        wps = " -> ".join(f"({x},{y})" for x, y in self.waypoints)
        modes = ",".join(SensorMode(int(s)).name for s in self.sensors)
        speeds = ",".join(f"{v:.2f}" for v in self.speeds)
        return (f"base={self.base} deploy={self.deployment_position}\n"
                f"    route : base -> {wps} -> base   (length {self.route_length():.1f})\n"
                f"    speeds: [{speeds}]\n"
                f"    sensor: [{modes}]")


class Chromosome:
    """A complete mission strategy: one DronePlan per UAV (drone i uses base i)."""

    def __init__(self, plans: List[DronePlan]):
        self.plans = list(plans)

    @property
    def n_drones(self) -> int:
        return len(self.plans)

    def validate(self, grid: DisasterGrid, uav: UAVConfig) -> List[str]:
        """Encoding validity: structure, bases, waypoint bounds, waypoint cells.

        This does NOT prove the routes are safe; segments are checked by
        ConstraintChecker / FitnessEvaluator.
        """
        problems = []
        if self.n_drones != len(grid.bases):
            problems.append(f"expected {len(grid.bases)} drone plans, got {self.n_drones}")
        for i, plan in enumerate(self.plans):
            problems += [f"drone {i}: {p}" for p in plan.validate(uav)]
            if i < len(grid.bases) and plan.base != tuple(grid.bases[i]):
                problems.append(f"drone {i}: base {plan.base} != grid base {grid.bases[i]}")
            for x, y in plan.waypoints:
                if not grid.in_bounds(int(x), int(y)):
                    problems.append(f"drone {i}: waypoint ({x},{y}) out of bounds")
                elif grid.is_blocked(int(x), int(y)):
                    problems.append(f"drone {i}: waypoint ({x},{y}) in obstacle/no-fly cell")
        return problems

    def is_valid(self, grid: DisasterGrid, uav: UAVConfig) -> bool:
        return not self.validate(grid, uav)

    def signature(self) -> tuple:
        """Hashable fingerprint of all genes (used to detect duplicates)."""
        return tuple((p.base, p.waypoints.tobytes(), p.speeds.tobytes(), p.sensors.tobytes())
                     for p in self.plans)

    def copy(self) -> "Chromosome":
        return Chromosome([p.copy() for p in self.plans])

    def describe(self) -> str:
        return "\n".join(f"  UAV {i + 1}: {p.describe()}" for i, p in enumerate(self.plans))
