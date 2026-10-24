"""Hard-constraint detection (penalty approach; no repair is performed).

A drone's route is feasible only if
  1. no cell traversed by ANY segment is an obstacle, a no-fly cell or outside the grid
     (exact grid traversal, not just waypoint endpoints), and
  2. the full loop length (including the return to base) is <= max_range.
Hazard cells are traversable and never make a route infeasible.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from uav_ga.chromosome import Chromosome, DronePlan
from uav_ga.config import UAVConfig
from uav_ga.grid import DisasterGrid

_TOL = 1e-9


@dataclass(frozen=True)
class DroneConstraintReport:
    distance: float
    range_excess: float
    obstacle_cells: int
    no_fly_cells: int
    out_of_bounds_cells: int

    @property
    def blocked_cells(self) -> int:
        return self.obstacle_cells + self.no_fly_cells + self.out_of_bounds_cells

    @property
    def feasible(self) -> bool:
        return self.blocked_cells == 0 and self.range_excess <= 0.0


class ConstraintChecker:
    def __init__(self, grid: DisasterGrid, uav: UAVConfig):
        self.grid = grid
        self.uav = uav

    def route_cells(self, plan: DronePlan) -> List[list]:
        """Traversal (x, y, length) lists, one per route segment."""
        return [self.grid.segment_cells(p0, p1) for p0, p1 in plan.segments()]

    def check_drone(self, plan: DronePlan, seg_cells: Optional[Sequence[list]] = None) -> DroneConstraintReport:
        if seg_cells is None:
            seg_cells = self.route_cells(plan)
        distance = plan.route_length()
        excess = max(0.0, distance - self.uav.max_range)
        if excess < _TOL:
            excess = 0.0
        obstacle = no_fly = oob = 0
        for x, y in {(x, y) for seg in seg_cells for x, y, _ in seg}:  # distinct cells
            if not self.grid.in_bounds(x, y):
                oob += 1
            elif self.grid.obstacles[y, x]:
                obstacle += 1
            elif self.grid.no_fly[y, x]:
                no_fly += 1
        return DroneConstraintReport(distance, excess, obstacle, no_fly, oob)

    def check(self, chromosome: Chromosome) -> List[DroneConstraintReport]:
        return [self.check_drone(p) for p in chromosome.plans]

    def is_feasible(self, chromosome: Chromosome) -> bool:
        return all(r.feasible for r in self.check(chromosome))
