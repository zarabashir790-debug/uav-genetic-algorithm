"""Random initial population of valid, mostly route-feasible chromosomes."""

from typing import List, Optional

import numpy as np

from uav_ga.chromosome import Chromosome, DronePlan, SensorMode
from uav_ga.config import PopulationConfig, UAVConfig
from uav_ga.constraints import ConstraintChecker
from uav_ga.grid import DisasterGrid


class PopulationGenerator:
    """Draws random chromosomes with a seeded RNG (same seed -> same population).

    Guarantees: three drones on the grid's bases, waypoints on flyable cells,
    speeds within bounds, valid sensor modes, no duplicate chromosomes.

    Waypoints are drawn sequentially and each must be reachable from the
    previous point by an unblocked segment. A drone plan is re-drawn up to
    ``max_tries`` times until its whole loop (return leg and range included)
    is feasible. If that fails the last draw is kept, so a few individuals may
    still be infeasible; the fitness function penalises those.
    """

    SEGMENT_PICKS = 30  # candidate waypoints tried per step

    def __init__(self, grid: DisasterGrid, uav: Optional[UAVConfig] = None,
                 config: Optional[PopulationConfig] = None):
        self.grid = grid
        self.uav = uav or UAVConfig()
        self.config = config or PopulationConfig()
        self.checker = ConstraintChecker(grid, self.uav)
        ys, xs = np.nonzero(grid.flyable_mask)
        self._flyable = np.column_stack([xs, ys])

    def generate(self) -> List[Chromosome]:
        rng = np.random.default_rng(self.config.seed)
        population: List[Chromosome] = []
        seen = set()
        while len(population) < self.config.size:
            chrom = self.random_chromosome(rng)
            if chrom.signature() in seen:
                continue
            seen.add(chrom.signature())
            population.append(chrom)
        return population

    def random_chromosome(self, rng: np.random.Generator) -> Chromosome:
        return Chromosome([self._random_plan(base, rng) for base in self.grid.bases])

    def _random_plan(self, base, rng: np.random.Generator) -> DronePlan:
        plan = None
        for _ in range(self.config.max_tries):
            plan = self._draw_plan(base, rng)
            if self.checker.check_drone(plan).feasible:
                break
        return plan

    def _draw_plan(self, base, rng: np.random.Generator) -> DronePlan:
        waypoints = []
        prev = base
        for _ in range(self.uav.n_waypoints):
            candidate = None
            for _ in range(self.SEGMENT_PICKS):
                candidate = tuple(int(v) for v in self._flyable[rng.integers(len(self._flyable))])
                if self._segment_clear(prev, candidate):
                    break
            waypoints.append(candidate)
            prev = candidate
        n_seg = self.uav.n_waypoints + 1
        speeds = rng.uniform(self.uav.speed_min, self.uav.speed_max, n_seg)
        sensors = rng.integers(0, len(SensorMode), n_seg)
        return DronePlan(base, waypoints, speeds, sensors)

    def _segment_clear(self, p0, p1) -> bool:
        return all(self.grid.is_flyable(x, y) for x, y, _ in self.grid.segment_cells(p0, p1))

    @staticmethod
    def diversity(population: List[Chromosome]) -> float:
        """Mean pairwise waypoint distance (cells) between chromosomes; 0 = all identical."""
        genes = np.array([np.vstack([p.waypoints for p in c.plans]).ravel() for c in population], dtype=float)
        n = len(genes)
        if n < 2:
            return 0.0
        dists = [np.abs(genes[i] - genes[j]).mean() for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(dists))
