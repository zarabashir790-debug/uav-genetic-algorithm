"""Multi-objective fitness evaluation.

Objectives (each normalised to [0, 1])
--------------------------------------
Maximise
  C  coverage      = sum(priority * coverage) / sum(priority)
  D  detection     = detected targets / total targets   (simulated proxy)
Minimise
  E  energy        = total energy / E_ref
  T  flight time   = total flight time / T_ref
  H  hazard        = total hazard exposure / H_ref

Scalar score for feasible plans (weights normalised so sum(w) = 1):

  raw = wC*C + wD*D + wE*(1 - E) + wT*(1 - T) + wH*(1 - H)       raw in [0, 1]

Hard constraints (obstacles, no-fly cells, out-of-bounds, max range):
  feasible   -> fitness = raw                                    (>= 0)
  infeasible -> fitness = -(infeasible_base + blocked_cell_penalty * blocked_cells
                            + range_penalty * sum(excess / max_range))   (< -1)
Because raw <= 1 <= infeasible_base, EVERY infeasible plan scores below EVERY
feasible plan regardless of coverage (feasibility-first rule). More violation
gives a lower score, so the GA still gets a gradient toward feasibility.

Physical models (abstract units; see UAVConfig)
-----------------------------------------------
Segment i: length L_i (Euclid between waypoints), speed v_i, sensor mode m_i.
  time    t_i = L_i / v_i
  power   P_i = hover_power + drag_coeff * v_i^3 + sensor_power[m_i]
  energy  e_i = P_i * t_i        (U-shaped in v: slow = long hover, fast = drag)
Totals are summed over all segments (including the return leg) and drones.
Hazard exposure = sum over traversed cells of (length inside cell * hazard
intensity), measured along every segment, not at waypoints only.

Sensing: while a segment's sensor is not OFF, a disc footprint (mode radius)
is placed on every grid cell the segment traverses. A cell's coverage is the
MAX effectiveness over all footprints of all drones, so cells and targets
are never double counted. A target is "detected" when its cell has coverage
> 0 (a geometric proxy, not computer-vision recognition). OFF gives nothing.

Normalisation references (design choices)
  E_ref = n_drones * max_range * worst energy per cell flown (at speed_min, HIGH sensor)
  T_ref = n_drones * max_range / speed_min
  H_ref = hazard_ref_fraction * n_drones * max_range * 1.0
Values above a reference are clipped to 1.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from uav_ga.chromosome import Chromosome, DronePlan, SensorMode
from uav_ga.config import FitnessConfig, SensorSpec, UAVConfig
from uav_ga.constraints import ConstraintChecker
from uav_ga.grid import DisasterGrid


@dataclass
class DroneMetrics:
    distance: float
    flight_time: float
    energy: float
    hazard_exposure: float
    obstacle_cells: int
    no_fly_cells: int
    out_of_bounds_cells: int
    range_excess: float
    feasible: bool


@dataclass
class FitnessBreakdown:
    fitness: float            # scalar used by the GA (negative => infeasible)
    raw_score: float          # weighted objective score, ignoring penalties
    feasible: bool
    penalty: float            # 0 if feasible
    # physical totals
    coverage: float           # priority-weighted coverage fraction in [0, 1]
    detection_rate: float     # detected / total targets
    detected_targets: int
    energy: float
    flight_time: float
    hazard_exposure: float
    # normalised objectives in [0, 1]
    energy_norm: float
    time_norm: float
    hazard_norm: float
    # violations
    obstacle_cells: int
    no_fly_cells: int
    out_of_bounds_cells: int
    range_excess: float
    drones: List[DroneMetrics] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        status = "FEASIBLE (all routes validated against obstacles, no-fly zones, bounds, range)" \
            if self.feasible else "INFEASIBLE (hard-constraint violation)"
        lines = [
            f"  status          : {status}",
            f"  fitness (GA)    : {self.fitness:.4f}   raw objective score {self.raw_score:.4f}",
            f"  coverage        : {self.coverage:.3f}  (priority-weighted)",
            f"  detection rate  : {self.detection_rate:.3f}  ({self.detected_targets} targets)",
            f"  energy          : {self.energy:.1f} eu   (normalised {self.energy_norm:.3f})",
            f"  flight time     : {self.flight_time:.1f} tu   (normalised {self.time_norm:.3f})",
            f"  hazard exposure : {self.hazard_exposure:.2f}   (normalised {self.hazard_norm:.3f})",
            f"  penalty         : {self.penalty:.3f}",
            f"  violations      : obstacle cells {self.obstacle_cells}, no-fly cells {self.no_fly_cells}, "
            f"out-of-bounds cells {self.out_of_bounds_cells}, range excess {self.range_excess:.1f}",
        ]
        for i, d in enumerate(self.drones, start=1):
            lines.append(f"  UAV {i}: dist {d.distance:.1f}, time {d.flight_time:.1f}, energy {d.energy:.1f}, "
                         f"hazard {d.hazard_exposure:.2f}, {'feasible' if d.feasible else 'INFEASIBLE'}")
        return "\n".join(lines)


class FitnessEvaluator:
    def __init__(self, grid: DisasterGrid, uav: Optional[UAVConfig] = None,
                 config: Optional[FitnessConfig] = None):
        self.grid = grid
        self.uav = uav or UAVConfig()
        self.config = config or FitnessConfig()
        self.checker = ConstraintChecker(grid, self.uav)

        c = self.config
        w = np.array([c.w_coverage, c.w_detection, c.w_energy, c.w_time, c.w_hazard], dtype=float)
        self.weights = w / w.sum()

        n = len(grid.bases)
        u = self.uav
        worst_energy_per_cell = max(
            (u.hover_power + u.drag_coeff * v ** 3 + u.high_sensor.power) / v
            for v in (u.speed_min, u.speed_max))
        self.energy_ref = n * u.max_range * worst_energy_per_cell
        self.time_ref = n * u.max_range / u.speed_min
        self.hazard_ref = c.hazard_ref_fraction * n * u.max_range * 1.0

        self._sensor: Dict[int, SensorSpec] = {SensorMode.LOW: u.low_sensor, SensorMode.HIGH: u.high_sensor}
        self._discs = {m: self._disc_offsets(s.radius) for m, s in self._sensor.items()}
        self._total_priority = float(grid.priority.sum())

    # ------------------------------------------------------------------
    def fitness(self, chromosome: Chromosome) -> float:
        return self.evaluate(chromosome).fitness

    def evaluate_population(self, population: List[Chromosome]) -> List[FitnessBreakdown]:
        return [self.evaluate(c) for c in population]

    def evaluate(self, chromosome: Chromosome) -> FitnessBreakdown:
        grid, u = self.grid, self.uav
        drones: List[DroneMetrics] = []
        centres = {SensorMode.LOW: set(), SensorMode.HIGH: set()}

        for plan in chromosome.plans:
            seg_cells = self.checker.route_cells(plan)
            report = self.checker.check_drone(plan, seg_cells)
            lengths = plan.segment_lengths()
            times = lengths / plan.speeds
            power = u.hover_power + u.drag_coeff * plan.speeds ** 3
            power = power + np.array([self._sensor[m].power if m != SensorMode.OFF else 0.0
                                      for m in plan.sensors])
            hazard = 0.0
            for seg, mode in zip(seg_cells, plan.sensors):
                for x, y, length in seg:
                    if not grid.in_bounds(x, y):
                        continue
                    hazard += length * grid.hazards[y, x]
                    if mode != SensorMode.OFF:
                        centres[SensorMode(int(mode))].add((x, y))
            drones.append(DroneMetrics(
                distance=report.distance, flight_time=float(times.sum()),
                energy=float((power * times).sum()), hazard_exposure=float(hazard),
                obstacle_cells=report.obstacle_cells, no_fly_cells=report.no_fly_cells,
                out_of_bounds_cells=report.out_of_bounds_cells,
                range_excess=report.range_excess, feasible=report.feasible))

        coverage_map = self._coverage_map(centres)
        coverage = (float((grid.priority * coverage_map).sum()) / self._total_priority
                    if self._total_priority > 0 else 0.0)
        n_targets = int(grid.targets.sum())
        detected = int(((coverage_map > 0) & grid.targets).sum())
        detection = detected / n_targets if n_targets else 0.0

        energy = sum(d.energy for d in drones)
        flight_time = sum(d.flight_time for d in drones)
        hazard_total = sum(d.hazard_exposure for d in drones)
        e_n = min(1.0, energy / self.energy_ref)
        t_n = min(1.0, flight_time / self.time_ref)
        h_n = min(1.0, hazard_total / self.hazard_ref)

        wC, wD, wE, wT, wH = self.weights
        raw = float(wC * coverage + wD * detection + wE * (1 - e_n) + wT * (1 - t_n) + wH * (1 - h_n))

        feasible = all(d.feasible for d in drones)
        blocked = sum(d.obstacle_cells + d.no_fly_cells + d.out_of_bounds_cells for d in drones)
        excess = sum(d.range_excess for d in drones)
        c = self.config
        penalty = 0.0 if feasible else (c.infeasible_base + c.blocked_cell_penalty * blocked
                                        + c.range_penalty * excess / u.max_range)
        return FitnessBreakdown(
            fitness=raw if feasible else -penalty, raw_score=raw, feasible=feasible, penalty=penalty,
            coverage=coverage, detection_rate=detection, detected_targets=detected,
            energy=energy, flight_time=flight_time, hazard_exposure=hazard_total,
            energy_norm=e_n, time_norm=t_n, hazard_norm=h_n,
            obstacle_cells=sum(d.obstacle_cells for d in drones),
            no_fly_cells=sum(d.no_fly_cells for d in drones),
            out_of_bounds_cells=sum(d.out_of_bounds_cells for d in drones),
            range_excess=excess, drones=drones)

    # ------------------------------------------------------------------
    def coverage_map(self, chromosome: Chromosome) -> np.ndarray:
        """Per-cell coverage in [0, 1] for a chromosome (useful for later heatmaps)."""
        centres = {SensorMode.LOW: set(), SensorMode.HIGH: set()}
        for plan in chromosome.plans:
            for seg, mode in zip(self.checker.route_cells(plan), plan.sensors):
                if mode == SensorMode.OFF:
                    continue
                centres[SensorMode(int(mode))].update((x, y) for x, y, _ in seg)
        return self._coverage_map(centres)

    def _coverage_map(self, centres) -> np.ndarray:
        cov = np.zeros((self.grid.height, self.grid.width), dtype=float)
        for mode in (SensorMode.LOW, SensorMode.HIGH):
            eff = self._sensor[mode].effectiveness
            dx, dy = self._discs[mode]
            for cx, cy in centres[mode]:
                xs, ys = cx + dx, cy + dy
                ok = (xs >= 0) & (xs < self.grid.width) & (ys >= 0) & (ys < self.grid.height)
                xs, ys = xs[ok], ys[ok]
                cov[ys, xs] = np.maximum(cov[ys, xs], eff)
        return cov

    @staticmethod
    def _disc_offsets(radius: float):
        r = int(np.ceil(radius))
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        inside = dx ** 2 + dy ** 2 <= radius ** 2
        return dx[inside], dy[inside]
