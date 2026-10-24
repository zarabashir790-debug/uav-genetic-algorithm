"""Genetic operators designed for the hybrid UAV chromosome.

All operators return NEW chromosomes; their inputs are never modified and
children never share arrays with parents.

Gene alignment used by the crossover/mutation operators
-------------------------------------------------------
A DronePlan has n waypoints and n + 1 segments. Waypoint ``w`` is paired with
the segment that ARRIVES at it (segment ``w``), so speeds[w] and sensors[w]
travel with waypoints[w]. The last segment (index n, the return leg to the
fixed base) is only exchanged by drone-level crossover. Bases are never changed.

These operators only guarantee chromosome STRUCTURE and waypoint-cell validity.
Route feasibility (segments, return leg, range) is checked afterwards by the GA
through ConstraintChecker; a valid waypoint is not proof of a safe route.
"""

from typing import List, Sequence, Tuple

import numpy as np

from uav_ga.chromosome import Chromosome, SensorMode
from uav_ga.config import UAVConfig
from uav_ga.fitness import FitnessBreakdown
from uav_ga.grid import DisasterGrid


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
def selection_key(ev: FitnessBreakdown) -> Tuple[bool, float]:
    """Ordering used everywhere: feasible beats infeasible, then higher fitness wins.

    Fitness is already negative for infeasible plans, so this agrees with plain
    fitness ordering; the explicit flag keeps the rule visible and safe with
    negative values.
    """
    return (ev.feasible, ev.fitness)


def tournament_selection(evaluations: Sequence[FitnessBreakdown], tournament_size: int,
                         rng: np.random.Generator) -> int:
    """Return the index of the winner of a tournament of random contestants.

    Contestants are drawn without replacement (with replacement only if the
    tournament is larger than the population). The population is not modified.
    """
    n = len(evaluations)
    if n == 0:
        raise ValueError("cannot select from an empty population")
    if tournament_size < 1:
        raise ValueError("tournament_size must be >= 1")
    contestants = rng.choice(n, size=tournament_size, replace=tournament_size > n)
    return int(max(contestants, key=lambda i: selection_key(evaluations[i])))


# ----------------------------------------------------------------------
# Crossover
# ----------------------------------------------------------------------
def drone_crossover(a: Chromosome, b: Chromosome, rng: np.random.Generator) -> Tuple[Chromosome, Chromosome]:
    """Swap k complete drone plans (1 <= k < n_drones) between the parents.

    Plan i of both parents flies from base i, so swapped plans keep the base
    assignment. Whole plans move, so each route keeps its own speeds and sensors.
    """
    n = a.n_drones
    c1, c2 = a.copy(), b.copy()
    if n < 2:
        return c1, c2
    k = int(rng.integers(1, n))
    for i in rng.choice(n, size=k, replace=False):
        c1.plans[i], c2.plans[i] = c2.plans[i], c1.plans[i]
    return c1, c2


def waypoint_crossover(a: Chromosome, b: Chromosome, rng: np.random.Generator) -> Tuple[Chromosome, Chromosome]:
    """Swap a consecutive section of waypoints (with arrival speeds and sensors) per drone.

    Each drone is chosen with probability 0.5 (at least one is always chosen).
    For a chosen drone a start/end index i <= j is drawn and waypoints, speeds
    and sensors at positions i..j are exchanged, so route order stays meaningful
    and the three arrays stay aligned. The return-leg genes are never swapped.
    """
    c1, c2 = a.copy(), b.copy()
    chosen = [i for i in range(a.n_drones) if rng.random() < 0.5]
    if not chosen:
        chosen = [int(rng.integers(a.n_drones))]
    for d in chosen:
        p1, p2 = c1.plans[d], c2.plans[d]
        n = p1.n_waypoints
        i = int(rng.integers(n))
        j = int(rng.integers(i, n))
        sl = slice(i, j + 1)
        for arr1, arr2 in ((p1.waypoints, p2.waypoints), (p1.speeds, p2.speeds), (p1.sensors, p2.sensors)):
            tmp = arr1[sl].copy()
            arr1[sl] = arr2[sl]
            arr2[sl] = tmp
    return c1, c2


# ----------------------------------------------------------------------
# Mutation
# ----------------------------------------------------------------------
class Mutator:
    """Gene-level mutations: each gene mutates independently with probability ``prob``."""

    def __init__(self, grid: DisasterGrid, uav: UAVConfig, waypoint_sigma: float = 3.0,
                 replace_prob: float = 0.1, speed_sigma_fraction: float = 0.15):
        self.grid = grid
        self.uav = uav
        self.waypoint_sigma = waypoint_sigma
        self.replace_prob = replace_prob
        self.speed_sigma = speed_sigma_fraction * (uav.speed_max - uav.speed_min)
        ys, xs = np.nonzero(grid.flyable_mask)
        self._flyable = np.column_stack([xs, ys])

    def mutate_waypoints(self, c: Chromosome, prob: float, rng: np.random.Generator) -> Chromosome:
        """Spatial mutation: move a waypoint to a nearby flyable cell, or (with
        probability ``replace_prob``) replace it by a random flyable cell."""
        child = c.copy()
        for plan in child.plans:
            for w in range(plan.n_waypoints):
                if rng.random() >= prob:
                    continue
                if rng.random() < self.replace_prob:
                    x, y = self._flyable[rng.integers(len(self._flyable))]
                    plan.waypoints[w] = (x, y)
                    continue
                for _ in range(10):  # bounded search for a valid nearby cell
                    step = np.rint(rng.normal(0.0, self.waypoint_sigma, 2)).astype(int)
                    x, y = plan.waypoints[w] + step
                    if (step != 0).any() and self.grid.in_bounds(int(x), int(y)) \
                            and self.grid.is_flyable(int(x), int(y)):
                        plan.waypoints[w] = (x, y)
                        break
        return child

    def mutate_speeds(self, c: Chromosome, prob: float, rng: np.random.Generator) -> Chromosome:
        """Gaussian speed perturbation, clipped to [speed_min, speed_max]."""
        child = c.copy()
        for plan in child.plans:
            for s in range(plan.n_segments):
                if rng.random() < prob:
                    plan.speeds[s] = np.clip(plan.speeds[s] + rng.normal(0.0, self.speed_sigma),
                                             self.uav.speed_min, self.uav.speed_max)
        return child

    def mutate_sensors(self, c: Chromosome, prob: float, rng: np.random.Generator) -> Chromosome:
        """Switch a segment's sensor to one of the two other modes."""
        child = c.copy()
        modes = [m.value for m in SensorMode]
        for plan in child.plans:
            for s in range(plan.n_segments):
                if rng.random() < prob:
                    plan.sensors[s] = rng.choice([m for m in modes if m != plan.sensors[s]])
        return child

    def mutate(self, c: Chromosome, prob: float, rng: np.random.Generator) -> Chromosome:
        child = self.mutate_waypoints(c, prob, rng)
        child = self.mutate_speeds(child, prob, rng)
        return self.mutate_sensors(child, prob, rng)
