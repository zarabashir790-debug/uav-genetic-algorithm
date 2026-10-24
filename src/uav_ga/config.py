"""Configuration dataclasses.

Implemented: GridConfig (Step 2).
Planned, not implemented: UAV limits, fitness weights, GA hyperparameters.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:
    """Parameters that control how a DisasterGrid scenario is generated."""

    width: int = 50
    height: int = 50
    seed: int = 42
    n_obstacles: int = 8
    n_no_fly_zones: int = 3
    n_hazard_zones: int = 5
    n_priority_zones: int = 4
    n_targets: int = 15
    n_bases: int = 3
    base_keepout: int = 2  # cells around each base kept free of obstacles/no-fly/hazards


@dataclass(frozen=True)
class SensorSpec:
    """One active sensor mode. Units: cells, dimensionless, energy units per time unit."""

    radius: float         # footprint radius around the sensing position (cells)
    effectiveness: float  # coverage credit for a covered cell, in (0, 1]
    power: float          # extra power draw while the sensor is on


@dataclass(frozen=True)
class UAVConfig:
    """Physical and sensing parameters shared by all UAVs.

    Abstract units: distance in grid cells, time in time units (tu),
    speed in cells/tu, energy in energy units (eu), power in eu/tu.
    """

    n_waypoints: int = 5
    speed_min: float = 1.0
    speed_max: float = 5.0
    max_range: float = 140.0     # max length of the whole loop base -> waypoints -> base
    hover_power: float = 1.0     # power needed just to stay airborne
    drag_coeff: float = 0.02     # aerodynamic power = drag_coeff * v^3
    low_sensor: SensorSpec = SensorSpec(radius=2.0, effectiveness=0.5, power=0.3)
    high_sensor: SensorSpec = SensorSpec(radius=4.0, effectiveness=1.0, power=1.0)


@dataclass(frozen=True)
class FitnessConfig:
    """Objective weights (normalised to sum to 1 by the evaluator) and penalty settings."""

    w_coverage: float = 0.30
    w_detection: float = 0.25
    w_energy: float = 0.15
    w_time: float = 0.10
    w_hazard: float = 0.20
    hazard_ref_fraction: float = 0.25   # "heavy" exposure = this fraction of max range at hazard 1.0
    infeasible_base: float = 1.0        # must be >= 1 so infeasible < every feasible score
    blocked_cell_penalty: float = 0.05  # per distinct blocked/out-of-bounds cell traversed
    range_penalty: float = 1.0          # per (excess distance / max_range), summed over drones

    def __post_init__(self):
        weights = (self.w_coverage, self.w_detection, self.w_energy, self.w_time, self.w_hazard)
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("fitness weights must be non-negative with a positive sum")
        if self.infeasible_base < 1.0:
            raise ValueError("infeasible_base must be >= 1 so infeasible plans always rank last")


@dataclass(frozen=True)
class PopulationConfig:
    """Initial-population settings."""

    size: int = 30
    seed: int = 42
    max_tries: int = 200  # attempts per drone to draw a route-feasible plan
