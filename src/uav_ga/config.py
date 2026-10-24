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
