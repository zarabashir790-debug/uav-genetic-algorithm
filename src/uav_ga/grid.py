"""Disaster grid environment.

Coordinate convention
---------------------
A cell is addressed as ``(x, y)``: ``x`` is the column (0 .. width-1, left to
right) and ``y`` is the row (0 .. height-1, bottom to top when plotted).
Every layer is a NumPy array of shape ``(height, width)`` and is indexed
``layer[y, x]``.

Layers (kept separate on purpose)
---------------------------------
obstacles : bool,  True = blocked, cannot be crossed
no_fly    : bool,  True = restricted, must never be entered
hazards   : float, risk intensity in [0, 1]; traversable but costly
priority  : float, coverage importance weight in [0, 1]
targets   : bool,  True = a simulated target to be detected later
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from uav_ga.config import GridConfig

Cell = Tuple[int, int]


class DisasterGrid:
    """A reproducible, discretised 2D disaster scenario."""

    def __init__(self, config: Optional[GridConfig] = None):
        self.config = config or GridConfig()
        if self.config.width < 10 or self.config.height < 10:
            raise ValueError("grid must be at least 10x10")
        self.width = self.config.width
        self.height = self.config.height
        self.seed = self.config.seed

        shape = (self.height, self.width)
        self.obstacles = np.zeros(shape, dtype=bool)
        self.no_fly = np.zeros(shape, dtype=bool)
        self.hazards = np.zeros(shape, dtype=float)
        self.priority = np.zeros(shape, dtype=float)
        self.targets = np.zeros(shape, dtype=bool)
        self.bases: List[Cell] = []

        self._rng = np.random.default_rng(self.seed)
        self._generate()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        """True if (x, y) lies inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and bool(self.obstacles[y, x])

    def is_no_fly(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and bool(self.no_fly[y, x])

    def is_blocked(self, x: int, y: int) -> bool:
        """True if a drone may not occupy the cell (out of bounds, obstacle or no-fly)."""
        if not self.in_bounds(x, y):
            return True
        return bool(self.obstacles[y, x] or self.no_fly[y, x])

    def is_flyable(self, x: int, y: int) -> bool:
        """True if a drone may occupy the cell. Hazard cells are flyable."""
        return not self.is_blocked(x, y)

    def hazard_at(self, x: int, y: int) -> float:
        self._check(x, y)
        return float(self.hazards[y, x])

    def priority_at(self, x: int, y: int) -> float:
        self._check(x, y)
        return float(self.priority[y, x])

    def has_target(self, x: int, y: int) -> bool:
        self._check(x, y)
        return bool(self.targets[y, x])

    def cell_info(self, x: int, y: int) -> Dict[str, object]:
        """Everything the grid knows about one cell."""
        self._check(x, y)
        return {
            "x": x,
            "y": y,
            "obstacle": bool(self.obstacles[y, x]),
            "no_fly": bool(self.no_fly[y, x]),
            "flyable": self.is_flyable(x, y),
            "hazard": float(self.hazards[y, x]),
            "priority": float(self.priority[y, x]),
            "target": bool(self.targets[y, x]),
        }

    def segment_cells(self, p0: Cell, p1: Cell) -> List[Tuple[int, int, float]]:
        """Exact grid traversal of the straight segment p0 -> p1 (conservative).

        Cell (x, y) is the square [x-0.5, x+0.5) x [y-0.5, y+0.5) centred on
        the integer coordinate. Returns ``(x, y, length)`` for every cell the
        segment passes through, in travel order, where ``length`` is the
        distance flown inside that cell. Cells outside the grid are returned
        too (not clipped) so callers can detect out-of-bounds routes.

        Conservative corner policy: if the segment passes exactly through a
        shared corner of four cells, all four cells count as touched. The two
        cells it only grazes are reported with ``length == 0.0``. So a route can
        never slip diagonally between two touching blocked cells, and grazing
        the corner of a single blocked cell is also a collision. Zero-length
        entries carry no flight distance, hazard or sensing.
        """
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx, dy = x1 - x0, y1 - y0
        total = math.hypot(dx, dy)
        if total == 0.0:
            return [(math.floor(x0 + 0.5), math.floor(y0 + 0.5), 0.0)]

        crossings = ([], [])  # parametric t of x-boundary and y-boundary crossings
        for axis, (start, delta, lo, hi) in enumerate(((x0, dx, min(x0, x1), max(x0, x1)),
                                                       (y0, dy, min(y0, y1), max(y0, y1)))):
            if delta == 0.0:
                continue
            n = math.floor(lo - 0.5) + 1  # first cell boundary strictly above lo
            while n + 0.5 < hi:
                crossings[axis].append((n + 0.5 - start) / delta)
                n += 1
        ts = sorted([0.0, 1.0] + crossings[0] + crossings[1])
        corner_ts = [t for t in crossings[0] if any(abs(t - u) < 1e-9 for u in crossings[1])]

        cells: List[Tuple[int, int, float]] = []
        for t_a, t_b in zip(ts, ts[1:]):
            if t_b - t_a < 1e-9:
                continue
            tm = 0.5 * (t_a + t_b)
            cx = math.floor(x0 + tm * dx + 0.5)
            cy = math.floor(y0 + tm * dy + 0.5)
            cells.append((cx, cy, (t_b - t_a) * total))
            if any(abs(t_b - t) < 1e-9 for t in corner_ts):
                px, py = x0 + t_b * dx, y0 + t_b * dy  # the shared corner (half-integer coords)
                for gx in (math.floor(px), math.floor(px) + 1):
                    for gy in (math.floor(py), math.floor(py) + 1):
                        cells.append((gx, gy, 0.0))
        return cells

    @property
    def flyable_mask(self) -> np.ndarray:
        """Boolean layer, True where a drone may fly."""
        return ~(self.obstacles | self.no_fly)

    @property
    def target_cells(self) -> List[Cell]:
        ys, xs = np.nonzero(self.targets)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def summary(self) -> Dict[str, float]:
        """Basic statistics, used by the demo."""
        n = self.width * self.height
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "obstacle_cells": int(self.obstacles.sum()),
            "no_fly_cells": int(self.no_fly.sum()),
            "flyable_cells": int(self.flyable_mask.sum()),
            "flyable_pct": 100.0 * float(self.flyable_mask.sum()) / n,
            "hazard_cells": int((self.hazards > 0).sum()),
            "max_hazard": float(self.hazards.max()),
            "priority_cells": int((self.priority > 0).sum()),
            "targets": int(self.targets.sum()),
            "bases": len(self.bases),
        }

    def _check(self, x: int, y: int) -> None:
        if not self.in_bounds(x, y):
            raise IndexError(f"cell ({x}, {y}) is outside the {self.width}x{self.height} grid")

    # ------------------------------------------------------------------
    # Scenario generation (runs once, from the seeded RNG)
    # ------------------------------------------------------------------
    def _generate(self) -> None:
        cfg = self.config
        self.bases = self._base_positions()
        keepout = self._keepout_mask(self.bases, cfg.base_keepout)

        self._place_rectangles(self.obstacles, cfg.n_obstacles, (2, 7), keepout)
        self._place_rectangles(self.no_fly, cfg.n_no_fly_zones, (4, 9), keepout | self.obstacles)

        free = self.flyable_mask
        self.hazards = self._gaussian_layer(cfg.n_hazard_zones, (2.0, 5.0), free & ~keepout, 0.15)
        self.priority = self._gaussian_layer(cfg.n_priority_zones, (3.0, 6.0), free, 0.25)
        self._place_targets(free & ~keepout)

    def _base_positions(self) -> List[Cell]:
        """Evenly spaced along the bottom edge; fixed so bases are deterministic."""
        n = self.config.n_bases
        y = 1
        return [(int(round((i + 1) * self.width / (n + 1))), y) for i in range(n)]

    def _keepout_mask(self, cells: List[Cell], radius: int) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        for x, y in cells:
            mask[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1] = True
        return mask

    def _place_rectangles(self, layer: np.ndarray, count: int,
                          size_range: Tuple[int, int], forbidden: np.ndarray) -> None:
        """Place up to `count` non-overlapping rectangles that avoid `forbidden` cells."""
        lo, hi = size_range
        placed = 0
        for _ in range(200 * max(count, 1)):
            if placed == count:
                break
            w = int(self._rng.integers(lo, hi + 1))
            h = int(self._rng.integers(lo, hi + 1))
            x0 = int(self._rng.integers(0, self.width - w + 1))
            y0 = int(self._rng.integers(0, self.height - h + 1))
            region = (slice(y0, y0 + h), slice(x0, x0 + w))
            if forbidden[region].any() or layer[region].any():
                continue
            layer[region] = True
            placed += 1

    def _gaussian_layer(self, count: int, sigma_range: Tuple[float, float],
                        allowed: np.ndarray, cutoff: float) -> np.ndarray:
        """Sum of Gaussian blobs, scaled to peak at <= 1, zero outside `allowed`."""
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        layer = np.zeros((self.height, self.width), dtype=float)
        for _ in range(count):
            cx = self._rng.uniform(0, self.width - 1)
            cy = self._rng.uniform(0, self.height - 1)
            sigma = self._rng.uniform(*sigma_range)
            peak = self._rng.uniform(0.5, 1.0)
            layer = np.maximum(layer, peak * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
        layer = np.round(layer, 2)
        layer[layer < cutoff] = 0.0
        layer[~allowed] = 0.0
        return layer

    def _place_targets(self, allowed: np.ndarray) -> None:
        """~70% of targets in priority zones, the rest anywhere allowed."""
        n = self.config.n_targets
        n_priority = int(round(0.7 * n))
        for pool, goal in ((np.argwhere(allowed & (self.priority > 0)), n_priority),
                           (np.argwhere(allowed), n)):
            if len(pool) == 0:
                continue
            while self.targets.sum() < goal:
                y, x = pool[int(self._rng.integers(len(pool)))]
                self.targets[y, x] = True
