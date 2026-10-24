"""Tests for DisasterGrid (Step 2)."""

import numpy as np
import pytest

from uav_ga.config import GridConfig
from uav_ga.grid import DisasterGrid

LAYERS = ["obstacles", "no_fly", "hazards", "priority", "targets"]


@pytest.fixture(scope="module")
def grid():
    return DisasterGrid(GridConfig(seed=42))


def test_default_dimensions(grid):
    assert (grid.width, grid.height) == (50, 50)
    for name in LAYERS:
        assert getattr(grid, name).shape == (50, 50)


def test_custom_dimensions():
    g = DisasterGrid(GridConfig(width=30, height=20, seed=1))
    assert g.obstacles.shape == (20, 30)  # (height, width)


def test_too_small_grid_rejected():
    with pytest.raises(ValueError):
        DisasterGrid(GridConfig(width=5, height=5))


def test_coordinate_validity(grid):
    assert grid.in_bounds(0, 0) and grid.in_bounds(49, 49)
    assert not grid.in_bounds(-1, 0) and not grid.in_bounds(0, 50)
    assert not grid.in_bounds(50, 10)
    with pytest.raises(IndexError):
        grid.cell_info(50, 0)


def test_xy_convention_matches_arrays(grid):
    x, y = grid.target_cells[0]
    assert grid.targets[y, x] and grid.has_target(x, y)


def test_layers_have_content(grid):
    assert grid.obstacles.any() and grid.no_fly.any()
    assert grid.hazards.any() and grid.priority.any()
    assert grid.targets.sum() == grid.config.n_targets


def test_blocked_cell_detection(grid):
    ys, xs = np.nonzero(grid.obstacles)
    x, y = int(xs[0]), int(ys[0])
    assert grid.is_obstacle(x, y) and grid.is_blocked(x, y) and not grid.is_flyable(x, y)
    assert grid.is_blocked(-1, 0)  # out of bounds counts as blocked


def test_no_fly_restriction(grid):
    ys, xs = np.nonzero(grid.no_fly)
    x, y = int(xs[0]), int(ys[0])
    assert grid.is_no_fly(x, y) and grid.is_blocked(x, y) and not grid.is_flyable(x, y)
    assert not (grid.obstacles & grid.no_fly).any()  # layers do not overlap


def test_hazard_intensity_values(grid):
    assert grid.hazards.min() >= 0.0 and grid.hazards.max() <= 1.0
    assert grid.hazards.max() > 0.0
    assert not (grid.hazards > 0)[grid.obstacles | grid.no_fly].any()


def test_hazards_are_traversable(grid):
    ys, xs = np.nonzero(grid.hazards > 0)
    x, y = int(xs[0]), int(ys[0])
    assert grid.is_flyable(x, y)
    assert grid.hazard_at(x, y) > 0


def test_priority_weights_in_range(grid):
    assert grid.priority.min() >= 0.0 and grid.priority.max() <= 1.0


def test_targets_on_flyable_cells(grid):
    for x, y in grid.target_cells:
        assert grid.is_flyable(x, y)


def test_three_valid_start_positions(grid):
    assert len(grid.bases) == 3
    assert len(set(grid.bases)) == 3
    for x, y in grid.bases:
        assert grid.in_bounds(x, y)
        assert not grid.obstacles[y, x]
        assert not grid.no_fly[y, x]
        assert grid.is_flyable(x, y)


@pytest.mark.parametrize("seed", [0, 1, 7, 123, 2024])
def test_bases_valid_across_seeds(seed):
    g = DisasterGrid(GridConfig(seed=seed))
    assert len(g.bases) == 3 and all(g.is_flyable(x, y) for x, y in g.bases)


def test_same_seed_is_reproducible():
    a = DisasterGrid(GridConfig(seed=7))
    b = DisasterGrid(GridConfig(seed=7))
    for name in LAYERS:
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert a.bases == b.bases


def test_different_seed_changes_scenario():
    a = DisasterGrid(GridConfig(seed=7))
    b = DisasterGrid(GridConfig(seed=8))
    assert not np.array_equal(a.obstacles, b.obstacles)
