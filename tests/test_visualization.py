"""Figure generation from real GA data (Step 4)."""

import matplotlib

matplotlib.use("Agg")

from uav_ga.config import GAConfig, GridConfig, UAVConfig
from uav_ga.genetic_algorithm import GeneticAlgorithm
from uav_ga.grid import DisasterGrid
from uav_ga.run_ga import save_all_figures
from uav_ga.visualization import plot_grid

PNG = b"\x89PNG\r\n\x1a\n"
EXPECTED = ["fitness_convergence.png", "evolved_uav_routes.png", "route_snapshots.png",
            "coverage_initial.png", "coverage_middle.png", "coverage_final.png"]


def test_all_step4_figures_are_written(tmp_path):
    grid = DisasterGrid(GridConfig(seed=42))
    ga = GeneticAlgorithm(grid, config=GAConfig(population_size=8, max_generations=4))
    result = ga.run()
    paths = save_all_figures(grid, ga.evaluator, result, tmp_path / "out")
    assert len(paths) == 6
    for name in EXPECTED:
        f = tmp_path / "out" / name
        assert f.exists() and f.stat().st_size > 5000
        assert f.read_bytes()[:8] == PNG


def test_middle_generation_definition():
    grid = DisasterGrid(GridConfig(seed=42))
    r = GeneticAlgorithm(grid, config=GAConfig(population_size=6, max_generations=5)).run()
    assert r.middle_generation == 2


def test_plot_grid_still_works(tmp_path):
    out = tmp_path / "g.png"
    plot_grid(DisasterGrid(GridConfig(seed=42)), save_path=str(out))
    assert out.read_bytes()[:8] == PNG
