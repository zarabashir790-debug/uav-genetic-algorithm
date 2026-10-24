"""Step 2 demo: build the disaster grid, print statistics, save a preview image.

Run from the project root:  python -m uav_ga.demo_grid [--seed N] [--output PATH]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed

from uav_ga.config import GridConfig  # noqa: E402
from uav_ga.grid import DisasterGrid  # noqa: E402
from uav_ga.visualization import plot_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and preview the disaster grid.")
    parser.add_argument("--seed", type=int, default=GridConfig.seed)
    parser.add_argument("--output", default="outputs/grid_preview.png")
    args = parser.parse_args()

    grid = DisasterGrid(GridConfig(seed=args.seed))
    print("Disaster grid statistics")
    for key, value in grid.summary().items():
        print(f"  {key:>14}: {value:.1f}" if isinstance(value, float) else f"  {key:>14}: {value}")
    print(f"  {'base cells':>14}: {grid.bases}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_grid(grid, save_path=args.output)
    print(f"Saved preview to {args.output}")


if __name__ == "__main__":
    main()
