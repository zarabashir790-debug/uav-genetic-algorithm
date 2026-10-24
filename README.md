# Multi-Objective UAV Strategy Engine (Genetic Algorithm)

University AI Assignment 2. The goal is to use a Genetic Algorithm (GA) to
optimise UAV deployment positions, waypoint trajectories, speeds and sensor
activation profiles over a simulated disaster grid, trading off coverage and
target detection against energy, flight time and hazard exposure, while
respecting maximum flight range and no-fly zones.

## Status

- **Step 1 (setup):** done.
- **Step 2 (disaster grid):** done. `DisasterGrid`, its plot and a demo.
- **Planned, not implemented:** chromosome, fitness function, operators, GA,
  random-search baseline, route/heatmap/convergence plots, experiments.

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Setup (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows use `.venv\Scripts\activate` instead of the `source` line.

## Verify

```bash
python -c "import uav_ga; print(uav_ga.__version__)"
pytest -v
```

## Run the grid demo (Step 2)

```bash
python -m uav_ga.demo_grid               # seed 42, writes outputs/grid_preview.png
python -m uav_ga.demo_grid --seed 7 --output outputs/grid_seed7.png
```

It builds the 50x50 grid, prints statistics and saves the preview image.

Grid conventions: a cell is `(x, y)` with `x` = column, `y` = row (origin
bottom-left in plots). Layers are separate `(height, width)` NumPy arrays
indexed `layer[y, x]`: `obstacles`, `no_fly`, `hazards` (0-1, traversable),
`priority` (0-1), `targets`. Obstacles and no-fly cells are blocked; hazards are
flyable but carry risk. Three drone bases sit along the bottom edge.

## Layout

```
uav-genetic-algorithm/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docs/IMPLEMENTATION_PLAN.md
├── src/uav_ga/
│   ├── config.py             # GridConfig (GA/UAV/fitness config planned)
│   ├── grid.py               # DisasterGrid (done)
│   ├── chromosome.py         # encoding (planned)
│   ├── fitness.py            # objectives + constraints (planned)
│   ├── operators.py          # selection/crossover/mutation/repair (planned)
│   ├── genetic_algorithm.py  # GA loop (planned)
│   ├── random_search.py      # baseline (planned)
│   ├── visualization.py      # plot_grid done; other plots planned
│   ├── demo_grid.py          # Step 2 demo
│   └── experiments.py        # sensitivity + comparison (planned)
├── tests/                    # test_package.py, test_grid.py
├── outputs/                  # generated figures/results (git-ignored)
└── reports/                  # report material
```
