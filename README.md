# Multi-Objective UAV Strategy Engine (Genetic Algorithm)

University AI Assignment 2. The goal is to use a Genetic Algorithm (GA) to
optimise UAV deployment positions, waypoint trajectories, speeds and sensor
activation profiles over a simulated disaster grid, trading off coverage and
target detection against energy, flight time and hazard exposure, while
respecting maximum flight range and no-fly zones.

## Status

**Step 1 (project setup) only.** The package skeleton, docs and tooling exist.
The grid simulation, chromosome, fitness function, operators, GA, random-search
baseline, plots and experiments are **planned, not implemented**. See
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

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

## Layout

```
uav-genetic-algorithm/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docs/IMPLEMENTATION_PLAN.md
├── src/uav_ga/
│   ├── config.py             # parameters (planned)
│   ├── grid.py               # disaster grid (planned)
│   ├── chromosome.py         # encoding (planned)
│   ├── fitness.py            # objectives + constraints (planned)
│   ├── operators.py          # selection/crossover/mutation/repair (planned)
│   ├── genetic_algorithm.py  # GA loop (planned)
│   ├── random_search.py      # baseline (planned)
│   ├── visualization.py      # plots (planned)
│   └── experiments.py        # sensitivity + comparison (planned)
├── tests/test_package.py
├── outputs/                  # generated figures/results (git-ignored)
└── reports/                  # report material
```
