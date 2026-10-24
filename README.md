# Multi-Objective UAV Strategy Engine (Genetic Algorithm)

University AI Assignment 2. A Genetic Algorithm (GA) optimises, for three UAVs over
a simulated 50x50 disaster grid, the deployment position, waypoint route, segment
speeds and sensor modes (OFF/LOW/HIGH). It maximises high-priority coverage and
target detection and minimises energy, flight time and hazard exposure, under hard
constraints (obstacles, no-fly zones, flight range). It is compared with random
search and studied with hyperparameter-sensitivity experiments.

Report: [reports/technical_report.pdf](reports/technical_report.pdf) (4 pages; source
[reports/technical_report.md](reports/technical_report.md)). Design details and
assumptions: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Assignment objectives and where they are implemented

| Requirement | Where |
|---|---|
| Discretised grid with obstacles, no-fly zones, hazards, priorities, targets, 3 bases | `grid.py`, `config.py` |
| Hybrid chromosome (5 waypoints, 6 segment speeds, sensor modes, return to base) | `chromosome.py`, `population.py` |
| Five-objective fitness + hard constraints at segment level | `fitness.py`, `constraints.py` |
| Custom selection, crossover, mutation, elitism, termination | `operators.py`, `genetic_algorithm.py` |
| Convergence plot, route maps, coverage heatmaps | `visualization.py`, `run_ga.py` |
| Hyperparameter sensitivity (population, mutation, selection pressure) | `experiments.py`, `run_experiments.py` |
| Fair comparison with random search | `random_search.py`, `run_experiments.py` |
| 3-5 page technical report | `reports/` |

All weights, parameters and physical models are our own engineering assumptions,
not instructor-prescribed values. Simulated target detection is a geometric proxy,
not computer vision. Mutation probability is **per gene**, not per chromosome.

## Setup (Linux/macOS)

Python 3.9+ is required. Dependencies: NumPy, Matplotlib, pytest (`requirements.txt`).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Only to rebuild the PDF report: `pip install -r requirements-report.txt` and have
Google Chrome and `pdfinfo` (poppler) installed.

## Commands (run from the project root, venv active)

```bash
python -m uav_ga.demo_grid       # Step 2: grid statistics + outputs/grid_preview.png
python -m uav_ga.demo_step3      # Step 3: 30 random chromosomes evaluated (no evolution)
python -m uav_ga.run_ga          # Step 4: full GA, seed 42, 60 generations (~15 s)
python -m uav_ga.run_experiments # Step 5: all experiments (80 runs, ~6 min with 8 workers)
python -m uav_ga.run_experiments --replot   # rebuild tables/figures from saved CSV (seconds)
python reports/build_report.py   # rebuild technical_report.md/.pdf from saved results
pytest -q                        # full test suite (about one minute)
```

Useful options: `run_ga --seed 7 --generations 80 --patience 15 --crossover waypoint`;
`run_experiments --seeds 11 22 33 --budget 600 --workers 4 --output-dir outputs/test_run`
(a reduced run for a quick check, written to a separate directory).

## What the experiments do

- Environment fixed (grid seed 42). One factor at a time (others default 30 / 0.15 / 3):
  population {20, 30, 50}, mutation {0.05, 0.15, 0.30}, tournament {2, 3, 5}.
- 10 optimiser seeds (11, 22, 33, 44, 55, 66, 77, 88, 99, 110) per configuration.
- Every GA and random-search run uses exactly 1710 fitness evaluations
  (`GAConfig.max_evaluations`), so comparisons are budget-fair.
- Statistics: mean, sample SD, paired same-seed differences, exact sign-flip p-values.
- Random search draws independent chromosomes with the same sampler, fitness
  evaluator and constraints, with no selection, crossover or mutation.

Results are in `outputs/experiments/`. Interpretation is in the report; the
numbers there are generated from these files.

## Output files

| File | Content |
|---|---|
| `outputs/grid_preview.png` | the disaster grid (all layers, targets, bases) |
| `outputs/fitness_convergence.png` | generation best / average / best-so-far fitness (seed-42 run) |
| `outputs/evolved_uav_routes.png` | best evolved routes with feasibility status |
| `outputs/route_snapshots.png` | best-so-far routes at generation 0, middle, final |
| `outputs/coverage_initial/middle/final.png` | sensor-coverage heatmaps (fixed 0-1 scale) |
| `outputs/ga_run_summary.json` | configuration, seeds, history, initial/final evaluation |
| `outputs/experiments/raw_runs.csv` | one row per run (every metric, seed, evaluations, generations) |
| `outputs/experiments/progress_curves.json` | best-so-far fitness vs evaluations per run |
| `outputs/experiments/summary_sensitivity.csv`, `summary_comparison.csv` | means, SDs, paired tests |
| `outputs/experiments/sensitivity_*.png`, `ga_vs_random_*.png` | experiment figures |
| `reports/technical_report.pdf` / `.md` | the report |

## Reproducibility

The environment seed (42), each optimiser seed, and every configuration are stored in
`outputs/ga_run_summary.json` and `outputs/experiments/experiment_config.json`.
The same seed gives the same result; experiment results do not depend on the number of
worker processes. Regenerating the experiments overwrites `outputs/experiments/`.

## Grid conventions

A cell is `(x, y)`: x = column, y = row, origin bottom-left in plots; layers are
`(height, width)` arrays indexed `layer[y, x]`: `obstacles`, `no_fly` (blocked),
`hazards` (0-1, traversable), `priority` (0-1), `targets`. Routes are checked with an
exact segment traversal and a conservative corner policy (touching a blocked cell's
corner is a collision). Every UAV flies base -> wp1..wp5 -> base (six segments).

## Project layout

```
uav-genetic-algorithm/
├── README.md  requirements.txt  requirements-report.txt  pyproject.toml  .gitignore
├── docs/IMPLEMENTATION_PLAN.md
├── src/uav_ga/
│   ├── config.py  grid.py  chromosome.py  population.py  constraints.py  fitness.py
│   ├── operators.py  genetic_algorithm.py  random_search.py
│   ├── experiments.py  visualization.py
│   └── demo_grid.py  demo_step3.py  run_ga.py  run_experiments.py
├── tests/                    # grid, chromosome, population, constraints, fitness, operators,
│                             # GA, visualization, random search, experiments
├── outputs/                  # figures, run summary, experiments/
└── reports/                  # report template, build script, technical_report.md/.pdf
```
