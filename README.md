# Multi-Objective UAV Strategy Engine (Genetic Algorithm)

University AI Assignment 2. The goal is to use a Genetic Algorithm (GA) to
optimise UAV deployment positions, waypoint trajectories, speeds and sensor
activation profiles over a simulated disaster grid, trading off coverage and
target detection against energy, flight time and hazard exposure, while
respecting maximum flight range and no-fly zones.

## Status

- **Step 1 (setup):** done.
- **Step 2 (disaster grid):** done.
- **Step 3 (chromosome, population, fitness, constraints):** done.
- **Step 4 (Genetic Algorithm and visualisations):** done.
- **Planned, not implemented (Step 5):** hyperparameter sensitivity experiments,
  random-search comparison, report material.

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the design,
formulas and assumptions.

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

## Run the Step 3 demo

```bash
python -m uav_ga.demo_step3
```

Loads the seed-42 grid, generates 30 chromosomes, prints one, evaluates all,
prints best/average/worst fitness and the best chromosome's breakdown and
feasibility. No evolution happens here.

## Step 3 summary

- **Chromosome:** 3 `DronePlan`s (one per base). Each has 5 integer (x, y)
  waypoints, 6 float segment speeds and 6 sensor modes (OFF/LOW/HIGH).
  Route: `base -> wp1..wp5 -> base` (drones **return to base**).
  Deployment position = first waypoint.
- **Population:** `PopulationGenerator`, default 30, seeded and reproducible,
  waypoints on flyable cells, no duplicates.
- **Fitness:** weighted sum of normalised coverage, detection, energy, time and
  hazard (`raw` in [0, 1]); defaults 0.30/0.25/0.15/0.10/0.20 (design choices).
- **Constraints:** every flight segment is traversed cell by cell. Routes through
  obstacles, no-fly cells or out of bounds, or longer than the range limit, are
  infeasible and score below -1, always below any feasible plan.

Use in code:

```python
from uav_ga.config import GridConfig
from uav_ga.grid import DisasterGrid
from uav_ga.population import PopulationGenerator
from uav_ga.fitness import FitnessEvaluator

grid = DisasterGrid(GridConfig(seed=42))
population = PopulationGenerator(grid).generate()
breakdown = FitnessEvaluator(grid).evaluate(population[0])
print(breakdown.fitness, breakdown.feasible)
```

## Run the Genetic Algorithm (Step 4)

```bash
python -m uav_ga.run_ga                     # seed 42, writes everything to outputs/
python -m uav_ga.run_ga --seed 7 --generations 80 --patience 15 --crossover waypoint
```

It runs the GA, prints progress, initial/final statistics, the best chromosome,
its full evaluation and feasibility, the generation count and stopping reason,
and saves: `fitness_convergence.png`, `evolved_uav_routes.png`,
`route_snapshots.png`, `coverage_initial.png`, `coverage_middle.png`,
`coverage_final.png` and `ga_run_summary.json` (config, seeds, history).

How it works (details and formulas in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)):
- **Selection:** tournament (default size 3); feasible always beats infeasible.
- **Crossover:** drone crossover swaps whole drone plans; waypoint crossover
  swaps a consecutive waypoint section with its speeds/sensors. Configurable.
- **Mutation:** per-gene (default 0.15): waypoint move/replacement, speed change,
  sensor switch; always on valid cells and within bounds.
- **Elitism:** 2 best chromosomes survive each generation.
- **Termination:** max generations (default 60) or optional early stopping.
- **Constraints:** every offspring is validated at route level; failed
  regeneration (max 10 attempts) falls back to the unchanged parent. The corner
  policy is conservative: a route touching a blocked cell's corner, or squeezing
  diagonally between touching blocked cells, is infeasible.
- Defaults are experimental choices, not instructor-prescribed values.

Interpreting convergence: *generation best* is the best of one generation;
*best so far* is the best of all generations to date and never decreases. A
single seed is only one sample; do not read it as a general result.

```bash
pytest -q          # full suite
```

## Layout

```
uav-genetic-algorithm/
├── README.md  requirements.txt  pyproject.toml  .gitignore
├── docs/IMPLEMENTATION_PLAN.md
├── src/uav_ga/
│   ├── config.py             # Grid/UAV/Sensor/Fitness/Population/GA configs
│   ├── grid.py               # DisasterGrid + exact, conservative segment traversal
│   ├── chromosome.py         # SensorMode, DronePlan, Chromosome
│   ├── population.py         # PopulationGenerator
│   ├── constraints.py        # ConstraintChecker (segment-level)
│   ├── fitness.py            # FitnessEvaluator, FitnessBreakdown
│   ├── operators.py          # selection, crossover, mutation
│   ├── genetic_algorithm.py  # GeneticAlgorithm, GAResult
│   ├── visualization.py      # grid, convergence, routes, coverage plots
│   ├── demo_grid.py          # Step 2 demo
│   ├── demo_step3.py         # Step 3 demo (population + fitness only)
│   ├── run_ga.py             # Step 4 runnable simulation
│   ├── random_search.py      # planned (Step 5)
│   └── experiments.py        # planned (Step 5)
├── tests/                    # grid, chromosome, population, constraints, fitness,
│                             # operators, genetic algorithm, visualization
├── outputs/                  # generated figures and run summary
└── reports/
```
