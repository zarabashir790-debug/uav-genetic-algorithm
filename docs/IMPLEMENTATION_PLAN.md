# Implementation Plan

Everything below is a **plan**; nothing except the empty package skeleton is
implemented yet. Details may be refined as each step is built.

## 1. Assignment objectives

Build an object-oriented Python system that uses a GA to optimise, over a
simulated disaster grid:

- UAV deployment (start) positions
- waypoint trajectories
- speed adjustments
- sensor activation profiles

Maximise high-priority area coverage and target detection; minimise energy
consumption, flight time and hazard exposure. Respect maximum flight range and
no-fly zones. Use custom chromosome encoding and genetic operators. Produce
fitness convergence plots, UAV route maps and coverage heatmaps. Later:
hyperparameter sensitivity experiments and a fair comparison with random search.

## 2. Architecture (src/uav_ga)

| Module | Planned responsibility |
|---|---|
| config.py | Dataclasses: grid size, UAV count/limits, fitness weights, GA hyperparameters, seed |
| grid.py | `DisasterGrid`: priority map, target locations, hazard map, no-fly mask; cell/coordinate helpers |
| chromosome.py | `Chromosome`: encoding, random initialisation, decoding into per-UAV plans |
| fitness.py | `FitnessEvaluator`: simulates a plan, returns objective values, penalties, scalar fitness |
| operators.py | Selection, crossover, mutation, repair (all custom, gene-aware) |
| genetic_algorithm.py | `GeneticAlgorithm`: population loop, elitism, history recording |
| random_search.py | `RandomSearch`: baseline using the same evaluator and budget |
| visualization.py | Convergence, route map, coverage heatmap plotting |
| experiments.py | Sensitivity sweeps and GA vs random-search runs over multiple seeds |

## 3. Chromosome representation

Per UAV, a block of genes:

- deployment position `(x, y)` on the grid
- `W` waypoints, each `(x, y)` (ordered)
- a speed level per leg (continuous within `[v_min, v_max]` or a discrete level)
- a sensor on/off flag (or activation level) per waypoint

The chromosome is the concatenation of all UAV blocks, stored as NumPy arrays
(one per gene type) so operators can treat each gene group appropriately.

## 4. GA workflow

1. Initialise a random population (seeded RNG).
2. Decode and evaluate each individual.
3. Select parents (tournament).
4. Crossover (UAV-block and waypoint-segment aware).
5. Mutate (Gaussian position jitter, speed perturbation, sensor bit flip, waypoint swap).
6. Repair constraint violations where cheap.
7. Elitism, form next generation, record best/mean fitness.
8. Stop at max generations (optionally on stagnation).

## 5. Fitness objectives

Maximise: weighted priority-area coverage; target detection (targets inside an
active sensor footprint).
Minimise: energy (function of distance and speed); flight time; hazard exposure
(hazard accumulated along the route).

Initial approach: weighted-sum scalarisation with normalised terms. Whether to
also report a Pareto view is an open decision for a later step.

## 6. Constraint handling

- **Maximum flight range:** route length over the limit is penalised in
  proportion to the excess; repair may truncate trailing waypoints.
- **No-fly zones:** waypoints or segments entering a zone are penalised;
  repair may move a waypoint to the nearest allowed cell.
- Penalties are scaled so infeasible solutions rank below feasible ones.

## 7. Visualisations

- Fitness convergence (best and mean per generation, multiple runs)
- UAV route map over the grid with no-fly zones, hazards, targets
- Coverage heatmap (visit/sensing frequency per cell)

## 8. Experiments

- Hyperparameter sensitivity: population size, mutation rate, crossover rate,
  tournament size, elitism; several seeds each, report mean and spread.
- GA vs random search: same evaluator, same number of fitness evaluations,
  same seeds, same scenarios.

## 9. Report deliverables

Problem description, encoding and operator design, fitness/constraint
definitions, parameter settings, convergence plots, route maps and heatmaps,
sensitivity results, random-search comparison, discussion and limitations.

## 10. Step roadmap

1. Project setup (this step)
2. Grid simulation and config
3. Chromosome
4. Fitness and constraints
5. Operators and GA
6. Visualisation
7. Random search and experiments
8. Report
