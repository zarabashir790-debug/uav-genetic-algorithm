# Implementation Plan

Status legend: **[done]** implemented and tested, **[planned]** not implemented yet.

## 1. Assignment objectives

Object-oriented Python system using a GA to optimise, over a simulated disaster
grid: UAV deployment positions, waypoint trajectories, speed adjustments and
sensor activation profiles. Maximise high-priority coverage and target
detection; minimise energy, flight time and hazard exposure; respect maximum
flight range and no-fly zones. Produce convergence plots, route maps and
coverage heatmaps; later run hyperparameter sensitivity experiments and a fair
comparison with random search.

## 2. Roadmap

| Step | Content | Status |
|---|---|---|
| 1 | Project setup | done |
| 2 | 50x50 disaster grid, layers, bases, preview plot | done |
| 3 | Chromosome, population init, fitness, hard constraints | done |
| 4 | GA operators and evolution loop, convergence plot, route map, heatmap | planned |
| 5 | Sensitivity experiments, random-search comparison, report material | planned |

## 3. Architecture (src/uav_ga)

| Module | Responsibility | Status |
|---|---|---|
| config.py | `GridConfig`, `UAVConfig`, `SensorSpec`, `FitnessConfig`, `PopulationConfig` (GA hyperparameters come in Step 4) | done |
| grid.py | `DisasterGrid` (layers, queries) and exact `segment_cells` traversal | done |
| chromosome.py | `SensorMode`, `DronePlan`, `Chromosome` | done |
| population.py | `PopulationGenerator` | done |
| constraints.py | `ConstraintChecker`, `DroneConstraintReport` | done |
| fitness.py | `FitnessEvaluator`, `FitnessBreakdown`, `DroneMetrics` | done |
| visualization.py | `plot_grid` done; convergence, route map, heatmap planned | partial |
| operators.py, genetic_algorithm.py | selection, crossover, mutation, GA loop | planned |
| random_search.py, experiments.py | baseline and experiments | planned |

## 4. Disaster grid (Step 2)

50x50, coordinate `(x, y)` = (column, row), layers indexed `layer[y, x]`.
Separate arrays: `obstacles` (blocked), `no_fly` (restricted), `hazards`
(0-1, traversable but risky), `priority` (0-1 coverage weight), `targets`.
Obstacles and no-fly zones are blocked; hazards are not. Bases are fixed at
(12,1), (25,1), (38,1) with a keep-out so they are always valid. Seed 42 gives a
reproducible scenario.

## 5. Chromosome encoding (Step 3)

One `Chromosome` = three `DronePlan`s (drone *i* uses base *i*). Hybrid encoding
per drone (`n = 5` waypoints):

| Gene | Type | Size | Meaning |
|---|---|---|---|
| waypoints | int (x, y) | 5 x 2 | ordered operational waypoints |
| speeds | float | 6 | speed of each flight segment, in `[speed_min, speed_max]` |
| sensors | int enum | 6 | OFF / LOW / HIGH for each segment |
| base | fixed | 1 | grid base, never mutated |

**Route model (applied consistently to distance, time, energy, sensing and
constraints):** the drone flies `base -> wp1 -> ... -> wp5 -> base`, i.e.
**it returns to its base**, giving 6 segments. Waypoint order defines the route.
Speed and sensor genes belong to segments, so the return leg has its own speed
and sensor mode. The *deployment position* is derived as `wp1`, where the
mission starts. Straight-line flight between waypoints is assumed.

## 6. Population initialisation (Step 3)

`PopulationGenerator` (default size 30, seeded `numpy` RNG, same seed gives the
same population): three drones per chromosome, waypoints sampled uniformly from
flyable cells, speeds uniform in bounds, sensor modes uniform over OFF/LOW/HIGH.
Waypoints are chosen so each segment from the previous point is unblocked, and a
whole drone plan is re-drawn (up to 200 tries) until its closed loop is feasible
(including range). Duplicates are rejected; `diversity()` reports the mean
pairwise waypoint distance. If the retries fail the last draw is kept, so
feasibility is **not guaranteed** and the fitness function penalises such
individuals. Validation: `Chromosome.validate(grid, uav)` checks structure,
bases, speed/sensor ranges and waypoint cells (waypoints only; it does not
prove routes are safe).

## 7. Fitness function (Step 3)

All quantities use abstract units: distance in cells, time in tu, speed in
cells/tu, energy in eu, power in eu/tu. These are design assumptions, not
calibrated to a real aircraft.

### Sensor model

| Mode | Footprint radius | Coverage effectiveness | Power |
|---|---|---|---|
| OFF | none | 0 | 0 |
| LOW | 2 cells | 0.5 | 0.3 eu/tu |
| HIGH | 4 cells | 1.0 | 1.0 eu/tu |

While a segment's sensor is on, a disc footprint is placed on every grid cell
that segment traverses (exact traversal). Cell coverage = **maximum**
effectiveness over all footprints of all drones, so cells are never counted
twice. A target is *detected* if its cell has coverage > 0. This is a simulated
geometric proxy, **not** computer-vision recognition. The sensor runs for the
whole segment, including the return leg. Sensing is not blocked by obstacles.

### Physical models

For segment *i* with length `L_i`, speed `v_i`, sensor mode `m_i`:

```
t_i = L_i / v_i
P_i = P_hover + k_drag * v_i^3 + P_sensor(m_i)
e_i = P_i * t_i
```
Defaults: `P_hover = 1.0`, `k_drag = 0.02`. Slow flight wastes hover energy and
fast flight pays cubic drag, so energy per cell is U-shaped with a minimum near
`v = 2.9`, while time always prefers speed: a real speed trade-off for the GA.
Totals are summed over all segments and drones (including return legs). Flight
time is the **sum** of drone flight times (fleet flight time), not the mission
duration. Hazard exposure = sum over traversed cells of
`(length flown inside the cell) * hazard intensity`, measured along segments,
not only at waypoints.

### Objectives and normalisation

```
C = sum(priority * coverage) / sum(priority)        in [0, 1]
D = detected targets / total targets                in [0, 1]
E = min(1, total energy / E_ref)
T = min(1, total flight time / T_ref)
H = min(1, total hazard exposure / H_ref)
E_ref = n_drones * max_range * max_v[(P_hover + k v^3 + P_high) / v]   (v = speed_min or speed_max)
T_ref = n_drones * max_range / speed_min
H_ref = 0.25 * n_drones * max_range * 1.0
```
The references are "worst-case-ish" scales chosen so normalised values spread
over a useful part of [0, 1]; `H_ref` is deliberately smaller than the worst case
because hazards cover only ~14% of the grid.

### Scalar fitness (maximised)

```
raw = wC*C + wD*D + wE*(1-E) + wT*(1-T) + wH*(1-H),   sum(w) = 1,   raw in [0, 1]
fitness = raw                                          if feasible
fitness = -(1.0 + 0.05*blocked_cells + 1.0*sum(range_excess/max_range))   if infeasible
```

### Default weights and rationale (design choices, not prescribed)

| Weight | Value | Reason |
|---|---|---|
| coverage `wC` | 0.30 | primary mission goal |
| detection `wD` | 0.25 | second mission goal; targets are few, so each matters |
| hazard `wH` | 0.20 | safety: strongest of the cost terms |
| energy `wE` | 0.15 | battery limits endurance |
| time `wT` | 0.10 | lowest: range limits already bound time |

Mission benefit totals 0.55 against 0.45 for costs so that sensing is worth its
energy but the GA cannot win by not flying. Weights are in `FitnessConfig` for
sensitivity studies in Step 5. Because of the `1 - cost` form, a plan that
does nothing useful still scores about 0.4-0.45; this is a known property of
weighted-sum scalarisation.

## 8. Hard-constraint handling (Step 3)

Constraints: (1) max range per drone (full loop including return, default 140
cells), (2) obstacles, (3) no-fly zones, (4) out-of-bounds.

**Segment checking:** `DisasterGrid.segment_cells` computes the exact cells each
segment passes through (parametric crossings of cell boundaries), so a route
that jumps over a blocked region with free endpoints is still detected. It also
reports out-of-bounds cells. Assumption: a segment that only touches a cell
corner does not enter that cell.

**Penalty approach (no repair):** any violation makes the plan infeasible
(`FitnessBreakdown.feasible = False`). Since `raw <= 1 <= infeasible_base`,
every infeasible plan scores `< -1`, below every feasible plan (`>= 0`),
regardless of coverage. More blocked cells or range excess give a lower score, so
the GA keeps a gradient toward feasibility. `FitnessConfig` rejects
`infeasible_base < 1`. No repair operator exists, so no unverified "safe" route
is ever invented. Hazard cells never cause infeasibility.

## 9. GA workflow (Step 4, planned)

Tournament selection, block-aware crossover (per-drone, per-segment), mutation
(waypoint jitter, speed perturbation, sensor change, waypoint swap), elitism,
generation history, optional stagnation stop. Fitness is the scalar defined above.

## 10. Visualisations (planned)

Convergence curves, UAV route map over the grid, coverage heatmap.
(`plot_grid` and `FitnessEvaluator.coverage_map` already exist as building blocks.)

## 11. Experiments (planned)

Sensitivity to population size, mutation/crossover rate, tournament size,
elitism over several seeds. GA vs random search with the same evaluator,
evaluation budget, seeds and scenario.

## 12. Report deliverables (planned)

Problem, encoding, operators, fitness and constraints (Sections 5-8 are written
to be reusable), parameters, plots, sensitivity, random-search comparison,
limitations.

## 13. Assumptions and limitations (for the report)

- Abstract units; physics (hover + cubic drag) is a simplified model.
- Straight-line segments; no turning-rate or altitude model.
- Detection is a geometric proxy; sensing ignores obstacles/occlusion.
- Footprints are centred on traversed cell centres (error under one cell).
- Drones are assumed not to collide with each other.
- Total fleet time (sum) is used, not makespan.
- All weights and normalisation references are design choices.
- The weighted sum is one scalarisation; no Pareto front is computed.
