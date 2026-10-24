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
| 4 | GA operators and evolution loop, convergence plot, route maps, coverage heatmaps | done |
| 5 | Sensitivity experiments, random-search comparison, technical report | done |

## 3. Architecture (src/uav_ga)

| Module | Responsibility | Status |
|---|---|---|
| config.py | `GridConfig`, `UAVConfig`, `SensorSpec`, `FitnessConfig`, `PopulationConfig`, `GAConfig` | done |
| grid.py | `DisasterGrid` (layers, queries) and exact `segment_cells` traversal | done |
| chromosome.py | `SensorMode`, `DronePlan`, `Chromosome` | done |
| population.py | `PopulationGenerator` | done |
| constraints.py | `ConstraintChecker`, `DroneConstraintReport` | done |
| fitness.py | `FitnessEvaluator`, `FitnessBreakdown`, `DroneMetrics` | done |
| operators.py | tournament selection, drone/waypoint crossover, `Mutator` | done |
| genetic_algorithm.py | `GeneticAlgorithm`, `GAResult` | done |
| visualization.py | `plot_grid`, `plot_convergence`, `plot_routes`, `plot_route_snapshots`, `plot_coverage` | done |
| run_ga.py | runnable Step 4 simulation, saves figures and `ga_run_summary.json` | done |
| random_search.py | `RandomSearch` baseline (same sampler, evaluator, constraints) | done |
| experiments.py, run_experiments.py | experiment framework, statistics, CLI, figures | done |
| reports/ | `report_template.md` + `build_report.py` produce `technical_report.md/.pdf` from saved results | done |

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
reports out-of-bounds cells.

**Conservative corner policy (Step 4 fix):** Step 3 let a segment that touches
only a cell corner pass, so a diagonal route could squeeze between two
diagonally touching blocked cells. Now, if a segment passes exactly through a
shared corner of four cells, **all four count as touched** (the two grazed cells
are returned with length 0). Consequences: a route can never slip between
touching blocked cells, and grazing the corner of even a single blocked cell is a
collision. Zero-length cells add no distance, hazard or sensing. The effect on
Step 3 results: the same seed-42 initial population now has average fitness
0.6516 (was 0.6550) because the generator's segment check is stricter; the best
(0.7468) is unchanged. Fitness weights and normalisation were not changed.

**Penalty approach (no repair):** any violation makes the plan infeasible
(`FitnessBreakdown.feasible = False`). Since `raw <= 1 <= infeasible_base`,
every infeasible plan scores `< -1`, below every feasible plan (`>= 0`),
regardless of coverage. More blocked cells or range excess give a lower score, so
the GA keeps a gradient toward feasibility. `FitnessConfig` rejects
`infeasible_base < 1`. No repair operator exists, so no unverified "safe" route
is ever invented. Hazard cells never cause infeasibility. In Step 4 the GA adds a
bounded regeneration guard around crossover and mutation (Section 9).

## 9. Genetic Algorithm (Step 4)

Fitness is the scalar from Section 7, maximised. Defaults are experimental
choices, not instructor-prescribed values (`GAConfig`):

| Parameter | Default | Meaning |
|---|---|---|
| population_size | 30 | |
| max_generations | 60 | |
| crossover_prob | 0.8 | per selected parent pair |
| mutation_prob | 0.15 | **per gene**: every waypoint, speed and sensor gene mutates independently |
| tournament_size | 3 | selection pressure (to be studied in Step 5) |
| elite_count | 2 | best chromosomes copied unchanged |
| seed | 42 | seeds the initial population and every GA random choice |
| crossover_method | "mixed" | "drone", "waypoint" or "mixed" (random choice per pair) |
| early_stopping_patience | None | off by default; else stop after N generations without improvement |
| min_improvement | 1e-6 | smallest best-so-far gain that counts as improvement |
| waypoint_sigma / waypoint_replace_prob | 3.0 / 0.1 | spatial mutation step (cells) / chance of random replacement |
| speed_sigma_fraction | 0.15 | speed mutation std-dev as a fraction of the speed range |
| max_attempts | 10 | bounded regeneration attempts per crossover/mutation |

**Generation cycle:** evaluate; copy the elites; fill the rest by tournament
selection of two parents, crossover (else copies), mutation, evaluation;
generational replacement; record statistics; stop at `max_generations` or on
early stopping. The stopping reason is reported.

**Selection:** tournament of `tournament_size` random contestants (without
replacement). Order key is `(feasible, fitness)`, so feasible always beats
infeasible and negative fitness is handled. Selection never modifies parents.

**Crossover** (suited to the structure: per-drone blocks, ordered waypoints with
arrival-segment speed/sensor genes):
- *Drone crossover:* swap k (1 to n-1) complete drone plans between parents.
  Plan i always belongs to base i, so base assignment is preserved, and each
  swapped route keeps its own speeds and sensors.
- *Waypoint crossover:* for each chosen drone, swap one consecutive section of
  waypoints `i..j` **together with the speeds and sensors of the segments
  arriving at them**, so the three arrays stay aligned. The return leg and base
  are never swapped.

**Mutation** (each gene with probability `mutation_prob`): spatial waypoint
move (rounded Gaussian step, retried up to 10 times to land on a flyable
in-bounds cell) or, with `waypoint_replace_prob`, replacement by a random
flyable cell; Gaussian speed change clipped to the bounds; sensor switch to one
of the two other modes. Array sizes, bases and value ranges never change.

**Constraint handling in reproduction:** a result is accepted only if
`Chromosome.validate` passes and every route is feasible according to
`ConstraintChecker` (all segments including return leg, range), or if the
parent(s) were already infeasible. Otherwise the operator is retried (bounded by
`max_attempts`); if all attempts fail, an unchanged copy of the parent is used
and the fallback is counted (`GAResult.crossover_fallbacks/mutation_fallbacks`).
Every individual is evaluated by `FitnessEvaluator`, which re-checks routes
independently, so a failed regeneration can never be classified feasible, and
infeasible chromosomes always rank below feasible ones.

**Elitism and termination:** the best `elite_count` survive each generation, so
the generation-best never drops. *Generation-best* is the best fitness inside one
generation; *best-so-far* is the best over all generations up to that point and
never decreases. Both are recorded, together with average, worst, feasible count
and diversity.

**Diversity measure:** mean pairwise distance between chromosomes, where the
distance is the mean absolute difference of all waypoint coordinates (cells).
Speeds and sensors are not included.

**Convergence interpretation (seed 42, one run, default config):** best-so-far
rose from 0.7468 (generation 0) to 0.8587 (generation 60); the population average
rose from 0.6516 to 0.8377 and all 30 individuals were feasible in every logged
generation. Most of the gain came early; late generations improve slowly (the
curve is flattening, not flat). Diversity fell from 12.1 to about 1.5 cells, i.e.
the population largely converged, so the search is probably partly exploitation by
the end. This is a single seed, so no statistical claim is made; Step 5 will use
multiple seeds.

**Reproduce:**
```bash
python -m uav_ga.run_ga                      # seed 42, 60 generations, writes outputs/
python -m uav_ga.run_ga --seed 7 --patience 15 --crossover drone
```
`outputs/ga_run_summary.json` records the configuration, seeds, history and
figure paths.

## 10. Visualisations (Step 4)

All drawn from recorded data: `fitness_convergence.png` (generation best,
average, best-so-far), `evolved_uav_routes.png` (best routes, numbered waypoints,
dashed return legs, status from route-level validation), `route_snapshots.png`
(best-so-far at generation 0, middle, final), `coverage_initial.png`,
`coverage_middle.png`, `coverage_final.png` (actual sensor coverage from the
fitness model on a fixed 0-1 colour scale, union over drones, targets marked
detected/missed). The middle generation is `generations_run // 2`.

## 11. Experiments (Step 5)

Environment fixed at grid seed 42. One-factor-at-a-time sensitivity (others at
default 30 / 0.15 per gene / 3): population {20, 30, 50}, mutation probability
{0.05, 0.15, 0.30} (per gene), tournament size {2, 3, 5}. Each of the 7 distinct
GA configurations and the random-search baseline is run with 10 optimiser seeds
(11, 22, ..., 99, 110). Every run has exactly 1710 fitness evaluations
(`GAConfig.max_evaluations`; the default 60-generation run uses 30 + 60 x 28
= 1710). When the budget ends mid-generation the missing slots are filled with
already-evaluated parents, so the count is exact and the number of generations
varies with population size. Only fitness evaluations count; feasibility checks
(inside the regeneration guard and the candidate sampler) are not counted, for
both methods.

Random search draws independent chromosomes with the same sampler used for the
GA's initial population, evaluates them with the identical `FitnessEvaluator`
and keeps the best (feasible first). With the same seed its first 30 draws equal
the GA's initial population; it then keeps drawing until the budget is spent.

Statistics: mean and sample SD (ddof = 1) over seeds; paired same-seed
differences; exact two-sided paired sign-flip permutation test (smallest
attainable p with 10 seeds is 2/1024 = 0.002); six level-vs-default
comparisons give a Bonferroni threshold of 0.0083. Outputs in
`outputs/experiments/`: `raw_runs.csv` (every run), `progress_curves.json`,
`summary_sensitivity.csv`, `summary_comparison.csv`, `experiment_config.json` and
six figures. `--replot` recreates summaries and figures from the saved CSV.

## 12. Technical report (Step 5)

`reports/report_template.md` holds the text with fill-in tokens;
`reports/build_report.py` computes every number from `outputs/ga_run_summary.json`
and `outputs/experiments/`, writes `reports/technical_report.md`, renders the PDF
(markdown + headless Chrome) and fails if the page count is outside 3-5.

## 13. Assumptions and limitations (for the report)

- Abstract units; physics (hover + cubic drag) is a simplified model.
- Straight-line segments; no turning-rate or altitude model.
- Detection is a geometric proxy; sensing ignores obstacles/occlusion.
- Footprints are centred on traversed cell centres (error under one cell).
- Corner policy is conservative: touching a blocked cell's corner is a collision.
- GA mutation is per gene, so its effective strength depends on chromosome size.
- Diversity is measured on waypoints only.
- Drones are assumed not to collide with each other.
- Total fleet time (sum) is used, not makespan.
- All weights and normalisation references are design choices.
- Experiments use one environment (seed 42) and 10 seeds; p-values are indicative.
- Random search shares the GA's feasibility-aware candidate sampler.
- The weighted sum is one scalarisation; no Pareto front is computed.
