"""Hyperparameter sensitivity and GA vs random-search experiments (Step 5).

Design
------
* Environment fixed (DisasterGrid seed 42) so differences come from the algorithm.
* One-factor-at-a-time: population size, mutation probability (PER GENE, as in
  GAConfig) and tournament size are varied one at a time, the others stay at the
  defaults (30 / 0.15 / 3).
* Every GA run and every random-search run gets the SAME budget of fitness
  evaluations (default 1710, the evaluation count of the default 60-generation
  run). The GA stops through ``GAConfig.max_evaluations``.
* Each configuration is repeated with the same list of optimiser seeds. The
  seed drives the initial population and all GA randomness (or the random-search
  draws), so runs are independent and reproducible.
* Statistics: mean and SAMPLE standard deviation (ddof = 1) over seeds; paired
  differences use the same seed for both sides; the p-value is an exact
  two-sided paired sign-flip permutation test on the mean difference. With few
  seeds and several comparisons the p-values are indicative, not conclusive.
"""

import csv
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from uav_ga.config import GAConfig, GridConfig
from uav_ga.genetic_algorithm import GeneticAlgorithm
from uav_ga.grid import DisasterGrid
from uav_ga.random_search import RandomSearch

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
ENV_SEED = 42
BUDGET = 1710
CURVE_STEP = 10
DEFAULTS = {"population_size": 30, "mutation_prob": 0.15, "tournament_size": 3}
LEVELS = {"population_size": (20, 30, 50), "mutation_prob": (0.05, 0.15, 0.30),
          "tournament_size": (2, 3, 5)}
METRICS = ("best_fitness", "coverage", "detection_rate", "energy", "flight_time", "hazard_exposure",
           "diversity_final", "evaluations", "generations")


# ----------------------------------------------------------------------
# Single runs
# ----------------------------------------------------------------------
def checkpoints(budget: int, step: int = CURVE_STEP) -> List[int]:
    pts = list(range(step, budget + 1, step))
    return pts if pts and pts[-1] == budget else pts + [budget]


def _ga_curve(result, budget: int) -> List[Optional[float]]:
    """Best-so-far fitness at each checkpoint (None before the first generation completes)."""
    curve = []
    for e in checkpoints(budget):
        idx = [g for g, n in enumerate(result.cumulative_evaluations) if n <= e]
        curve.append(result.best_so_far[idx[-1]] if idx else None)
    return curve


def _row(algorithm: str, params: dict, seed: int, ev, **extra) -> dict:
    row = {"algorithm": algorithm, **params, "seed": seed, "best_fitness": ev.fitness,
           "coverage": ev.coverage, "detection_rate": ev.detection_rate,
           "detected_targets": ev.detected_targets, "energy": ev.energy, "flight_time": ev.flight_time,
           "hazard_exposure": ev.hazard_exposure, "penalty": ev.penalty, "feasible": int(ev.feasible)}
    row.update(extra)
    return row


def execute_job(spec: dict) -> Tuple[dict, list]:
    """Run one GA or random-search job. Top-level so it can run in a worker process."""
    grid = DisasterGrid(GridConfig(seed=spec["env_seed"]))
    seed, budget = spec["seed"], spec["budget"]
    if spec["algorithm"] == "GA":
        p = spec["params"]
        cfg = GAConfig(population_size=p["population_size"], mutation_prob=p["mutation_prob"],
                       tournament_size=p["tournament_size"], seed=seed,
                       max_generations=10 ** 6, max_evaluations=budget)
        r = GeneticAlgorithm(grid, config=cfg).run()
        row = _row("GA", p, seed, r.best_evaluation, diversity_final=r.diversity[-1],
                   diversity_initial=r.diversity[0], evaluations=r.evaluations, generations=r.generations_run,
                   feasible_final_pop=r.feasible_count[-1], stop_reason=r.stop_reason,
                   runtime_s=r.runtime_seconds)
        return row, _ga_curve(r, budget)
    r = RandomSearch(grid).run(budget, seed)
    row = _row("RS", {"population_size": "", "mutation_prob": "", "tournament_size": ""}, seed,
               r.best_evaluation, diversity_final="", diversity_initial="", evaluations=r.evaluations,
               generations="", feasible_final_pop=r.feasible_count, stop_reason="budget",
               runtime_s=r.runtime_seconds)
    return row, [r.best_so_far[e - 1] for e in checkpoints(budget)]


def build_jobs(seeds: Sequence[int], budget: int, env_seed: int = ENV_SEED) -> List[dict]:
    configs = [dict(DEFAULTS)]
    for factor, levels in LEVELS.items():
        for level in levels:
            cfg = {**DEFAULTS, factor: level}
            if cfg not in configs:
                configs.append(cfg)
    jobs = [{"algorithm": "GA", "params": c, "seed": s, "budget": budget, "env_seed": env_seed}
            for c in configs for s in seeds]
    jobs += [{"algorithm": "RS", "params": {}, "seed": s, "budget": budget, "env_seed": env_seed}
             for s in seeds]
    return jobs


def run_all(seeds: Sequence[int] = SEEDS, budget: int = BUDGET, env_seed: int = ENV_SEED,
            workers: int = 1) -> Tuple[List[dict], Dict[str, list]]:
    """Run every job (optionally in parallel). Results do not depend on ``workers``."""
    jobs = build_jobs(seeds, budget, env_seed)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(execute_job, jobs))
    else:
        results = [execute_job(j) for j in jobs]
    rows = [r for r, _ in results]
    curves = {curve_key(r): c for r, (_, c) in zip(rows, results)}
    return rows, curves


def curve_key(row: dict) -> str:
    if row["algorithm"] == "RS":
        return f"RS|seed{row['seed']}"
    return (f"GA|pop{row['population_size']}|mut{row['mutation_prob']}|"
            f"tour{row['tournament_size']}|seed{row['seed']}")


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in values if x != ""], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (math.nan, math.nan)


def select(rows: List[dict], algorithm: str, **params) -> List[dict]:
    """Rows of one algorithm/configuration, ordered by seed."""
    out = [r for r in rows if r["algorithm"] == algorithm and all(r[k] == v for k, v in params.items())]
    return sorted(out, key=lambda r: r["seed"])


def summarize(group_rows: List[dict]) -> dict:
    """Mean/std of every metric plus best/worst observed fitness and feasible rate."""
    out = {"n": len(group_rows)}
    for m in METRICS:
        mu, sd = mean_std([r[m] for r in group_rows])
        out[f"{m}_mean"], out[f"{m}_std"] = mu, sd
    fit = [r["best_fitness"] for r in group_rows]
    out["best_fitness_max"], out["best_fitness_min"] = max(fit), min(fit)
    out["feasible_rate"] = sum(r["feasible"] for r in group_rows) / len(group_rows)
    return out


def sign_flip_p(diffs: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip permutation p-value for the mean difference."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0 or n > 20:
        return math.nan
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    stats = np.abs((signs * d).sum(axis=1))
    return float(np.mean(stats >= abs(d.sum()) - 1e-12))


def paired(rows_a: List[dict], rows_b: List[dict], metric: str = "best_fitness") -> dict:
    """Per-seed paired difference a - b (same seeds)."""
    a = {r["seed"]: r[metric] for r in rows_a}
    b = {r["seed"]: r[metric] for r in rows_b}
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    mu, sd = mean_std(diffs)
    return {"n": len(diffs), "mean_diff": mu, "std_diff": sd, "wins_a": sum(d > 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs), "p_value": sign_flip_p(diffs), "diffs": diffs}


def sensitivity_tables(rows: List[dict]) -> List[dict]:
    """One summary row per (factor, level) + paired comparison against the default level."""
    table = []
    for factor, levels in LEVELS.items():
        default_rows = select(rows, "GA", **DEFAULTS)
        for level in levels:
            grp = select(rows, "GA", **{**DEFAULTS, factor: level})
            entry = {"factor": factor, "level": level, "is_default": level == DEFAULTS[factor],
                     **summarize(grp)}
            if level != DEFAULTS[factor]:
                pr = paired(grp, default_rows)
                entry.update(vs_default_mean_diff=pr["mean_diff"], vs_default_std_diff=pr["std_diff"],
                             vs_default_wins=pr["wins_a"], vs_default_p=pr["p_value"])
            table.append(entry)
    return table


def comparison_table(rows: List[dict]) -> List[dict]:
    ga, rs = select(rows, "GA", **DEFAULTS), select(rows, "RS")
    return [{"algorithm": "GA (default)", **summarize(ga)}, {"algorithm": "Random search", **summarize(rs)}]


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------
def _write_csv(path: Path, rows: List[dict]) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_results(out_dir: Path, rows: List[dict], curves: Dict[str, list], meta: dict) -> Dict[str, str]:
    """Write raw runs, curves, summaries and configuration so every statistic is reproducible."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"raw_runs": out_dir / "raw_runs.csv", "progress_curves": out_dir / "progress_curves.json",
             "sensitivity_summary": out_dir / "summary_sensitivity.csv",
             "comparison_summary": out_dir / "summary_comparison.csv",
             "experiment_config": out_dir / "experiment_config.json"}
    _write_csv(paths["raw_runs"], rows)
    paths["progress_curves"].write_text(json.dumps({"checkpoints": checkpoints(meta["budget"]),
                                                    "curves": curves}))
    _write_csv(paths["sensitivity_summary"], sensitivity_tables(rows))
    comp = comparison_table(rows)
    pr = paired(select(rows, "GA", **DEFAULTS), select(rows, "RS"))
    comp.append({"algorithm": "GA minus RS (paired)", "n": pr["n"], "best_fitness_mean": pr["mean_diff"],
                 "best_fitness_std": pr["std_diff"], "wins_ga": pr["wins_a"], "p_value": pr["p_value"]})
    _write_csv(paths["comparison_summary"], comp)
    paths["experiment_config"].write_text(json.dumps(meta, indent=2))
    return {k: str(v) for k, v in paths.items()}


def load_results(out_dir: Path) -> Tuple[List[dict], Dict[str, list], dict]:
    """Reload saved raw runs, progress curves and config (to re-create tables/figures without re-running)."""
    ints = {"population_size", "tournament_size", "seed", "feasible", "evaluations", "generations",
            "detected_targets", "feasible_final_pop"}
    floats = {"mutation_prob", "best_fitness", "coverage", "detection_rate", "energy", "flight_time",
              "hazard_exposure", "penalty", "diversity_final", "diversity_initial", "runtime_s"}
    rows = []
    with open(Path(out_dir) / "raw_runs.csv") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if v == "":
                    continue
                r[k] = int(v) if k in ints else float(v) if k in floats else v
            rows.append(r)
    curves = json.loads((Path(out_dir) / "progress_curves.json").read_text())["curves"]
    meta = json.loads((Path(out_dir) / "experiment_config.json").read_text())
    return rows, curves, meta
