"""Step 5: hyperparameter sensitivity study and GA vs random-search comparison.

Run from the project root:   python -m uav_ga.run_experiments
Options: --seeds 11 22 33 ...  --budget 1710  --workers 8  --output-dir outputs/experiments
         --replot   rebuild the figures and tables from the saved raw_runs.csv (no re-running)

Fixed environment (grid seed 42); each configuration is repeated with every seed;
all runs use the same fitness-evaluation budget. Raw results, summaries and
figures are written to --output-dir. Results do not depend on --workers.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only

import numpy as np  # noqa: E402

from uav_ga import experiments as ex  # noqa: E402
from uav_ga.visualization import (plot_algorithm_comparison, plot_progress_comparison,  # noqa: E402
                                  plot_sensitivity, plot_sensitivity_overview)

FACTOR_LABELS = {"population_size": "Population size", "mutation_prob": "Mutation probability (per gene)",
                 "tournament_size": "Tournament size"}
FACTOR_FILES = {"population_size": "sensitivity_population_size.png",
                "mutation_prob": "sensitivity_mutation_rate.png",
                "tournament_size": "sensitivity_tournament_size.png"}


def make_figures(rows, curves, meta, out: Path) -> dict:
    paths = {}
    seeds = meta["seeds"]
    overview = []
    for factor, levels in ex.LEVELS.items():
        fit, div = {}, {}
        for lv in levels:
            grp = ex.select(rows, "GA", **{**ex.DEFAULTS, factor: lv})
            fit[lv] = [r["best_fitness"] for r in grp]
            div[lv] = [r["diversity_final"] for r in grp]
        overview.append((FACTOR_LABELS[factor], levels, fit, ex.DEFAULTS[factor]))
        path = out / FACTOR_FILES[factor]
        plot_sensitivity(FACTOR_LABELS[factor], levels, fit, div, ex.DEFAULTS[factor], len(seeds),
                         meta["budget"], str(path))
        paths[factor] = str(path)
    paths["sensitivity_overview"] = str(out / "sensitivity_overview.png")
    plot_sensitivity_overview(overview, len(seeds), meta["budget"], paths["sensitivity_overview"])
    ga_curves = [curves[ex.curve_key(r)] for r in ex.select(rows, "GA", **ex.DEFAULTS)]
    rs_curves = [curves[ex.curve_key(r)] for r in ex.select(rows, "RS")]
    paths["progress"] = str(out / "ga_vs_random_progress.png")
    plot_progress_comparison(ex.checkpoints(meta["budget"]), ga_curves, rs_curves, len(seeds), paths["progress"])
    paths["comparison"] = str(out / "ga_vs_random_comparison.png")
    plot_algorithm_comparison(ex.select(rows, "GA", **ex.DEFAULTS), ex.select(rows, "RS"), paths["comparison"])
    return paths


def print_report(rows) -> None:
    f = lambda m, s: f"{s[m + '_mean']:.4f} ± {s[m + '_std']:.4f}"
    print("\nSensitivity (final best fitness, mean ± sample SD; paired difference vs default):")
    for e in ex.sensitivity_tables(rows):
        tag = " (default)" if e["is_default"] else (f"  diff {e['vs_default_mean_diff']:+.4f}, beats default in "
                                                   f"{e['vs_default_wins']}/{e['n']} seeds, p={e['vs_default_p']:.3f}")
        print(f"  {e['factor']:>16} = {e['level']!s:<5} {f('best_fitness', e)}  gens {e['generations_mean']:.1f}"
              f"  diversity {e['diversity_final_mean']:.2f}  feasible {e['feasible_rate']:.0%}{tag}")
    print("\nGA (default) vs random search:")
    for c in ex.comparison_table(rows):
        print(f"  {c['algorithm']:<14} fitness {f('best_fitness', c)}  best {c['best_fitness_max']:.4f}  "
              f"coverage {c['coverage_mean']:.3f}  detection {c['detection_rate_mean']:.3f}  "
              f"energy {c['energy_mean']:.1f}  time {c['flight_time_mean']:.1f}  "
              f"hazard {c['hazard_exposure_mean']:.2f}  feasible {c['feasible_rate']:.0%}")
    pr = ex.paired(ex.select(rows, "GA", **ex.DEFAULTS), ex.select(rows, "RS"))
    print(f"  paired GA - RS: mean {pr['mean_diff']:+.4f} ± {pr['std_diff']:.4f}, GA better in "
          f"{pr['wins_a']}/{pr['n']} seeds, exact sign-flip p = {pr['p_value']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity + random-search comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ex.SEEDS))
    parser.add_argument("--budget", type=int, default=ex.BUDGET)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="parallel worker processes (default: CPU count, max 8; 1 = serial)")
    parser.add_argument("--env-seed", type=int, default=ex.ENV_SEED)
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--replot", action="store_true", help="recreate figures/summaries from saved results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.replot:
        rows, curves, meta = ex.load_results(out)
        meta["files"].update(make_figures(rows, curves, meta, out))
        ex.save_results(out, rows, curves, {k: v for k, v in meta.items() if k != "files"})
        (out / "experiment_config.json").write_text(json.dumps(meta, indent=2))
        print(f"Re-created figures and summaries from {out / 'raw_runs.csv'}")
        print_report(rows)
        return
    n_jobs = len(ex.build_jobs(args.seeds, args.budget, args.env_seed))
    print(f"Environment seed {args.env_seed} | seeds {args.seeds} | budget {args.budget} evaluations/run | "
          f"{n_jobs} runs | workers {args.workers}")
    rows, curves = ex.run_all(args.seeds, args.budget, args.env_seed, args.workers)

    meta = {"env_seed": args.env_seed, "seeds": list(args.seeds), "budget": args.budget,
            "defaults": ex.DEFAULTS, "levels": {k: list(v) for k, v in ex.LEVELS.items()},
            "mutation_definition": "per-gene probability", "std": "sample SD (ddof=1)",
            "p_value": "exact two-sided paired sign-flip permutation test on the mean difference"}
    meta["files"] = ex.save_results(out, rows, curves, meta)
    meta["files"].update(make_figures(rows, curves, meta, out))
    (out / "experiment_config.json").write_text(json.dumps(meta, indent=2))

    bad = [r for r in rows if r["evaluations"] != args.budget]
    print(f"Evaluation budget check: {'all runs used exactly ' + str(args.budget) if not bad else str(len(bad)) + ' runs differ'}")
    print_report(rows)
    for k, v in meta["files"].items():
        print(f"Saved {k}: {v}")


if __name__ == "__main__":
    main()
