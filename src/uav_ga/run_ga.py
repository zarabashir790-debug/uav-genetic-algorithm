"""Step 4: run the Genetic Algorithm on the seed-42 grid and save all figures.

Run from the project root:   python -m uav_ga.run_ga
Options:                     --seed N  --generations N  --population N  --patience N
                             --crossover {drone,waypoint,mixed}  --output-dir DIR
Writes (in --output-dir): fitness_convergence.png, evolved_uav_routes.png, route_snapshots.png,
coverage_initial.png, coverage_middle.png, coverage_final.png, ga_run_summary.json
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed

from uav_ga.config import FitnessConfig, GAConfig, GridConfig, UAVConfig  # noqa: E402
from uav_ga.fitness import FitnessEvaluator  # noqa: E402
from uav_ga.genetic_algorithm import GAResult, GeneticAlgorithm  # noqa: E402
from uav_ga.grid import DisasterGrid  # noqa: E402
from uav_ga.visualization import (plot_convergence, plot_coverage, plot_route_snapshots,  # noqa: E402
                                  plot_routes)


def save_all_figures(grid: DisasterGrid, evaluator: FitnessEvaluator, result: GAResult, out: Path) -> dict:
    """Create every Step 4 figure from the recorded GA history. Returns {name: path}."""
    out.mkdir(parents=True, exist_ok=True)
    picks = [("Generation 0 (initial population best)", 0, "initial"),
             (f"Generation {result.middle_generation} (middle, best so far)", result.middle_generation, "middle"),
             (f"Generation {result.generations_run} (final, best so far)", result.generations_run, "final")]
    paths = {"convergence": out / "fitness_convergence.png", "routes": out / "evolved_uav_routes.png",
             "snapshots": out / "route_snapshots.png"}
    plot_convergence(result, paths["convergence"])
    plot_routes(grid, result.best_chromosome, result.best_evaluation, paths["routes"],
                title=f"Best evolved UAV routes (found at generation "
                      f"{result.best_so_far.index(result.best_evaluation.fitness)}, seed {result.config.seed})")
    snaps = []
    for label, g, tag in picks:
        chrom, ev = result.best_so_far_chromosomes[g], result.best_so_far_evaluations[g]
        snaps.append((label, chrom, ev))
        paths[f"coverage_{tag}"] = out / f"coverage_{tag}.png"
        plot_coverage(grid, evaluator.coverage_map(chrom), chrom, ev, f"Sensor coverage: {label}",
                      paths[f"coverage_{tag}"])
    plot_route_snapshots(grid, snaps, paths["snapshots"])
    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UAV Genetic Algorithm (Step 4).")
    defaults = GAConfig()
    parser.add_argument("--seed", type=int, default=defaults.seed, help="GA + initial population seed")
    parser.add_argument("--grid-seed", type=int, default=GridConfig.seed)
    parser.add_argument("--generations", type=int, default=defaults.max_generations)
    parser.add_argument("--population", type=int, default=defaults.population_size)
    parser.add_argument("--patience", type=int, default=None, help="early-stopping patience (off by default)")
    parser.add_argument("--crossover", default=defaults.crossover_method, choices=["drone", "waypoint", "mixed"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    grid = DisasterGrid(GridConfig(seed=args.grid_seed))
    config = GAConfig(population_size=args.population, max_generations=args.generations, seed=args.seed,
                      crossover_method=args.crossover, early_stopping_patience=args.patience)
    uav, fit_cfg = UAVConfig(), FitnessConfig()
    ga = GeneticAlgorithm(grid, uav, fit_cfg, config)

    print(f"Grid seed {args.grid_seed} | GA seed {config.seed} | population {config.population_size} | "
          f"max generations {config.max_generations}")
    print(f"crossover {config.crossover_method} (p={config.crossover_prob}), mutation p={config.mutation_prob}"
          f" per gene, tournament {config.tournament_size}, elites {config.elite_count}, "
          f"early-stopping patience {config.early_stopping_patience}")
    print(f"{'gen':>4} {'gen-best':>9} {'average':>9} {'best-so-far':>12} {'feasible':>9} {'diversity':>10}")

    def progress(gen: int, r: GAResult) -> None:
        if gen % args.log_every == 0 or gen == config.max_generations:
            print(f"{gen:>4} {r.generation_best[-1]:>9.4f} {r.generation_average[-1]:>9.4f} "
                  f"{r.best_so_far[-1]:>12.4f} {r.feasible_count[-1]:>5}/{config.population_size:<3} "
                  f"{r.diversity[-1]:>10.2f}")

    result = ga.run(progress=progress)
    print(f"\nStopped after {result.generations_run} generations: {result.stop_reason}")
    print(f"Fitness evaluations: {result.evaluations} | runtime {result.runtime_seconds:.1f}s | "
          f"crossover fallbacks {result.crossover_fallbacks} | mutation fallbacks {result.mutation_fallbacks}")

    initial = result.best_so_far_evaluations[0]
    final = result.best_evaluation
    print(f"\nInitial population: best {result.generation_best[0]:.4f}, average {result.generation_average[0]:.4f}, "
          f"worst {result.generation_worst[0]:.4f}")
    print(f"Final population  : best {result.generation_best[-1]:.4f}, average {result.generation_average[-1]:.4f}, "
          f"worst {result.generation_worst[-1]:.4f}")
    gain = final.fitness - initial.fitness
    print(f"Best-so-far: {initial.fitness:.4f} -> {final.fitness:.4f}  (change {gain:+.4f}, "
          f"{100 * gain / abs(initial.fitness):+.1f}% of initial best)")

    print("\nBest evolved chromosome:")
    print(result.best_chromosome.describe())
    print("\nBest evolved chromosome evaluation:")
    print(final.summary())
    print("\nInitial best vs final best:")
    for name in ("coverage", "detection_rate", "energy", "flight_time", "hazard_exposure", "penalty"):
        print(f"  {name:>16}: {getattr(initial, name):9.3f} -> {getattr(final, name):9.3f}")

    # independent re-check of the final answer, not trusting the stored evaluation
    recheck = FitnessEvaluator(grid, uav, fit_cfg).evaluate(result.best_chromosome)
    problems = result.best_chromosome.validate(grid, uav)
    print(f"\nRe-validation of final chromosome: encoding problems {len(problems)}, "
          f"route-level feasible = {recheck.feasible}")

    paths = save_all_figures(grid, ga.evaluator, result, Path(args.output_dir))
    for name, path in paths.items():
        print(f"Saved {name}: {path}")

    summary = {
        "grid_seed": args.grid_seed, "ga_config": asdict(config), "uav_config": asdict(uav),
        "fitness_config": asdict(fit_cfg), "stop_reason": result.stop_reason,
        "generations_run": result.generations_run, "evaluations": result.evaluations,
        "initial_best_fitness": initial.fitness, "final_best_fitness": final.fitness,
        "final_feasible": recheck.feasible, "final_evaluation": {k: v for k, v in final.as_dict().items()
                                                                if k != "drones"},
        "history": {"generation_best": result.generation_best, "generation_average": result.generation_average,
                    "best_so_far": result.best_so_far, "feasible_count": result.feasible_count,
                    "diversity": result.diversity},
        "figures": paths,
    }
    summary_path = Path(args.output_dir) / "ga_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
