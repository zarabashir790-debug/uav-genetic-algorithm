"""Step 3 demo: population of 30 chromosomes, evaluated with the fitness function.

Run from the project root:  python -m uav_ga.demo_step3
No evolution happens here; that is Step 4.
"""

import numpy as np

from uav_ga.config import FitnessConfig, GridConfig, PopulationConfig, UAVConfig
from uav_ga.fitness import FitnessEvaluator
from uav_ga.grid import DisasterGrid
from uav_ga.population import PopulationGenerator


def main() -> None:
    grid = DisasterGrid(GridConfig(seed=42))
    uav, pop_cfg = UAVConfig(), PopulationConfig(size=30, seed=42)
    print(f"Grid {grid.width}x{grid.height}, seed {grid.seed}, bases {grid.bases}, "
          f"{int(grid.targets.sum())} targets")

    population = PopulationGenerator(grid, uav, pop_cfg).generate()
    problems = sum(len(c.validate(grid, uav)) for c in population)
    print(f"Generated {len(population)} chromosomes, encoding problems: {problems}, "
          f"diversity {PopulationGenerator.diversity(population):.2f} cells")

    print("\nExample chromosome (index 0):")
    print(population[0].describe())

    evaluator = FitnessEvaluator(grid, uav, FitnessConfig())
    results = evaluator.evaluate_population(population)
    scores = np.array([r.fitness for r in results])
    n_feasible = sum(r.feasible for r in results)
    print(f"\nFitness over {len(results)} chromosomes ({n_feasible} feasible, "
          f"{len(results) - n_feasible} infeasible)")
    print(f"  best {scores.max():.4f}   average {scores.mean():.4f}   worst {scores.min():.4f}")

    best = int(scores.argmax())
    print(f"\nBest chromosome (index {best}):")
    print(results[best].summary())
    if not results[best].feasible:
        print("  WARNING: no feasible chromosome was found; this one violates hard constraints.")


if __name__ == "__main__":
    main()
