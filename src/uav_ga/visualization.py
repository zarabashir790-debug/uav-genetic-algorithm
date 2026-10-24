"""Plotting helpers.

plot_grid (Step 2); plot_convergence, plot_routes, plot_route_snapshots, plot_coverage (Step 4);
plot_sensitivity, plot_sensitivity_overview, plot_progress_comparison, plot_algorithm_comparison
(Step 5 experiments).
All plots are drawn from real data (grid layers, GAResult, FitnessEvaluator output).
"""

from typing import List, Optional, Sequence, Tuple

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from uav_ga.grid import DisasterGrid


def grid_legend_handles() -> list:
    return [
        Patch(facecolor="#f4f4f4", edgecolor="grey", label="Open / flyable"),
        Patch(facecolor="#222222", label="Obstacle (blocked)"),
        Patch(facecolor="#7b2cbf", alpha=0.55, label="No-fly zone"),
        Patch(facecolor="#fd8d3c", alpha=0.75, label="Hazard (darker = riskier)"),
        Patch(facecolor="#41ab5d", alpha=0.6, label="High-priority zone"),
        Line2D([], [], marker="x", color="#0050ff", linestyle="", markersize=8, label="Target"),
        Line2D([], [], marker="^", color="#00b050", markeredgecolor="black", linestyle="",
               markersize=11, label="Drone base"),
    ]


def draw_grid(ax, grid: DisasterGrid, title: Optional[str] = None) -> None:
    """Draw every grid layer, targets and bases on ``ax``."""
    extent = (-0.5, grid.width - 0.5, -0.5, grid.height - 0.5)

    def layer(mask_or_values, cmap, vmin=None, vmax=None, alpha=1.0, zorder=1):
        data = np.ma.masked_where(mask_or_values == 0, mask_or_values)
        ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                  alpha=alpha, zorder=zorder, interpolation="nearest")

    ax.set_facecolor("#f4f4f4")  # open, flyable cells
    layer(grid.priority, "Greens", 0, 1.3, alpha=0.6, zorder=1)
    layer(grid.hazards, "YlOrRd", 0, 1.0, alpha=0.75, zorder=2)
    layer(grid.no_fly.astype(int), ListedColormap(["#7b2cbf"]), 0, 1, alpha=0.55, zorder=3)
    layer(grid.obstacles.astype(int), ListedColormap(["#222222"]), 0, 1, zorder=4)

    tx, ty = zip(*grid.target_cells) if grid.target_cells else ([], [])
    ax.scatter(tx, ty, marker="x", c="#0050ff", s=60, linewidths=2, zorder=5)
    bx, by = zip(*grid.bases)
    ax.scatter(bx, by, marker="^", c="#00b050", edgecolors="black", s=170, zorder=6)
    for i, (x, y) in enumerate(grid.bases, start=1):
        ax.annotate(f"B{i}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold", zorder=7)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks(range(0, grid.width, 5))
    ax.set_yticks(range(0, grid.height, 5))
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, zorder=0)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    ax.set_title(title or f"Disaster grid {grid.width}x{grid.height} (seed {grid.seed})")


def plot_grid(grid: DisasterGrid, save_path: Optional[str] = None, show: bool = False):
    """Draw every layer of the grid on one map and optionally save it as an image."""
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_grid(ax, grid)
    ax.legend(handles=grid_legend_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ----------------------------------------------------------------------
# Step 4 plots
# ----------------------------------------------------------------------
DRONE_COLORS = ["#d7191c", "#00838f", "#6d4c41"]


def _save(fig, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def feasibility_label(breakdown) -> str:
    """Honest status text from the fitness evaluation (route-level validation)."""
    if breakdown.feasible:
        return "FEASIBLE: all routes validated (no obstacle/no-fly/out-of-bounds cells, within range)"
    return (f"INFEASIBLE: obstacle cells {breakdown.obstacle_cells}, no-fly cells {breakdown.no_fly_cells}, "
            f"out-of-bounds {breakdown.out_of_bounds_cells}, range excess {breakdown.range_excess:.1f}")


def draw_routes(ax, grid: DisasterGrid, chromosome, breakdown, title: str) -> None:
    """Grid + the three UAV routes: arrows follow flight order, dashed = return to base."""
    draw_grid(ax, grid, title=f"{title}\nfitness {breakdown.fitness:.4f} | {feasibility_label(breakdown)}")
    ax.title.set_fontsize(9)
    for d, plan in enumerate(chromosome.plans):
        color = DRONE_COLORS[d % len(DRONE_COLORS)]
        last = plan.n_segments - 1
        for k, (p0, p1) in enumerate(plan.segments()):
            ax.annotate("", xy=p1, xytext=p0, zorder=8,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8, shrinkA=0, shrinkB=0,
                                        linestyle="--" if k == last else "-", mutation_scale=12))
        for w, (x, y) in enumerate(plan.waypoints, start=1):
            ax.scatter([x], [y], s=210 if w == 1 else 120, c="white", edgecolors=color,
                       linewidths=2.5 if w == 1 else 1.6, zorder=9)
            ax.text(x, y, str(w), ha="center", va="center", fontsize=7, fontweight="bold",
                    color=color, zorder=10)


def route_legend_handles(chromosome) -> list:
    handles = grid_legend_handles()
    for d in range(chromosome.n_drones):
        handles.append(Line2D([], [], color=DRONE_COLORS[d % len(DRONE_COLORS)], lw=2, label=f"UAV {d + 1} route"))
    handles += [
        Line2D([], [], color="black", lw=1.8, linestyle="-", label="Outbound / between waypoints"),
        Line2D([], [], color="black", lw=1.8, linestyle="--", label="Return to base"),
        Line2D([], [], marker="o", color="grey", markerfacecolor="white", linestyle="",
               markersize=9, label="Waypoint (number = visit order; large = deployment wp 1)"),
    ]
    return handles


def plot_routes(grid: DisasterGrid, chromosome, breakdown, save_path: Optional[str] = None,
                title: str = "Best evolved UAV routes"):
    fig, ax = plt.subplots(figsize=(12, 9))
    draw_routes(ax, grid, chromosome, breakdown, title)
    ax.legend(handles=route_legend_handles(chromosome), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_route_snapshots(grid: DisasterGrid, snapshots: Sequence[Tuple[str, object, object]],
                         save_path: Optional[str] = None):
    """Side-by-side best-so-far routes. ``snapshots`` = [(label, chromosome, breakdown), ...]."""
    fig, axes = plt.subplots(1, len(snapshots), figsize=(8.5 * len(snapshots), 9.5))
    axes = np.atleast_1d(axes)
    for ax, (label, chrom, ev) in zip(axes, snapshots):
        draw_routes(ax, grid, chrom, ev, label)
    fig.legend(handles=route_legend_handles(snapshots[0][1]), loc="lower center",
               ncol=5, fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, save_path)
    return fig


def plot_convergence(result, save_path: Optional[str] = None):
    """Generation-best, population-average and best-so-far fitness from a real GA run."""
    gens = result.generations
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(gens, result.generation_best, color="#1f77b4", lw=1.6, label="Generation best")
    ax.plot(gens, result.generation_average, color="#ff7f0e", lw=1.6, label="Generation average")
    ax.plot(gens, result.best_so_far, color="#2ca02c", lw=2, linestyle="--", label="Best so far")
    if min(result.generation_average) < 0:
        ax.axhline(0, color="grey", lw=0.8, linestyle=":")
        ax.text(gens[-1], 0, " feasible above / infeasible below", va="bottom", ha="right", fontsize=8,
                color="grey")
    ax.set_xlabel("Generation (0 = initial population)")
    ax.set_ylabel("Fitness score")
    ax.set_title(f"GA fitness convergence (seed {result.config.seed}, "
                 f"{result.generations_run} generations, population {result.config.population_size})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_coverage(grid: DisasterGrid, coverage_map: np.ndarray, chromosome, breakdown,
                  title: str, save_path: Optional[str] = None):
    """Heatmap of actual sensor coverage (0 = unseen, 0.5 = LOW, 1 = HIGH), fixed 0-1 colour scale."""
    fig, ax = plt.subplots(figsize=(10, 8.5))
    extent = (-0.5, grid.width - 0.5, -0.5, grid.height - 0.5)
    im = ax.imshow(coverage_map, origin="lower", extent=extent, cmap="Blues", vmin=0.0, vmax=1.0,
                   interpolation="nearest", zorder=1)
    ax.imshow(np.ma.masked_where(~grid.obstacles, grid.obstacles.astype(int)), origin="lower",
              extent=extent, cmap=ListedColormap(["#222222"]), zorder=2, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~grid.no_fly, grid.no_fly.astype(int)), origin="lower", extent=extent,
              cmap=ListedColormap(["#7b2cbf"]), alpha=0.5, zorder=2, interpolation="nearest")
    ax.contour(grid.priority, levels=[0.3], colors=["#238b45"], linewidths=1.3, zorder=3)  # priority edge
    for d, plan in enumerate(chromosome.plans):
        pts = plan.route_points
        ax.plot(pts[:, 0], pts[:, 1], color=DRONE_COLORS[d % len(DRONE_COLORS)], lw=1.2, alpha=0.8, zorder=4)
    seen = [(x, y) for x, y in grid.target_cells if coverage_map[y, x] > 0]
    missed = [(x, y) for x, y in grid.target_cells if coverage_map[y, x] == 0]
    if seen:
        ax.scatter(*zip(*seen), marker="o", facecolors="none", edgecolors="#00c000", s=90, linewidths=2, zorder=5)
    if missed:
        ax.scatter(*zip(*missed), marker="x", c="#e60000", s=60, linewidths=2, zorder=5)
    ax.scatter(*zip(*grid.bases), marker="^", c="#00b050", edgecolors="black", s=150, zorder=6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    ax.set_title(f"{title}\npriority-weighted coverage {breakdown.coverage:.3f} | "
                 f"targets detected {breakdown.detected_targets}/{int(grid.targets.sum())} | "
                 f"{'feasible' if breakdown.feasible else 'INFEASIBLE'}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                 label="Sensor coverage effectiveness (0 none, 0.5 LOW, 1.0 HIGH)")
    ax.legend(handles=[
        Patch(facecolor="#222222", label="Obstacle"),
        Patch(facecolor="#7b2cbf", alpha=0.5, label="No-fly zone"),
        Line2D([], [], color="#238b45", lw=1.3, label="High-priority zone edge (priority 0.3)"),
        Line2D([], [], color="grey", lw=1.2, label="UAV routes"),
        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#00c000", linestyle="",
               markersize=9, label="Target detected"),
        Line2D([], [], marker="x", color="#e60000", linestyle="", markersize=8, label="Target missed"),
        Line2D([], [], marker="^", color="#00b050", markeredgecolor="black", linestyle="", markersize=10,
               label="Drone base"),
    ], loc="upper left", bbox_to_anchor=(1.22, 1.0), fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ----------------------------------------------------------------------
# Step 5 experiment plots (colours validated for colour-blind separation: blue = GA, orange = random search)
# ----------------------------------------------------------------------
GA_COLOR = "#2a78d6"
RS_COLOR = "#eb6834"
_INK, _MUTED = "#0b0b0b", "#52514e"


def _clean_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a96")
    ax.tick_params(colors=_MUTED)
    ax.yaxis.grid(True, color="#e3e3e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)


def _mean_sd_points(ax, xs, groups, color, width=0.12, label=None):
    """Mean +/- sample SD error bars with every seed's value as a small dot."""
    rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible
    for x, vals in zip(xs, groups):
        vals = np.asarray(vals, dtype=float)
        ax.scatter(x + rng.uniform(-width, width, len(vals)), vals, s=14, color=color, alpha=0.35,
                   linewidths=0, zorder=2)
    means = [np.mean(v) for v in groups]
    sds = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in groups]
    ax.errorbar(xs, means, yerr=sds, fmt="o-", color=color, ecolor=color, elinewidth=2, capsize=5,
                markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5, linewidth=1.5, zorder=3,
                label=label)


def plot_sensitivity(label: str, levels: Sequence, fitness: dict, diversity: dict, default_level,
                     n_seeds: int, budget: int, save_path: Optional[str] = None):
    """Final best fitness (left) and final population diversity (right) vs one hyperparameter.

    ``fitness`` / ``diversity`` map each level to the list of per-seed values.
    """
    xs = list(range(len(levels)))
    ticks = [f"{lv}\n(default)" if lv == default_level else f"{lv}" for lv in levels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, data, ylab, title in ((axes[0], fitness, "Final best fitness", "Final best fitness"),
                                  (axes[1], diversity, "Final population diversity (cells)",
                                   "Final population diversity")):
        _mean_sd_points(ax, xs, [data[lv] for lv in levels], GA_COLOR)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_xlabel(label)
        ax.set_ylabel(ylab)
        ax.set_title(title, color=_INK, fontsize=11)
        _clean_axes(ax)
    fig.suptitle(f"Sensitivity to {label.lower()}  (mean ± SD over {n_seeds} seeds, dots = single seeds, "
                 f"budget {budget} evaluations)", color=_INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)
    return fig


def plot_progress_comparison(evals: Sequence[int], ga_curves, rs_curves, n_seeds: int,
                             save_path: Optional[str] = None):
    """Best-so-far fitness against fitness evaluations: mean ± SD band over seeds, GA vs random search."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.asarray(evals, dtype=float)
    for curves, color, name in ((rs_curves, RS_COLOR, "Random search"), (ga_curves, GA_COLOR, "Genetic algorithm")):
        arr = np.array([[np.nan if v is None else v for v in c] for c in curves], dtype=float)
        with warnings.catch_warnings():  # early checkpoints can be all-NaN for the GA (no generation finished yet)
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, sd = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1)
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mu, color=color, linewidth=2.2, label=name)
        ax.annotate(name, (x[-1], mu[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=_INK, fontsize=9)
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Best-so-far fitness")
    ax.set_title(f"GA vs random search: best-so-far fitness (mean ± SD, {n_seeds} seeds)", color=_INK,
                 fontsize=11)
    ax.set_xlim(x[0], x[-1] * 1.22)
    ax.legend(loc="lower right", frameon=False)
    _clean_axes(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_algorithm_comparison(ga_rows: List[dict], rs_rows: List[dict], save_path: Optional[str] = None):
    """Final performance of GA vs random search on every objective (mean ± SD, dots = seeds)."""
    panels = [("best_fitness", "Final best fitness", "higher is better"),
              ("coverage", "Priority-weighted coverage", "higher is better"),
              ("detection_rate", "Target detection rate", "higher is better"),
              ("energy", "Energy (eu)", "lower is better"),
              ("flight_time", "Flight time (tu)", "lower is better"),
              ("hazard_exposure", "Hazard exposure", "lower is better")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for ax, (key, title, note) in zip(axes.ravel(), panels):
        for x, rows, color in ((0, ga_rows, GA_COLOR), (1, rs_rows, RS_COLOR)):
            vals = [r[key] for r in rows]
            if key == "best_fitness":  # zoomed axis: coloured dots instead of bars that would start at 0
                _mean_sd_points(ax, [x], [vals], color, width=0.1)
                continue
            ax.bar(x, np.mean(vals), width=0.5, color=color, alpha=0.85, zorder=1)
            _mean_sd_points(ax, [x], [vals], "#222222", width=0.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["GA", "Random\nsearch"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{title}\n({note})", color=_INK, fontsize=9.5)
        _clean_axes(ax)
    fig.suptitle(f"Final performance after {int(ga_rows[0]['evaluations'])} fitness evaluations "
                 f"(mean ± SD over {len(ga_rows)} seeds, dots = single seeds)", color=_INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, save_path)
    return fig


def plot_sensitivity_overview(panels: Sequence[tuple], n_seeds: int, budget: int,
                              save_path: Optional[str] = None):
    """Compact 1x3 overview: final best fitness vs each hyperparameter on a shared y-axis.

    ``panels`` = [(axis label, levels, {level: [per-seed fitness]}, default level), ...].
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.3), sharey=True)
    for ax, (label, levels, fitness, default) in zip(np.atleast_1d(axes), panels):
        xs = list(range(len(levels)))
        _mean_sd_points(ax, xs, [fitness[lv] for lv in levels], GA_COLOR)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lv}*" if lv == default else f"{lv}" for lv in levels])
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_xlabel(label)
        _clean_axes(ax)
    np.atleast_1d(axes)[0].set_ylabel("Final best fitness")
    fig.suptitle(f"Hyperparameter sensitivity: mean ± SD over {n_seeds} seeds, dots = single seeds, "
                 f"{budget} evaluations per run (* = default)", color=_INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, save_path)
    return fig
