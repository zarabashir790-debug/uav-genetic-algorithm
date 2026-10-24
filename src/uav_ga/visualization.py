"""Plotting helpers.

plot_grid (Step 2); plot_convergence, plot_routes, plot_route_snapshots, plot_coverage (Step 4).
All plots are drawn from real data (grid layers, GAResult, FitnessEvaluator output).
"""

from typing import List, Optional, Sequence, Tuple

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
