"""Plotting helpers.

Implemented: plot_grid (Step 2).
Planned, not implemented: convergence plots, UAV route maps, coverage heatmaps.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from uav_ga.grid import DisasterGrid


def plot_grid(grid: DisasterGrid, save_path: Optional[str] = None, show: bool = False):
    """Draw every layer of the grid on one map and optionally save it as an image."""
    fig, ax = plt.subplots(figsize=(10, 9))
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
    ax.set_title(f"Disaster grid {grid.width}x{grid.height} (seed {grid.seed})")

    handles = [
        Patch(facecolor="#f4f4f4", edgecolor="grey", label="Open / flyable"),
        Patch(facecolor="#222222", label="Obstacle (blocked)"),
        Patch(facecolor="#7b2cbf", alpha=0.55, label="No-fly zone"),
        Patch(facecolor="#fd8d3c", alpha=0.75, label="Hazard (darker = riskier)"),
        Patch(facecolor="#41ab5d", alpha=0.6, label="High-priority zone"),
        Line2D([], [], marker="x", color="#0050ff", linestyle="", markersize=8, label="Target"),
        Line2D([], [], marker="^", color="#00b050", markeredgecolor="black", linestyle="",
               markersize=11, label="Drone base"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
