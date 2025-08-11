"""Module for visualizing a trajectory in 2D with contextual background."""

import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt
import colorsys

def plot_trajectory(
    trajectories: np.ndarray,
    working_space: Tuple[float, float, float, float],
    initial_positions: np.ndarray,
    final_positions: np.ndarray,
    context: np.ndarray,
    title: str):
    """Plot the trajectory of a fleet state input in 2D.

    Args:
        trajectories: (horizon, n_robots, 2) array of positions.
        working_space: A tuple defining the working space (x_min, y_min, x_max, y_max).
        initial_positions: (n_robots, 2) array of initial positions.
        final_positions: (n_robots, 2) array of final positions.
        context: (res, res, 1) array providing contextual background values.
        title: Title of the plot.

    Returns:
        fig: The figure object.
        ax: The axis object.
    """
    horizon, n_robots, _ = trajectories.shape
    x_min, y_min, x_max, y_max = working_space

    # Figure and contextual background
    fig, ax = plt.subplots(figsize=(8, 6))

    # If a context map is provided, render it as a grayscale background
    if context is not None and context.size:
        # Squeeze the singleton channel dimension -> (res, res)
        context_2d = context.squeeze()

        # Use ``extent`` so image coordinates map directly to workspace coords
        im = ax.imshow(
            context_2d,
            cmap="gray",
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto",
        )

        # Append a colourbar describing the normalisation
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Context value", rotation=270, labelpad=15)

    # working‑space boundary (dashed red rectangle)
    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Working Space Boundary",
    )

    # Markers for trajectories
    v_min, v_max = 0.75, 1.0 # brightness range (0=black, 1=full colour)
    saturation = 1.0 # colour saturation (0=gray, 1=full)

    for i in range(n_robots):
        hue = i / n_robots  # unique hue per robot

        # Precompute colours per timestep (HSV -> RGB) for markers / segments
        seg_colours = [
            colorsys.hsv_to_rgb(hue, saturation, v_min + (v_max - v_min) * (t / (horizon - 1)))
            for t in range(horizon)
        ]

        # Markers for start (triangle) and end (square)
        ax.scatter(*initial_positions[i], marker="^", c=[seg_colours[0]], s=100, edgecolors="k",
                   label=f"Robot {i} start" if i == 0 else "")
        ax.scatter(*final_positions[i], marker="s", c=[seg_colours[-1]], s=100, edgecolors="k",
                   label=f"Robot {i} end" if i == 0 else "")

        # Trajectory points (small filled circles)
        for t, point in enumerate(trajectories[:, i, :]):
            ax.scatter(*point, color=[seg_colours[t]], s=30, edgecolors="k", alpha=0.8)

    ax.set_aspect("equal", "box")
    ax.grid(False)
    ax.set_title(title)
    plt.tight_layout()

    return fig, ax
