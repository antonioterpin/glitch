"""Module for visualizing a trajectory in 2D."""

import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import colorsys

from glitch.definitions.dynamics import FleetStateInput

def plot_trajectory(
    trajectories: np.ndarray, 
    working_space: Tuple[float, float, float, float],
    initial_positions: np.ndarray,
    final_positions: np.ndarray,
    title: str):
    """Plot the trajectory of a fleet state input in 2D.

    Args:
        trajectories: (horizon, n_robots, 2) array of positions.
        initial_positions: (n_robots, 2) array of initial positions.
        final_positions: (n_robots, 2) array of final positions.
        working_space: A tuple defining the working space (x_min, y_min, x_max, y_max).
        title: Title of the plot.

    Returns:
        fig: The figure object.
        ax: The axis object.
    """
    horizon, n_robots, _ = trajectories.shape

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the working space as a red rectangle
    x_min, y_min, x_max, y_max = working_space
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color='red', linestyle='--', linewidth=2, label='Working Space Boundary')

    # Gradient parameters
    v_min, v_max = 0.5, 1.0   # brightness range (0=black, 1=full color)
    saturation = 1.0          # color saturation (0=gray, 1=full)

    for i in range(n_robots):
        # Build a color for each time‐segment using HSV -> RGB
        # We hold hue constant per robot (evenly spaced around the color wheel),
        # vary 'value' from v_min to v_max over time.
        hue = i / n_robots
        seg_colors = []
        for t in range(horizon):
            value = v_min + (v_max - v_min) * (t / (horizon - 1))
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            seg_colors.append(rgb)

        # Plot start (triangle '^') and end ('s') markers
        start_rgb = colorsys.hsv_to_rgb(hue, saturation, v_min)
        end_rgb   = colorsys.hsv_to_rgb(hue, saturation, v_max)
        ax.scatter(*initial_positions[i], marker='^', c=[start_rgb],
                   s=100, edgecolors='k', label=f'Robot {i} start' if i == 0 else "")
        ax.scatter(*final_positions[i],   marker='s', c=[end_rgb],
                   s=100, edgecolors='k', label=f'Robot {i} end' if i == 0 else "")

        traj = trajectories[:, i, :]  # shape: (horizon, 2)
        # # Create line segments between successive timepoints
        # segments = np.stack([traj[:-1], traj[1:]], axis=1)  # shape: (horizon-1, 2, 2)
        # lc = LineCollection(segments, colors=seg_colors, linewidths=2)
        # ax.add_collection(lc)

        # Add filled circles at each point in the trajectory
        for t, point in enumerate(traj):
            value = v_min + (v_max - v_min) * (t / (horizon - 1))
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            ax.scatter(*point, color=[rgb], s=30, edgecolors='black')
        
    # Set labels and title
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    ax.set_title(title)
    plt.tight_layout()
    
    # Return the figure and axis for further customization if needed
    return fig, ax
