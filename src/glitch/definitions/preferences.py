"""Module containing the preferences for the optimization."""

import jax.numpy as jnp

from glitch.definitions.dynamics import FleetStateInput
from glitch.utils import JAX_DEBUG_JIT

def input_effort(
    fsu: FleetStateInput,
    compensation: jnp.ndarray,
) -> float:
    """Compute the input effort of a fleet state.

    Args:
        fsu: Fleet state input.
        compensation: Generalized compensation (e.g., gravity).

    Returns:
        Input effort.
    """
    if JAX_DEBUG_JIT:
        assert compensation.ndim == 1, "Compensation must be of shape (n_states,)"
        assert fsu.p.shape[2] == compensation.shape[0], (
            "Compensation must match robot state size"
        )

    return jnp.sum((fsu.u + compensation[None, None, :]) ** 2)

def collision_penalty_log(
    fsu: FleetStateInput,
    collision_penalty: float,
) -> float:
    """Compute the collision penalty of a fleet state.

    Args:
        fsu: Fleet state input.
        collision_penalty: Collision penalty.

    Returns:
        Collision penalty.
    """
    raise NotImplementedError(
        "Collision penalty log is not implemented. "
    )

def collision_penalty_bump(
    fsu: FleetStateInput,
    collision_penalty: float,
) -> float:
    """Compute the collision penalty of a fleet state.

    Args:
        fsu: Fleet state input.
        collision_penalty: Collision penalty.

    Returns:
        Collision penalty.
    """
    raise NotImplementedError(
        "Collision penalty bump is not implemented. "
    )

def _all_pairs_distances(
    fs: FleetStateInput,
):
    """Compute the pairwise distances between all robots.

    Args:
        fs: Fleet state.

    Returns:
        Pairwise distances.
    """
    raise NotImplementedError(
        "Pairwise distances is not implemented. "
    )