"""Module containing the preferences for the optimization."""

import jax
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
    normalization_factor: float,
) -> float:
    """Compute the collision penalty of a fleet state.

    Args:
        fsu: Fleet state input.
        collision_penalty: Collision penalty.
        normalization_factor: This modulates the distance.

    Returns:
        Collision penalty.
    """
    # TODO: to implement this one, we would need to remove the self-collision
    # from the pairwise distances, whereas for the bump this amounts to a constant.
    raise NotImplementedError(
        "Collision penalty log is not implemented. "
    )

def collision_penalty_bump(
    fsu: FleetStateInput,
    collision_penalty: float,
    normalization_factor: float,
) -> float:
    """Compute the collision penalty of a fleet state.

    Args:
        fsu: Fleet state input.
        collision_penalty: Collision penalty.
        normalization_factor: This modulates the distance.

    Returns:
        Collision penalty.
    """
    def bump(d: jnp.ndarray) -> float:
        """Compute the bump penalty.

        Args:
            d: Distance between two robots.

        Returns:
            Bump penalty.
        """
        return collision_penalty * jnp.exp(-d / normalization_factor)
    
    all_distances = _all_pairs_distances(fsu)
    # TODO: currently, we compute each distance twice.
    return jnp.sum(jax.vmap(bump)(all_distances.flatten())) / 2


def _all_pairs_distances(
    fs: FleetStateInput,
):
    """Compute the pairwise distances between all robots.

    Args:
        fs: Fleet state.

    Returns:
        Pairwise distances.
    """
    def pairwise_distance(p, all_ps):
        # p is of shape (1, n_states)
        # all_ps is of shape (n_robots, n_states)
        return jnp.linalg.norm(p[None, ...] - all_ps, axis=-1)
    
    def all_distances(p):
        # p is of shape (n_robots, n_states)
        return jax.vmap(lambda x: pairwise_distance(x, p))(p)
    
    # fs.p is of shape (horizon + 1, n_robots, n_states)
    return jax.vmap(all_distances)(fs.p)