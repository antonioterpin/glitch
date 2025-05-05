"""Module containing the preferences for the optimization."""

import jax
import jax.numpy as jnp

from glitch.definitions.dynamics import FleetStateInput
from glitch.utils import JAX_DEBUG_JIT

def input_effort(
    fsu: FleetStateInput,
    compensation: jnp.ndarray,
    time_discretization: float,
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
        assert fsu.u.shape[2] == compensation.shape[0], (
            "Compensation must match robot state size"
        )

    return jnp.linalg.norm((fsu.u + compensation[None, None, :]), axis=-1).mean() * time_discretization

def repulsion_loss(p, fn):
    # compute all pairwise differences: shape [R, C, C, D]
    diffs = p[:, :, None, :] - p[:, None, :, :]
    # norm over the last (feature) axis: shape [R, C, C]
    dists = jnp.linalg.norm(diffs, axis=-1)
    # build mask that is 0 on the diagonal (i==j), 1 elsewhere: shape [C, C]
    mask = 1.0 - jnp.eye(p.shape[1])
    # apply bump elementwise and sum over everything
    return jnp.mean(fn(dists) * mask)

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
    
    return repulsion_loss(fsu.p, bump).mean()