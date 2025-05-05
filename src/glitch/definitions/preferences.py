"""Module containing the preferences for the optimization."""

import jax
import jax.numpy as jnp

from glitch.definitions.dynamics import FleetStateInput
from glitch.utils import JAX_DEBUG_JIT

def coverage(
    fsu: FleetStateInput,
    coverage_radius: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    pixels_per_meter: int,
) -> float:
    """Compute the coverage of a fleet state.

    Args:
        fsu: Fleet state input.
        coverage_radius: Coverage radius.
        min_x: Minimum x coordinate.
        max_x: Maximum x coordinate.
        min_y: Minimum y coordinate.
        max_y: Maximum y coordinate.
        pixels_per_meter: Pixels per meter.

    Returns:
        Coverage.
    """
    
    

def ishigami(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Ishigami function.

    .. math::
        f(v) = \sin(z_1) + 7 \sin(z_2)^2 + 0.1 \left(\frac{z_1 + z_2}{2}\right)^4 \sin(z_1)

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Ishigami function.
    """
    v *= 1.25
    d = v.shape[0]
    v0 = jnp.mean(v[:d//2])
    v1 = jnp.mean(v[d//2:])
    v2 = (v0 + v1) / 2
    return jnp.sin(v0) + 7 * jnp.sin(v1) ** 2 + 0.1 * v2 ** 4 * jnp.sin(v0)

def reward_2d_single_agent(
    fsu: FleetStateInput,
    ):
    """Compute the reward of a fleet state based on a 2D landscape.

    Args:
        fsu: Fleet state input.
    
    Returns:
        Reward.
    """
    return jnp.mean(jax.vmap(ishigami)(fsu.p.reshape(-1, fsu.p.shape[-1])))

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