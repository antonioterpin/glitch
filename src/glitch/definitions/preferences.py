"""Module containing the preferences for the optimization."""

import jax
import jax.numpy as jnp
from typing import Tuple

from glitch.definitions.dynamics import FleetStateInput
from glitch.utils import JAX_DEBUG_JIT

def coverage(
    fsu,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    image_shape: Tuple[int, int] = (16, 16),
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    rho: float = 0.0,
    amplitude: float = 200.0,
) -> float:
    """
    Differentiable coverage: place continuous Gaussians at each (possibly fractional)
    position, sum their contributions, and normalize.
    """
    H, W = image_shape

    # flatten positions to shape (N, 2)
    positions = fsu.p.reshape(-1, fsu.p.shape[-1])

    # map world coords -> pixel coords (still float)
    xp_x = jnp.array([min_x, max_x])
    fp_x = jnp.array([0, W])
    xp_y = jnp.array([min_y, max_y])
    fp_y = jnp.array([0, H])

    px = jnp.interp(positions[:, 0], xp_x, fp_x)
    py = jnp.interp(positions[:, 1], xp_y, fp_y)
    # stack as (N,2) with [y, x] for consistency
    centers = jnp.stack([py, px], axis=-1)

    # build pixel grid
    ys = jnp.arange(H)
    xs = jnp.arange(W)
    Y, X = jnp.meshgrid(ys, xs, indexing='ij')  # both shape (H, W)

    # expand dims to broadcast: (N, 1, 1)
    cy = centers[:, 0][:, None, None]
    cx = centers[:, 1][:, None, None]

    dx = X[None, :, :] - cx
    dy = Y[None, :, :] - cy

    one_minus_rho2 = 1.0 - rho**2
    # Mahalanobis‐like exponent
    z = (dx**2 / sigma_x**2) + (dy**2 / sigma_y**2) - (2 * rho * dx * dy) / (sigma_x * sigma_y)
    exponent = -z / (2 * one_minus_rho2)

    # each kernel: shape (N, H, W)
    kernels = amplitude * jnp.exp(exponent)

    # sum over particles -> image
    image = jnp.sum(kernels, axis=0)
    image = jnp.clip(image, 0, 255)
    coverage = jnp.sum(image / 255.0) / (H * W)
    return coverage


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


def _bilinear_interpolate(grid_vals, xs, ys, x_coords, y_coords):
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]

    ix = jnp.clip((xs - x_coords[0]) / dx, 0.0, x_coords.size - 1)
    iy = jnp.clip((ys - y_coords[0]) / dy, 0.0, y_coords.size - 1)

    i0x = ix.astype(jnp.int32)
    i0y = iy.astype(jnp.int32)
    i1x = jnp.clip(i0x + 1, 0, x_coords.size - 1)
    i1y = jnp.clip(i0y + 1, 0, y_coords.size - 1)

    fx = ix - i0x
    fy = iy - i0y

    v00 = grid_vals[i0y, i0x]
    v10 = grid_vals[i0y, i1x]
    v01 = grid_vals[i1y, i0x]
    v11 = grid_vals[i1y, i1x]

    return (
        (1.0 - fx) * (1.0 - fy) * v00
        + fx * (1.0 - fy) * v10
        + (1.0 - fx) * fy * v01
        + fx * fy * v11
    )

def reward_2d_single_agent_with_context(
    fsu: FleetStateInput,
    context: jax.Array,
    x_coords: jnp.ndarray = jnp.linspace(-5, 5, 64),
    y_coords: jnp.ndarray = jnp.linspace(-5, 5, 64),
    ):
    """Compute the reward of a fleet state based on a 2D landscape with context.

    Args:
        fsu: Fleet state input.
        context: Context for the coverage task.
        x_coords: x-coordinates of the grid.
        y_coords: y-coordinates of the grid.

    Returns:
        Reward.
    """
    positions = fsu.p.reshape(-1, fsu.p.shape[-1])
    xs, ys = positions[:, 0], positions[:, 1]

    return jnp.mean(_bilinear_interpolate(context, xs, ys, x_coords, y_coords))

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


def context_generation(key: jax.random.PRNGKey, grid_points: jnp.ndarray) -> jax.Array:
    """Generate a context for the fleet state.

    Args:
        key: Random key for JAX.
        grid_points: Grid points for the context.

    Returns:
        jax.Array: Generated context.
    """
    PRESET_CENTRES = jnp.array([
        [-2.5, -2.5],
        [ 2.5, -2.5],
        # [0, 0],
        [-2.5,  2.5],
        [ 2.5,  2.5],
    ])

    idx = jax.random.randint(key, shape=(), minval=0, maxval=PRESET_CENTRES.shape[0])
    c0 = PRESET_CENTRES[idx]
    amp = 1.0 
    sigma = 3
    two_s2 = 2 * sigma**2

    def fn(x: jax.Array) -> jax.Array:
        return amp * jnp.exp(-jnp.sum((x - c0) ** 2) / two_s2)

    return jax.vmap(fn)(grid_points).reshape(-1, 1)
