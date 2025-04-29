"""Module for generating random configurations."""

import jax
from glitch.dynamics import FleetState
from glitch.common import JAX_DEBUG_JIT

def sample_from_box(
    key: jax.random.PRNGKey,
    box: jax.numpy.ndarray,
    n_robots: int,
    zero_velocity: bool = False,
) -> jax.numpy.ndarray:
    """Sample from a box in R^n.

    Args:
        key: JAX random key.
        box: Box in R^n.
        n_robots: Number of robots.
        zero_velocity: If True, sample zero velocity.

    Returns:
        Samples from the box.
    """
    if JAX_DEBUG_JIT:
        assert box.ndim == 2, "Box must be of shape (n, 2)"
        assert n_robots > 0, "Number of robots must be positive"
        assert box.shape[1] == 2, "Box contain upper and lower bounds"

    keyp, keyv = jax.random.split(key)
    if zero_velocity:
        v = jax.numpy.zeros((n_robots, box.shape[0]))
    else:
        # Generate random velocities
        v = jax.random.uniform(
            keyv,
            shape=(n_robots, box.shape[0]),
            minval=box[:, 0],
            maxval=box[:, 1],
        )

    # Generate random samples
    p = jax.random.uniform(
        keyp,
        shape=(n_robots, box.shape[0]),
        minval=box[:, 0],
        maxval=box[:, 1],
    )

    return FleetState(
        p=p,
        v=v,
    )