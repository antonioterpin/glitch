"""Module for generating random configurations."""

import jax
from glitch.utils import JAX_DEBUG_JIT
from glitch.definitions.dynamics import FleetStateInput

def sample_from_box(
    key: jax.random.PRNGKey,
    box: jax.numpy.ndarray,
    n_robots: int,
    zero_velocity: bool = False,
) -> FleetStateInput:
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
        assert box.ndim == 2, "Box must be of shape (n_states, 2)"
        assert n_robots > 0, "Number of robots must be positive"
        assert box.shape[1] == 2 or (
            box.shape[1] == 4 and not zero_velocity
        ), "Box contain upper and lower bounds"

    keyp, keyv = jax.random.split(key)
    if zero_velocity:
        v = jax.numpy.zeros((1, n_robots, box.shape[0]))
    else:
        # Generate random velocities
        v = jax.random.uniform(
            keyv,
            shape=(1, n_robots, box.shape[0]),
            minval=box[:, 2],
            maxval=box[:, 3],
        )

    # Generate random samples
    p = jax.random.uniform(
        keyp,
        shape=(1, n_robots, box.shape[0]),
        minval=box[:, 0],
        maxval=box[:, 1],
    )

    return FleetStateInput(
        p=p,
        v=v,
        u=jax.numpy.zeros((0, n_robots, box.shape[0])),
    )