"""Module returning the matrices for the constraints."""

import jax.numpy as jnp
from typing import Tuple

from hcnn.project import Project
from hcnn.constraints.box import BoxConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.affine_equality import EqualityConstraint

from glitch.dynamics import (
    get_position_mask,
    get_velocity_mask,
    get_input_mask,
)

def get_working_space_constraints(
    horizon: int,
    n_states: int,
    n_robots: int,
    config: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the position constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        config: Configuration dictionary.

    Returns:
        Position constraints.
    """
    # ---- Box constraints ----
    # Position constraints.
    mask = get_position_mask(
        horizon,
        n_states,
        n_robots,
    )
    # TODO: get from config
    lb = -1
    ub = 1
    return lb * mask, ub * mask

def get_velocity_constraints(
    horizon: int,
    n_states: int,
    n_robots: int,
    config: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the velocity constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        config: Configuration dictionary.

    Returns:
        Velocity constraints.
    """
    # ---- Box constraints ----
    # Velocity constraints.
    mask = get_velocity_mask(
        horizon,
        n_states,
        n_robots,
    )
    # TODO: get from config
    lb = -jnp.inf
    ub = jnp.inf
    return lb * mask, ub * mask

def get_acceleration_constraints(
    horizon: int,
    n_states: int,
    n_robots: int,
    config: dict,
    compensation: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the acceleration constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        config: Configuration dictionary.

    Returns:
        Acceleration constraints.
    """
    if compensation is None:
        compensation = jnp.zeros((horizon * n_robots * n_states, 1))
    elif compensation.ndim != 1:
        raise ValueError("Compensation must be of shape (n_states,)")
    elif compensation.shape[0] != n_states:
        raise ValueError(
            "Compensation must match robot state size"
        )
    else:
        compensation = jnp.tile(
            compensation[:, None],
            (horizon * n_robots, 1),
        )

    # ---- Box constraints ----
    # TODO: allow second order cone constraints
    # Acceleration constraints.
    mask = get_input_mask(
        horizon,
        n_states,
        n_robots,
    )
    # TODO: get from config
    lb = -1
    ub = 1
    return lb * mask - compensation, ub * mask - compensation

def get_jerk_constraints(
    horizon: int,
    n_states: int,
    n_robots: int,
    config: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the jerk constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        config: Configuration dictionary.

    Returns:
        Jerk constraints.
    """
    # Affine inequality constraints.
    # TODO


def get_constraints(
    horizon: int,
    n_states: int,
    n_robots: int,
    config: dict,
) -> Project:
    """Compute the constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        config: Configuration dictionary.

    Returns:
        Constraints.
    """
    # ---- Box constraints ----
    # Note: at the moment, we can decouple all the constraints among the robots
    # TODO: allow to choose this in the config
    lb, ub = get_working_space_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        config=config,
    )
    _lb, _ub = get_velocity_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        config=config,
    )
    lb = jnp.maximum(lb,_lb,)
    ub = jnp.minimum(ub,_ub,)
    _lb, _ub = get_acceleration_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        config=config,
    )
    lb = jnp.maximum(lb,_lb)
    ub = jnp.minimum(ub,_ub)

    # TODO: add final state constraints if specified in config 
    # (needs enabling for variable box constraints)

    # TODO: check if using mask can improve efficiency
    box = BoxConstraint(
        lower_bound=lb,
        upper_bound=ub,
    )

    # ---- Affine inequality constraints ----
    C, lb, ub = get_jerk_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        config=config,
    )

    ineq = AffineInequalityConstraint(
        C=C,
        lb=lb,
        ub=ub,
    )

    # ---- Affine equality constraints ----
    # TODO: add initial state constraints
    # TODO: add final state constraints
    # TODO: add dynamics constraints
    eq = EqualityConstraint(
        A = A,
        b = b,
        var_b=True # We may need to solve for different initial and terminal states
    )


    # ---- Project ----
    # TODO: autotuning or get from config
    # TODO: perform matrix equilibration if specified in config
    project = Project(
        box_constraint=box,
        affine_inequality_constraint=ineq,
        affine_equality_constraint=eq,
    )

    return project