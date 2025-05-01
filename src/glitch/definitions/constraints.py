"""Module returning the matrices for the constraints."""

import jax.numpy as jnp
from typing import Tuple

from hcnn.project import Project
from hcnn.constraints.box import BoxConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.affine_equality import EqualityConstraint

from glitch.definitions.dynamics import (
    get_position_mask,
    get_velocity_mask,
    get_input_mask,
    get_jerk_matrix,
    get_dynamics,
    get_initial_states_extractor,
    get_final_states_extractor,
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
    h: float,
    config: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the jerk constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        h: Time discretization.
        config: Configuration dictionary.

    Returns:
        Jerk constraints.
    """
    # Affine inequality constraints.
    C = get_jerk_matrix(horizon, n_states, n_robots, h)
    # TODO: get from config
    lb = -1 * jnp.ones((C.shape[0], 1))
    ub = 1 * jnp.ones((C.shape[0], 1))
    return C, lb, ub


def get_constraints(
    horizon: int,
    n_states: int,
    n_robots: int,
    h: float,
    config_constraints: dict = None,
) -> Project:
    """Compute the constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        h: Time discretization.
        config_constraints: Configuration dictionary.

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
        config=config_constraints,
    )
    _lb, _ub = get_velocity_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        config=config_constraints,
    )
    lb = jnp.maximum(lb,_lb,)
    ub = jnp.minimum(ub,_ub,)
    _lb, _ub = get_acceleration_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        config=config_constraints,
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
    ineq = None
    if horizon > 1:
        C, lb, ub = get_jerk_constraints(
            horizon=horizon,
            n_states=n_states,
            n_robots=1, # We can decouple the constraints among the robots
            h=h,
            config=config_constraints,
        )

        ineq = AffineInequalityConstraint(
            C=C,
            lb=lb,
            ub=ub,
        )

    # ---- Affine equality constraints ----
    A, B = get_dynamics(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        h=h,
    )
    A_eq_dynamics = jnp.concatenate((
        jnp.zeros((B.shape[0], B.shape[0] - (B.shape[1] + A.shape[1]))),
        A, 
        B
    ), axis=1) - jnp.eye(B.shape[0])
    A_eq = jnp.concatenate((
        get_initial_states_extractor(
            horizon=horizon,
            n_states=n_states,
            n_robots=1, # We can decouple the constraints among the robots
        ),
        get_final_states_extractor(
            horizon=horizon,
            n_states=n_states,
            n_robots=1, # We can decouple the constraints among the robots
        ),
        A_eq_dynamics,
    ), axis=0)
    eq = EqualityConstraint(
        A = A_eq,
        b = jnp.zeros((A.shape[0], 1)), # b is considered variable anyway
        method=None,
        var_b=True # We may need to solve for different initial and terminal states
    )

    # ---- Project ----
    if (
        config_constraints["autotuning"]
        or config_constraints["equilibration"]
    ):
        # TODO: autotuning
        # TODO: perform matrix equilibration if specified in config
        raise NotImplementedError(
            "Autotuning and equilibration are not implemented yet."
        )
    project = Project(
        box_constraint=box,
        affine_inequality_constraint=ineq,
        affine_equality_constraint=eq,
        unroll=config_constraints["unroll"],
    )

    return project