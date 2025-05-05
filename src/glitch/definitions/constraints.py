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
    get_input_extractor,
    get_dynamics_outputs_extractor,
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
        horizon=horizon,
        n_states=n_states,
        n_robots=n_robots,
    )
    lb = config.get("lower_bound", -1)
    ub = config.get("upper_bound", 1)
    lower_bound = jnp.where(mask == 1, lb, -jnp.inf)
    upper_bound = jnp.where(mask == 1, ub, jnp.inf)
    return lower_bound, upper_bound

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
        horizon=horizon,
        n_states=n_states,
        n_robots=n_robots,
    )
    lb = config.get("lower_bound", -jnp.inf)
    ub = config.get("upper_bound", jnp.inf)
    lower_bound = jnp.where(mask == 1, lb, -jnp.inf)
    upper_bound = jnp.where(mask == 1, ub, jnp.inf)
    return lower_bound, upper_bound

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
        compensation = jnp.zeros((horizon * n_robots * n_states,))
    elif compensation.ndim != 1:
        raise ValueError("Compensation must be of shape (n_states,)")
    elif compensation.shape[0] != n_states:
        raise ValueError(
            "Compensation must match robot state size"
        )
    else:
        compensation = jnp.tile(
            compensation,
            (horizon * n_robots,),
        )

    # ---- Box constraints ----
    # TODO: allow second order cone constraints
    # Acceleration constraints.
    mask = get_input_mask(
        horizon=horizon,
        n_states=n_states,
        n_robots=n_robots,
    )
    lb = config.get("lower_bound", -1)
    ub = config.get("upper_bound", 1)
    lb_compensated = jnp.ones((n_states * n_robots * horizon,)) * lb - compensation
    ub_compensated = jnp.ones((n_states * n_robots * horizon,)) * ub - compensation
    lower_bound = (-jnp.inf * (1 - mask)).at[mask == 1].set(lb_compensated)
    upper_bound = (jnp.inf * (1 - mask)).at[mask == 1].set(ub_compensated)
    return lower_bound, upper_bound

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
    C = get_jerk_matrix(
        horizon=horizon,
        n_states=n_states, 
        n_robots=n_robots, 
        h=h
    )
    lb = config.get("lower_bound", -1) * jnp.ones((C.shape[0], 1))
    ub = config.get("upper_bound", 1) * jnp.ones((C.shape[0], 1))
    return C, lb, ub


def get_constraints(
    horizon: int,
    n_states: int,
    n_robots: int,
    h: float,
    input_compensation: jnp.ndarray,
    config_constraints: dict,
    config_hcnn: dict,
) -> Project:
    """Compute the constraints.

    Args:
        horizon: Time horizon.
        n_states: Number of states.
        n_robots: Number of robots.
        h: Time discretization.
        config_constraints: Configuration dictionary.
        autotune: Whether to autotune the constraints.
        equilibration: Whether to perform matrix equilibration.

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
        config=config_constraints["working_space"],
    )
    _lb, _ub = get_velocity_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        config=config_constraints["velocity"],
    )
    lb = jnp.maximum(lb,_lb,)
    ub = jnp.minimum(ub,_ub,)
    _lb, _ub = get_acceleration_constraints(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        compensation=input_compensation,
        config=config_constraints["acceleration"],
    )
    lb = jnp.maximum(lb,_lb)
    ub = jnp.minimum(ub,_ub)

    # TODO: add initial and final state constraints if specified in config 
    # (needs enabling for variable box constraints)

    # TODO: check if using mask can improve efficiency
    box = BoxConstraint(
        lower_bound=lb[None, ...],
        upper_bound=ub[None, ...],
    )

    # ---- Affine inequality constraints ----
    ineq = None
    if horizon > 1:
        C, lb, ub = get_jerk_constraints(
            horizon=horizon,
            n_states=n_states,
            n_robots=1, # We can decouple the constraints among the robots
            h=h,
            config=config_constraints["jerk"],
        )

        ineq = AffineInequalityConstraint(
            C=C[None, ...],
            lb=lb[None, ...],
            ub=ub[None, ...],
        )

    # ---- Affine equality constraints ----
    A_initial_states = get_initial_states_extractor(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
    )
    A_final_states = get_final_states_extractor(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
    )
    A_inputs = get_input_extractor(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
    )
    A, B = get_dynamics(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
        h=h,
    )
    A_dynamics_outputs = get_dynamics_outputs_extractor(
        horizon=horizon,
        n_states=n_states,
        n_robots=1, # We can decouple the constraints among the robots
    )
    A_eq = jnp.concatenate((
        A_initial_states,
        A_final_states,
        A @ A_initial_states + B @ A_inputs - A_dynamics_outputs,
    ), axis=0)
    eq = EqualityConstraint(
        A = A_eq[None, ...],
        b = jnp.zeros((1, A_eq.shape[0], 1)), # b is considered variable anyway
        method=None,
        var_b=True # We may need to solve for different initial and terminal states
    )

    # ---- Project ----
    if (
        config_hcnn["autotuning"]
        or config_hcnn["equilibration"]
    ):
        # TODO: autotuning
        # TODO: perform matrix equilibration if specified in config
        raise NotImplementedError(
            "Autotuning and equilibration are not implemented yet."
        )
    project = Project(
        box_constraint=box,
        ineq_constraint=ineq,
        eq_constraint=eq,
        unroll=config_hcnn["unroll"],
    )

    return project