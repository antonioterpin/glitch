"""Data formats for the Glitch module."""

import jax.numpy as jnp
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class

from glitch.utils import JAX_DEBUG_JIT

@dataclass
@register_pytree_node_class
class FleetStateInput:
    """Fleet state data format.
    
    We follow the notation of 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6385823.
    """
    # Robots state
    v: jnp.ndarray # Robots velocity, shape (horizon, n_robots, n_states)
    p: jnp.ndarray # Robots position, shape (horizon, n_robots, n_states)
    # Robots input
    u: jnp.ndarray # Robots input, shape (horizon - 1, n_robots, n_states)

    def __init__(self, v: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray):
        """Initialize the robot state.
        
        Args:
            v: Robots velocities.
            p: Robots positions.
            u: Robots inputs.
        """
        if JAX_DEBUG_JIT:
            if v.ndim != p.ndim:
                raise ValueError("v and p must have the same number of dimensions.")
            if v.ndim != 3:
                raise ValueError("v and p must be (horizon, n_robots, n_states).")
            if v.shape[0] != p.shape[0]:
                raise ValueError("v and p must have the same horizon length.")
            if v.shape[1] != p.shape[1]:
                raise ValueError("v and p must have the same number of robots.")
            if v.shape[2] != p.shape[2]:
                raise ValueError("v and p must have the same number of states.")
            if u.ndim != 3:
                raise ValueError("u must be a (horizon - 1, n_robots, n_states) array.")
            if u.shape[0] != v.shape[0] - 1:
                raise ValueError("u must have one less horizon length than v and p.")
            if u.shape[1] != v.shape[1]:
                raise ValueError("u must have the same number of robots as v and p.")
            if u.shape[2] != v.shape[2]:
                raise ValueError("u must have the same number of states as v and p.")
        self.v = v
        self.p = p
        self.u = u

    def tree_flatten(self):
        """Flatten the FleetState object."""
        return (self.v, self.p, self.u), None
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the FleetState object."""
        v, p, u = children
        return cls(v, p, u)
    
    def flatten(self):
        """Get the flattened state of the fleet: [p1; p2; ...; v1; v2; ...].
        
        Returns:
            Flattened state of the fleet.
        """
        return jnp.concatenate((
            self.p.flatten(order="C"),
            self.v.flatten(order="C"),
            self.u.flatten(order='C')
        ), axis=0)[..., None]
    
    def replace(self, **kwargs):
        """Return a new state with specified fields replaced."""
        return FleetStateInput(
            v=kwargs.get('v', self.v),
            p=kwargs.get('p', self.p),
            u=kwargs.get('u', self.u)
        )
    
    def unpack(self, x: jnp.ndarray):
        """Get the fleet state from the flattened state over a horizon.
        
        Args:
            x: Flattened state of the fleet.

        Returns:
            Unflattened state of the fleet of size (horizon, n_robots, 2, n_states).
        """
        horizon, n_robots, n_states = self.p.shape

        p_flatten = x[:n_robots * n_states * horizon]
        v_flatten = x[n_robots * n_states * horizon:n_robots * n_states * 2 * horizon]
        u_flatten = x[n_robots * n_states * 2 * horizon:]

        return FleetStateInput(
            v=v_flatten.reshape((horizon, n_robots, n_states), order='C'),
            p=p_flatten.reshape((horizon, n_robots, n_states), order='C'),
            u=u_flatten.reshape((horizon - 1, n_robots, n_states), order='C')
        )
    
    @property
    def n_robots(self):
        """Number of robots in the fleet."""
        return self.p.shape[1]
    @property
    def n_states(self):
        """Number of states in the fleet."""
        return self.p.shape[2]
    
    @property
    def horizon(self):
        """Horizon (0 to Horizon) of the fleet."""
        return self.p.shape[0] - 1
    
def get_position_mask(
    horizon: int,
    n_robots: int,
    n_states: int,
):
    """Get the position mask of the fleet.

    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
    
    Returns:
        Position mask of the fleet.
    """
    return jnp.concatenate((
        # mask one position
        jnp.ones(((horizon + 1) * n_robots * n_states, 1)),
        # mask zero velocity
        jnp.zeros(((horizon + 1) * n_robots * n_states, 1)),
        # mask zero input
        jnp.zeros((horizon * n_robots * n_states, 1))
    ), axis=0)

def get_velocity_mask(
    horizon: int,
    n_robots: int,
    n_states: int,
):
    """Get the position mask of the fleet.

    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
    
    Returns:
        Position mask of the fleet.
    """
    return jnp.concatenate((
        # mask zero position
        jnp.zeros(((horizon + 1) * n_robots * n_states, 1)),
        # mask one velocity
        jnp.ones(((horizon + 1) * n_robots * n_states, 1)),
        # mask zero input
        jnp.zeros((horizon * n_robots * n_states, 1))
    ), axis=0)
    
def get_input_mask(
    horizon: int,
    n_robots: int,
    n_states: int,
):
    """Get the input mask of the fleet.

    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
    
    Returns:
        Input mask of the fleet.
    """
    return jnp.concatenate((
        # mask zero for states
        jnp.zeros(((horizon + 1) * n_robots * n_states * 2, 1)),
        # mask one input
        jnp.ones((horizon * n_robots * n_states, 1)),
    ), axis=0)
    

def get_dynamics(
    horizon: int,
    n_robots: int,
    n_states: int,
    h: float):
    """Get the dynamics of the fleet.
    
    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
        h: Time discretization.
    """

    A = jnp.concatenate((
        jnp.concatenate((
            jnp.eye(n_states * n_robots),
            h * jnp.eye(n_states * n_robots),
        ), axis=1),
        jnp.concatenate((
            jnp.zeros((n_states * n_robots, n_states * n_robots)),
            jnp.eye(n_states * n_robots),
        ), axis=1)
    ), axis=0)
    B = jnp.concatenate((
        h ** 2 / 2 * jnp.eye(n_states * n_robots),
        h * jnp.eye(n_states * n_robots),
    ), axis=0)

    A_horizon = jnp.block([[
        jnp.linalg.matrix_power(A, i + 1)
    ] for i in range(horizon)])
    B_horizon = jnp.concatenate([
        jnp.concatenate([
            jnp.linalg.matrix_power(A, i - k) @ B if k <= i else jnp.zeros_like(B)
            for k in range(horizon)
        ], axis=1)
        for i in range(horizon)
    ], axis=0)
    
    return A_horizon, B_horizon

def get_initial_states_extractor(
    horizon: int,
    n_robots: int,
    n_states: int,
) -> jnp.ndarray:
    """Get the A matrix for initial and final constraints of the fleet.
    
    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
    
    Returns:
        Initial and final constraints of the fleet.
    """
    initial_p = jnp.concatenate((
        jnp.eye(n_states * n_robots), # initial p
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # other p
        jnp.zeros((n_states * n_robots, n_states * n_robots * (horizon + 1))), # v
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # u
    ), axis=1)
    initial_v = jnp.concatenate((
        jnp.zeros((n_states * n_robots, n_states * n_robots * (horizon + 1))), # p
        jnp.eye(n_states * n_robots), # initial v
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # other v
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # u
    ), axis=1)
    return jnp.concatenate((
        initial_p,
        initial_v,
    ), axis=0)

def get_final_states_extractor(
    horizon: int,
    n_robots: int,
    n_states: int,
) -> jnp.ndarray:
    """Get the A matrix for initial and final constraints of the fleet.
    
    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
    
    Returns:
        Initial and final constraints of the fleet.
    """
    final_p = jnp.concatenate((
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # other p
        jnp.eye(n_states * n_robots), # final p
        jnp.zeros((n_states * n_robots, n_states * n_robots * (horizon + 1))), # v
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # u
    ), axis=1)
    final_v = jnp.concatenate((
        jnp.zeros((n_states * n_robots, n_states * n_robots * (horizon + 1))), # p
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # other v
        jnp.eye(n_states * n_robots), # final v
        jnp.zeros((n_states * n_robots, n_states * n_robots * horizon)), # u
    ), axis=1)

    return jnp.concatenate((
        final_p,
        final_v,
    ), axis=0)

def get_input_extractor(
    horizon: int,
    n_robots: int,
    n_states: int,
) -> jnp.ndarray:
    """Get the A matrix for input constraints of the fleet.
    
    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
    
    Returns:
        Input constraints of the fleet.
    """
    return jnp.concatenate((
        # p
        jnp.zeros((horizon * n_robots * n_states, (horizon + 1) * n_robots * n_states)),
        # v
        jnp.zeros((horizon * n_robots * n_states, (horizon + 1) * n_robots * n_states)),
        # u
        jnp.eye(horizon * n_robots * n_states),
    ), axis=1)


def get_dynamics_outputs_extractor(
    horizon: int,
    n_robots: int,
    n_states: int,
) -> jnp.ndarray:
    """Get the A matrix for dynamics outputs of the fleet.
    
    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
    
    Returns:
        Dynamics outputs of the fleet.
    """
    extract_ps = jnp.concatenate((
        # first p -> skip
        jnp.zeros((horizon * n_robots * n_states, n_robots * n_states)),
        # other p
        jnp.eye(horizon * n_robots * n_states),
        # v
        jnp.zeros((horizon * n_robots * n_states, (horizon + 1) * n_robots * n_states)),
        # u
        jnp.zeros((horizon * n_robots * n_states, horizon * n_robots * n_states)),
    ), axis=1)
    extract_vs = jnp.concatenate((
        # p
        jnp.zeros((horizon * n_robots * n_states, (horizon + 1) * n_robots * n_states)),
        # first v -> skip
        jnp.zeros((horizon * n_robots * n_states, n_robots * n_states)),
        # other v
        jnp.eye(horizon * n_robots * n_states),
        # u
        jnp.zeros((horizon * n_robots * n_states, horizon * n_robots * n_states)),
    ), axis=1)
    return jnp.concatenate((
        extract_ps,
        extract_vs,
    ), axis=0)

def get_jerk_matrix(
    horizon: int,
    n_robots: int,
    n_states: int,
    h: float,
) -> jnp.ndarray:
    """Get the jerk matrix of the fleet.
    
    Args:
        horizon: Time horizon.
        n_robots: Number of robots.
        n_states: Number of states.
        h: Time discretization.

    Returns:
        Jerk matrix of the fleet.
    """
    if JAX_DEBUG_JIT:
        if horizon < 2:
            raise ValueError("Horizon must be greater than 1.")
    n_inputs = n_states * n_robots
    J = 1 / h * (
        -jnp.eye((horizon - 1) * n_inputs, n_inputs * horizon)
        + jnp.eye((horizon - 1) * n_inputs, n_inputs * horizon, k = n_inputs)
    )
    # ((horizon - 1) * n_inputs, N)
    # N = (horizon + 1) * n_states * n_robots * 2 + horizon * n_inputs
    return J @ get_input_extractor(horizon, n_robots, n_states)