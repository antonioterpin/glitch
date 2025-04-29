"""Data formats for the Glitch module."""

import jax.numpy as jnp
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class

from glitch.common import JAX_DEBUG_JIT

@dataclass
@register_pytree_node_class
class FleetState:
    """Fleet state data format.
    
    We follow the notation of 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6385823.
    """

    # Robots state
    v: jnp.ndarray # Robots velocity, shape (n_robots, n_states)
    p: jnp.ndarray # Robots position, shape (n_robots, n_states)

    def __init__(self, v: jnp.ndarray, p: jnp.ndarray):
        """Initialize the robot state.
        
        Args:
            v: Robots velocities.
            p: Robots positions.
        """
        if v.ndim != p.ndim:
            raise ValueError("v and p must have the same number of dimensions.")
        if v.ndim != 2:
            raise ValueError("v and p must be 2D arrays.")
        if v.shape[0] != p.shape[0]:
            raise ValueError("v and p must have the same number of robots.")
        if v.shape[1] != p.shape[1]:
            raise ValueError("v and p must have the same number of states.")
        self.v = v
        self.p = p

    def tree_flatten(self):
        """Flatten the FleetState object."""
        return (self.v, self.p), None
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the FleetState object."""
        v, p = children
        return cls(v, p)
    
    def get_dynamics(self, h: float):
        """Get the dynamics of the fleet.
        
        Args:
            h: Time discretization.
        """
        n_robots, n_states = self.v.shape

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

        return A, B
    
    def get_flatten_state(self):
        """Get the flattened state of the fleet: [p1; p2; ...; v1; v2; ...].
        
        Returns:
            Flattened state of the fleet.
        """
        return jnp.concatenate((
            self.p.flatten(order="C"), # row-major
            self.v.flatten(order="C")  # row-major
        ), axis=0)[..., None]
    
    def replace(self, **kwargs):
        """Return a new state with specified fields replaced."""
        return FleetState(
            v=kwargs.get('v', self.v),
            p=kwargs.get('p', self.p)
        )
    
    def unpack_flatten(self, x: jnp.ndarray):
        """Get the fleet state from the flattened state.
        
        Args:
            x: Flattened state of the fleet.
        """
        n_robots, n_states = self.v.shape
        x = self.get_unflatten_horizon(x)[0, ...]
        return self.replace(
            p=jnp.concatenate(x[:, 0, ...], axis=0).reshape(
                (n_robots, n_states), order='C'),
            v=jnp.concatenate(x[:, 1, ...], axis=0).reshape(
                (n_robots, n_states), order='C'),
        )
    
    def get_unflatten_horizon(self, x: jnp.ndarray):
        """Get the fleet state from the flattened state over a horizon.
        
        Args:
            x: Flattened state of the fleet.

        Returns:
            Unflattened state of the fleet of size (horizon, n_robots, 2, n_states).
        """
        n_robots, n_states = self.v.shape
        horizon = x.shape[0] // (n_robots * n_states * 2)
        return jnp.array([
            [
                [
                    # position
                    x[
                        t * n_robots * n_states * 2 \
                            + i * n_states
                        :
                        t * n_robots * n_states * 2 \
                            + (i + 1) * n_states
                    ],
                    # velocity
                    x[
                        t * n_robots * n_states * 2 + \
                            + n_robots * n_states \
                            + i * n_states
                        :
                        t * n_robots * n_states * 2 + \
                            + n_robots * n_states \
                            + (i + 1) * n_states
                    ]
                ]
                for i in range(n_robots)
            ]
            for t in range(horizon)
        ])
    
    def get_dynamics_over_horizon(self, h: float, N: int):
        """Get the dynamics of the fleet over a horizon.
        
        Args:
            h: Time discretization.
            N: Number of time steps.
        """
        assert N > 0, "N must be greater than 0."
        A, B = self.get_dynamics(h)

        A_horizon = jnp.block([[
            jnp.linalg.matrix_power(A, i + 1)
        ] for i in range(N)])
        B_horizon = jnp.concatenate([
            jnp.concatenate([
                jnp.linalg.matrix_power(A, i - k) @ B if k <= i else jnp.zeros_like(B)
                for k in range(N)
            ], axis=1)
            for i in range(N)
        ], axis=0)
        
        return A_horizon, B_horizon
    
    def get_flatten_input(self, u: jnp.ndarray):
        """Get the flattened input of the fleet: [u1; u2; ...].
        
        Args:
            u: Input of the fleet.
        """
        assert u.ndim == 2, "u must be a 2D array."
        assert u.shape[0] == self.v.shape[0], "u must have the same number of robots."
        assert u.shape[1] == self.v.shape[1], "u must have the same number of states."

        return self.get_flatten_input_horizon(u[None, ...])

    def get_flatten_input_horizon(self, u: jnp.ndarray):
        """Get the flattened input of the fleet over a horizon.
        
        Args:
            u: Input of the fleet.
        """
        assert u.ndim == 3, "u must be a 2D array."
        assert u.shape[1] == self.v.shape[0], "u must have the same number of robots."
        assert u.shape[2] == self.v.shape[1], "u must have the same number of states."

        horizon, n_robots, n_states = u.shape

        # Flatten the input (row-major)
        return u.reshape((horizon * n_robots * n_states, 1), order='C')

    def get_unflatten_input(self, u: jnp.ndarray):
        """Get the unflattened input of the fleet.
        
        Args:
            u: Input of the fleet.
        """
        n_robots, n_states = self.v.shape

        if JAX_DEBUG_JIT:
            assert u.ndim == 2, "u must be a 2D array."
            assert (u.shape[0] % (n_robots * n_states)) == 0, (
                "u must contain an input for each robot and state."
            )
            assert u.shape[1] == 1, "u must be a flatten input."

        n_horizon = u.shape[0] // (n_robots * n_states)

        return u.reshape((n_horizon, n_robots, n_states), order='C')
        

