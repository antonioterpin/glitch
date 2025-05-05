import os
import pickle

import jax
import jax.numpy as jnp
from flax import linen as nn
from hcnn.project import Project

from glitch.utils import logger
from glitch.definitions.dynamics import FleetStateInput
from glitch.definitions.dynamics import get_dynamics

class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""
    project: Project
    features_list: list
    unroll: bool # True if unrolling is used in the projection layer
    fpi: bool
    fsu: FleetStateInput # Dummy FleetStateInput for dimensions
    activation: nn.Module = nn.softplus

    @nn.compact
    def __call__(
        self, 
        initial_states_batched, 
        final_states_batched, 
        sigma, 
        omega, 
        n_iter=100, 
        n_iter_bwd=100, 
        raw=False
    ):
        """Call the NN."""
        x0 = jax.vmap(lambda x: x.flatten())(initial_states_batched)
        x = (jax.vmap(lambda x: x.flatten())(final_states_batched) - x0).squeeze(-1)
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        # The output of the MLP are the inputs
        n_inputs = self.fsu.n_states * self.fsu.n_robots * self.fsu.horizon
        u = nn.Dense(n_inputs)(x)[..., None]
        A, B = get_dynamics(
            horizon=self.fsu.horizon,
            n_robots=self.fsu.n_robots,
            n_states=self.fsu.n_states,
            h=0.5,
        )
        x_all = jax.vmap(lambda _x, _u: A @ _x + B @ _u)(x0, u)
        p_dim = self.fsu.n_states * self.fsu.n_robots * self.fsu.horizon
        p_all = x_all[:, :p_dim, :]
        v_all = x_all[:, p_dim:, :]
        p0 = x0[:, :self.fsu.n_states * self.fsu.n_robots, :]
        v0 = x0[:, self.fsu.n_states * self.fsu.n_robots:, :]
        print(f"{A.shape=}")
        print(f"{B.shape=}")
        print(f"{x_all.shape=}")
        print(f"{u.shape=}")
        print(f"{x0.shape=}")
        print(f"{p0.shape=}")
        print(f"{v0.shape=}")
        p_all = jnp.concatenate((p0, p_all), axis=(1))
        v_all = jnp.concatenate((v0, v_all), axis=(1))
        print(f"{p_all.shape=}")
        print(f"{v_all.shape=}")
        x_all = jnp.concatenate((p_all, v_all, u), axis=1)
        x = jax.vmap(lambda _x: self.fsu.unpack(_x))(x_all)

        # # TODO: allow the projection to be coupled
        # x = nn.Dense(self.project.dim * self.fsu.n_robots)(x)
        # x = jax.vmap(self.fsu.unpack)(x)
        if not raw:
            n_eq = self.project.eq_constraint.n_constraints
            batch_size = initial_states_batched.p.shape[0]
            # TODO: allow the projection to be coupled

            # rebatch so that the projection is for each robot
            b = prepare_b_from_batch(n_eq, initial_states_batched, final_states_batched)
            
            # next, we prepare the x for the projection layer
            x = predictions_to_projection_layer_format(x)
            init = self.project.get_init(x)
            if self.unroll:
                x = self.project.call(
                    init, 
                    x, 
                    b, 
                    interpolation_value=0.0, 
                    sigma=sigma,
                    omega=omega,
                    n_iter=n_iter)[0]
            else:
                x = self.project.call(
                    init,
                    x,
                    b,
                    interpolation_value=0.0,
                    sigma=sigma, 
                    omega=omega,
                    n_iter=n_iter,
                    n_iter_bwd=n_iter_bwd,
                    fpi=self.fpi,
                )[0]
            
            # we need to undo the rebatching
            x = projection_layer_format_to_predictions(
                x, 
                batch_size=batch_size,
                n_robots=self.fsu.n_robots,
                horizon=self.fsu.horizon,
                n_states=self.fsu.n_states
            )

        # x is now shape (batch_size, horizon(+1), n_robots, n_states)
        return x

def save_model(
    params: jax.Array,
    out_dir: str,
    model_name: str,
):
    """Save the model state to a file.

    Args:
        trainable_state (jax.Array): The trainable state of the model.
        out_dir (str): Directory where to save the model.
        model_name (str): Name of the model.
    """
    with open(os.path.join(out_dir, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(params, f)
    logger.info(f"Model saved at {out_dir}/{model_name}.pkl")


def load_model(
    trainable_state_path: str,
) -> jax.Array:
    """Load the model state from a file.

    Args:
        trainable_state_path (str): Path to the model state file.

    Returns:
        jax.Array: The loaded trainable state of the model.
    """
    with open(trainable_state_path, "rb") as f:
        trainable_state = pickle.load(f)
    logger.info(f"Model loaded from {trainable_state_path}.pkl")
    return trainable_state

def batch_robots(fleet_state: FleetStateInput) -> FleetStateInput:
    """Rebatch the fleet state to have the robots as the first dimension.

    Args:
        fleet_state (FleetStateInput): The fleet state to rebatch.

    Returns:
        FleetStateInput: The rebatch fleet state.
    """
    # (n_robots, horizon, 1, n_states)
    rebatched = jax.vmap(
        # (horizon, n_states)
        lambda x: FleetStateInput(
            p=x.p[:, None, ...],
            v=x.v[:, None, ...],
            u=x.u[:, None, ...],
        ),
        in_axes=1,
        out_axes=0
    )(fleet_state) # (horizon, n_robots, n_states)
    return rebatched

def unbatch_robots(fleet_state: jnp.ndarray, horizon: int, n_states: int) -> FleetStateInput:
    """Unbatch the fleet state so that robots are with the fleet.

    Args:
        fleet_state (jnp.ndarray): The fleet state to unbatch.
            (batch_size, n_robots, n_states)

    Returns:
        FleetStateInput: The rebatch fleet state.
    """
    fsu_mock = FleetStateInput(
        p=jnp.zeros((horizon + 1, 1, n_states)),
        v=jnp.zeros((horizon + 1, 1, n_states)),
        u=jnp.zeros((horizon + 1, 1, n_states)),
    )
    def body_fun(x):
        # x is of shape (n_robots, dim_flatten)
        # fsu_mock.unpack(r) is of shape (horizon( + 1), 1, n_states)
        # robots_unpacked is of shape (n_robots, horizon( + 1), 1, n_states)
        robots_unpacked = jax.vmap(lambda r: fsu_mock.unpack(r))(x)
        ps = jnp.concatenate(jax.vmap(lambda r: r.p)(robots_unpacked), axis=1)
        vs = jnp.concatenate(jax.vmap(lambda r: r.v)(robots_unpacked), axis=1)
        us = jnp.concatenate(jax.vmap(lambda r: r.u)(robots_unpacked), axis=1)
        return FleetStateInput(
            p=ps,
            v=vs,
            u=us,
        )

    return jax.vmap(body_fun)(fleet_state)

def vmap_flatten(x_batched, y_batched):
    """Flatten the input and output of the model.

    Args:
        x_batched (FleetStateInput): The initial states.
        y_batched (FleetStateInput): The final states.

    Returns:
        jnp.ndarray: The flattened input and output.
    """
    return jax.vmap(lambda x, y: jnp.concatenate(
        (x.flatten(), y.flatten()), axis=0),
        in_axes=(0, 0), out_axes=0
    )(
        x_batched,
        y_batched
    )

def predictions_to_projection_layer_format(x):
    print(f"predictions shape {x.p.shape=}")
    # x is now shape (batch_size, horizon(+1), n_robots, n_states),
    # we reshape it as for b1
    # x_list is (batch_size, n_robots, horizon(+1) * n_states, 1)
    x_list = jax.vmap(
        # robots_trajectory: (horizon(+1), n_robots, n_states)
        lambda robots_trajectory: jax.vmap(
            # robot_trajectory: (horizon(+1), 1, n_states)
            # flatten: (horizon(+1) * n_states, 1)
            lambda robot_trajectory: robot_trajectory.flatten()
        )(
            # shape (n_robots, horizon(+1), 1, n_states)
            batch_robots(robots_trajectory)
        ))(x)
    # x is now shape (batch_size * n_robots, horizon(+1) * n_states, 1)
    x = jnp.concatenate(x_list, axis=0)
    x = x.squeeze(-1) # remove last dimension to fit the projection layer
    print(f"prediction layer format {x.shape=}")
    return x

def projection_layer_format_to_predictions(x, batch_size, n_robots, horizon, n_states):
    x = x.reshape(batch_size, n_robots, -1)
    x = unbatch_robots(x, horizon, n_states)
    return x

def prepare_b_from_batch(n_eq, initial_states_batched, final_states_batched):
    # NOTE: All of the following is to decouple from how the fsu is flattened
    # initial/final_states_batched: (batch_size, 1, n_robots, n_states)
    # batch_robots(x/y) is (n_robots, 1, 1, n_states), 
    # x is (1, n_robots, n_states)
    # vmap_flatten(n_robots, n_states)
    # b1_list is (batch_size, n_robots, n_states * 2 * 2, 1)
    b1_list = jax.vmap(
        # initial_states are (1, n_robots, n_states)
        lambda initial_states, final_states: vmap_flatten(
            batch_robots(initial_states), 
            batch_robots(final_states)),
        in_axes=(0, 0), 
        out_axes=0
    )(initial_states_batched, final_states_batched)
    # b1 is (batch_size * n_robots, n_states * 2 * 2, 1)
    b1 = jnp.concatenate(b1_list, axis=0)
    # add zeros to the second channel of b1 to match n_eq
    # so b has shape (batch_size * n_robots, n_eq, 1)
    return jnp.concatenate(
        (b1, jnp.zeros((b1.shape[0], n_eq - b1.shape[1], 1))), 
        axis=1
    )