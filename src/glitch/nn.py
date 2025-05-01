import os
import pickle

import jax
import jax.numpy as jnp
from flax import linen as nn
from hcnn.project import Project

from glitch.utils import logger
from glitch.definitions.dynamics import FleetStateInput

class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""
    project: Project
    features_list: list
    unroll: bool # True if unrolling is used in the projection layer
    fpi: bool
    activation: nn.Module = nn.softplus

    @nn.compact
    def __call__(self, x, b, sigma, omega, n_iter=100, n_iter_bwd=100, raw=False):
        """Call the NN."""
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(self.project.dim)(x)
        if not raw:
            raise NotImplementedError(
                "Re-batching of the raw output is not implemented."
            )
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

def batch_to_input(
    initial_states_batched: FleetStateInput, 
    final_states_batched: FleetStateInput,
    n_eq: int,):
    """Convert the batch of initial and final states to the input format.

    Args:
        initial_states_batched (jax.Array): Batch of initial states.
        final_states_batched (jax.Array): Batch of final states.

    Returns:
        jax.Array: The input format for the model.
    """
    # initial_states_batched p, v are of dim (B, 1, n_robots, n_states)
    vmap_flatten = jax.vmap(lambda x: x.flatten())
    x_batch = jnp.concatenate((
        vmap_flatten(initial_states_batched),
        vmap_flatten(final_states_batched),
    ), axis=0)
    b_batch = jnp.concatenate((
        x_batch,
        jnp.zeros((x_batch.shape[0], n_eq - x_batch.shape[1], 1)),
    ))
    return x_batch, b_batch