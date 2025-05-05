import argparse
import datetime
import os
import time

import jax.flatten_util
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from tqdm import tqdm
import optax
from flax.training import train_state
import matplotlib.pyplot as plt

from glitch.plotting import plot_trajectory
from glitch.definitions.constraints import get_constraints
from glitch.nn import (
    HardConstrainedMLP, 
    load_model, 
    save_model, 
    predictions_to_projection_layer_format,
    prepare_b_from_batch
)
from glitch.dataloader import TransitionsDataset, create_dataloaders as load_dataset
from glitch.utils import load_configuration, GracefulShutdown, Logger
import glitch.definitions.preferences as preferences
from glitch.utils import logger


jax.config.update("jax_enable_x64", True)

def build_batched_objective(config_hcnn, config_problem):
    collision_penalty_fn_name = config_hcnn["collision_penalty_fn"]
    h = config_problem["h"]
    try:
        collision_penalty_fn = getattr(preferences, collision_penalty_fn_name)
    except AttributeError:
        raise ValueError(f"Unknown collision penalty '{collision_penalty_fn_name}'")
    compensation = jnp.array(config_problem["gravity"])
    
    def batched_objective(predictions, initial_states, final_states):
        return (
            preferences.input_effort(predictions, compensation, h) 
            +
            0.05 * preferences.reward_2d_single_agent(predictions)
            # + collision_penalty_fn(
            #     predictions, 
            #     config_hcnn["collision_penalty"], 
            #     config_hcnn["collision_normalization_factor"]
            # )
        )
    return jax.vmap(batched_objective)

def build_steps(project, config_hcnn, config_problem):
    """Build the training and evaluation step functions."""
    batched_objective = build_batched_objective(config_hcnn, config_problem)
    sigma, omega, n_iter_train, n_iter_test, n_iter_bwd = (
        config_hcnn["sigma"],
        config_hcnn["omega"],
        config_hcnn["n_iter_train"],
        config_hcnn["n_iter_test"],
        config_hcnn["n_iter_bwd"],
    )

    def train_step(state, initial_states, final_states):
        """Run a single training step."""

        def loss_fn(params):
            predictions = state.apply_fn(
                {"params": params}, 
                initial_states_batched=initial_states,
                final_states_batched=final_states,
                sigma=sigma,
                omega=omega,
                n_iter=n_iter_train, 
                n_iter_bwd=n_iter_bwd,
            )
            return batched_objective(predictions, initial_states, final_states).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        # Turn the pytree into one 1-D vector:
        grads_flat, _ = jax.flatten_util.ravel_pytree(grads)

        # Compute the L2 norm:
        grad_norm = jnp.linalg.norm(grads_flat)

        return loss, state.apply_gradients(grads=grads), grad_norm

    def eval_step(state, initial_states, final_states):
        predictions = state.apply_fn(
            {"params": state.params}, 
            initial_states_batched=initial_states, 
            final_states_batched=final_states,
            sigma=sigma,
            omega=omega,
            n_iter=n_iter_test, 
            n_iter_bwd=n_iter_bwd,
        )

        accuracy = batched_objective(predictions, initial_states, final_states).mean()
        # During training, we report the average constraint violation
        x = predictions_to_projection_layer_format(predictions)
        n_eq = project.eq_constraint.n_constraints
        b = prepare_b_from_batch(n_eq, initial_states, final_states)
        project.eq_constraint.b = b
        cv = project.cv(x).mean()

        return accuracy, cv, predictions
    # return train_step, eval_step
    return jax.jit(train_step), jax.jit(eval_step)

def load_hcnn(project, config_hcnn, config_problem):
    """Load the HCNN model based on the configuration.
    
    Args:
        config (dict): Configuration dictionary containing HCNN parameters.
    """
    try:
        activation = getattr(nn, config_hcnn["activation"])
    except AttributeError:
        raise ValueError(f"Unknown activation: {config_hcnn['activation']}")
    
    # Dummy fsu
    p = jnp.zeros((
        config_problem["horizon"] + 1, 
        config_problem["n_robots"], 
        config_problem["n_states"]
    ))
    v = jnp.zeros((
        config_problem["horizon"] + 1, 
        config_problem["n_robots"], 
        config_problem["n_states"]
    ))
    u = jnp.zeros((
        config_problem["horizon"], 
        config_problem["n_robots"], 
        config_problem["n_states"]
    ))
    fsu = preferences.FleetStateInput(p=p, v=v, u=u)

    return HardConstrainedMLP(
        project=project,
        fsu=fsu,
        features_list=config_hcnn["features"],
        fpi=config_hcnn["fpi"],
        unroll=config_hcnn["unroll"],
        activation=activation,
    )

def argument_parser():
    parser = argparse.ArgumentParser(description="A simple argument parser.")
    parser.add_argument(
        "--config-dataset",
        type=str,
        default="configs/dataset.yaml",
        help="Path to the dataset configuration file.",
    )

    parser.add_argument(
        "--config-hcnn",
        type=str,
        default="configs/hcnn.yaml",
        help="Path to the hcnn configuration file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/results/{timestamp}",
        help="Directory to save the results.",
    )

    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="Path to the trained model to load.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save the model every save_every training batches.",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Evaluate the model every eval_every training batches.",
    )

    parser.add_argument(
        "--plot-trajectory",
        type=int,
        nargs="+",
        help="Plot the trajectories with the given indices.",
    )

    args = parser.parse_args()

    return args

def train_hcnn(
    train_step: callable,
    eval_step: callable,
    state: train_state.TrainState,
    dataset_training: TransitionsDataset,
    dataset_validation: TransitionsDataset,
    save_every: int,
    eval_every: int,
    output_dir: str,
    run_name: str,
):
    """Train the HCNN model."""
    eval_initial_states, eval_final_states = dataset_validation[0]
    validation_loss = None
    validation_cv = None
    with (
        GracefulShutdown("Stop detected, finishing epoch...") as g,
        Logger(run_name) as data_logger,
    ):
        for step in (pbar := tqdm(range(len(dataset_training)))):
            if g.stop:
                break
            initial_states, final_states = dataset_training[step]
            t = time.time()
            loss, state, grad_norm = train_step(
                state, 
                initial_states,
                final_states,
            )
            t = time.time() - t
            data_logger.log(step, {
                "loss": loss,
                "batch_training_time": t,
                "grads_norms": grad_norm,
            })

            if step % eval_every == 0:
                validation_loss, validation_cv, _ = eval_step(
                    state, 
                    eval_initial_states, 
                    eval_final_states
                )
                data_logger.log(step, {
                    "validation_objective": validation_loss,
                    "validation_constraint_violation": validation_cv,
                })

            if output_dir is not None and step % save_every == 0:
                save_model(
                    state.params,
                    output_dir,
                    f"{run_name}_{step}",
                )

            pbar.set_description(
                f"Loss: {loss:.4f}, "
                f"Validation Loss: {validation_loss:.4f}, "
                f"Validation CV: {validation_cv:.4f}, "
                f"Grad Norm: {grad_norm:.4f}.")

    return state


if __name__ == "__main__":
    args = argument_parser()

    # Load the dataset
    config_dataset = load_configuration(args.config_dataset)
    if config_dataset is None:
        raise ValueError(f"Configuration file not found or empty: {args.config_hcnn}.")
    (
        dataset_training,
        dataset_validation,
        dataset_test,
    ) = load_dataset(config_dataset)
    
    # Load the HCNN configuration
    config_hcnn = load_configuration(args.config_hcnn)
    if config_hcnn is None:
        raise ValueError(f"Configuration file not found or empty: {args.config_hcnn}.")
    
    # Prepare the constraints
    project = get_constraints(
        horizon=config_dataset["problem"]["horizon"],
        n_robots=1, # TODO: Allow the option of coupling the projection
        n_states=config_dataset["problem"]["n_states"],
        h=config_dataset["problem"]["h"],
        input_compensation=jnp.array(config_dataset["problem"]["gravity"]),
        config_constraints=config_dataset["problem"]["constraints"],
        config_hcnn=config_hcnn,
    )
    hcnn = load_hcnn(project, config_hcnn, config_dataset["problem"])

    # Possibly load a pre-trained model
    trainable_state = None
    if args.load_from is not None:
        trainable_state = load_model(args.load_from)

    if trainable_state is not None:
        print(f"Loaded parameters from {args.load_from}")
    else:
        print("No parameters loaded. Initializing the network parameters from scratch.")
        # Initialize the parameters
        initial_states, final_states = dataset_training[0]
        trainable_state = hcnn.init(
            jax.random.PRNGKey(config_hcnn["seed"]),
            initial_states_batched=initial_states, 
            final_states_batched=final_states,
            sigma=config_hcnn["sigma"],
            omega=config_hcnn["omega"],
            n_iter=2,
        )

    
    # In any case, we re-initialize the train state
    state = train_state.TrainState.create(
        apply_fn=hcnn.apply,
        params=trainable_state["params"],
        tx=optax.adam(config_hcnn["learning_rate"]),
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = None
    if args.save_every > 0:
        # Create the output directory if it doesn't exist
        output_dir = args.output_dir.format(timestamp=timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved in: {output_dir}")

    train_step, eval_step = build_steps(
        project=hcnn.project,
        config_hcnn=config_hcnn,
        config_problem=config_dataset["problem"],
    )

    run_name = f"{config_hcnn['name']}_{config_dataset['name']}_{timestamp}"

    if args.train > 0:
        training_time_start = time.time()
        state = train_hcnn(
            train_step=train_step,
            eval_step=eval_step,
            state=state,
            dataset_training=dataset_training,
            dataset_validation=dataset_validation,
            save_every=args.save_every,
            eval_every=args.eval_every,
            output_dir=output_dir,
            run_name=run_name,
        )
        training_time = time.time() - training_time_start
        print(f"Training time: {training_time:.5f} seconds")

    # Evaluate the (trained) model on the test set
    eval_initial_states, eval_final_states = dataset_test[0]
    with (
        Logger(run_name) as data_logger,
    ):
        obj, cv, predictions = eval_step(
            state, 
            eval_initial_states, 
            eval_final_states
        )
        data_logger.log(0, {
            "evaluation_objective": obj,
            "evaluation_constraint_violation": cv,
        })
        if len(args.plot_trajectory) > 0:
            working_space = config_dataset["problem"]["constraints"]["working_space"]
            for idx in args.plot_trajectory:
                if idx >= predictions.p.shape[0]:
                    logger.warning(
                        f"Index {idx} is out of bounds."
                    )
                    continue
                # Plot the trajectory
                fig, ax = plot_trajectory(
                    trajectories=np.asarray(predictions.p[idx, :, :, :2]),
                    working_space=(
                        working_space["lower_bound"],
                        working_space["lower_bound"],
                        working_space["upper_bound"],
                        working_space["upper_bound"],
                    ),
                    initial_positions=np.asarray(eval_initial_states.p[idx, 0, :, :2]),
                    final_positions=np.asarray(eval_final_states.p[idx, 0, :, :2]),
                    title=f"Evaluation trajectory {idx}",
                )
                # fictitiously save the figure for different steps so that wandb
                # renders a slider
                data_logger.log_figure(
                    fig=fig,
                    key=f"evaluation_trajectories",
                )
                plt.close(fig)
