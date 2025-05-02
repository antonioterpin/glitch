import argparse
import datetime
import os
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
from tqdm import tqdm
import optax
from flax.training import train_state
from hcnn.project import Project

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

jax.config.update("jax_enable_x64", True)

def build_batched_objective(config_hcnn, config_problem):
    collision_penalty_fn_name = config_hcnn["collision_penalty_fn"]
    try:
        collision_penalty_fn = getattr(preferences, collision_penalty_fn_name)
    except AttributeError:
        raise ValueError(f"Unknown collision penalty '{collision_penalty_fn_name}'")
    compensation = jnp.array(config_problem["gravity"])
    
    def batched_objective(predictions):
        return jnp.sum(predictions.p ** 2)
        return (
            preferences.input_effort(predictions, compensation) 
            + collision_penalty_fn(
                predictions, 
                config_hcnn["collision_penalty"], 
                config_hcnn["collision_normalization_factor"]
            )
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
            return batched_objective(predictions).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, state.apply_gradients(grads=grads)

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

        accuracy = batched_objective(predictions).mean()
        # During training, we report the average constraint violation
        x = predictions_to_projection_layer_format(predictions)
        n_eq = project.eq_constraint.n_constraints
        b = prepare_b_from_batch(n_eq, initial_states, final_states)
        project.eq_constraint.b = b
        cv = project.cv(x).mean()

        return accuracy, cv
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
        "--plot-training-curves",
        action="store_true",
        help="Plot training curves.",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save the results.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save the model every save_every training batches.",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Evaluate the model every eval_every training batches.",
    )

    args = parser.parse_args()

    return args

def train_hcnn(
    projection_layer: Project,
    state: train_state.TrainState,
    dataset_training: TransitionsDataset,
    dataset_validation: TransitionsDataset,
    save_every: int,
    eval_every: int,
    output_dir: str,
    config_hcnn: dict,
    config_problem: dict,
):
    """Train the HCNN model."""
    eval_initial_states, eval_final_states = dataset_validation[0]
    train_step, eval_step = build_steps(
        project=projection_layer,
        config_hcnn=config_hcnn,
        config_problem=config_problem,
    )
    model_name = config_hcnn["model_name"]
    with (
        GracefulShutdown("Stop detected, finishing epoch...") as g,
        Logger(f"training-{model_name}") as data_logger,
    ):
        for step in (pbar := tqdm(range(len(dataset_training)))):
            if g.stop:
                break
            initial_states, final_states = dataset_training[step]
            t = time.time()
            loss, state = train_step(
                state, 
                initial_states,
                final_states,
            )
            t = time.time() - t
            pbar.set_description(f"Train Loss: {loss.mean():.5f}")
            data_logger.log(step, {
                "loss": loss,
                "batch_training_time": t,
            })

            if step % eval_every == 0:
                obj, cv = eval_step(
                    state, 
                    eval_initial_states, 
                    eval_final_states
                )
                data_logger.log(step, {
                    "eval_objective": obj,
                    "eval_constraint_violation": cv,
                })

            if output_dir is not None and step % save_every == 0:
                save_model(
                    state.params,
                    output_dir,
                    f"{model_name}_{step}",
                )


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
    
    output_dir = None
    if args.save_results:
        # Create the output directory if it doesn't exist
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir.format(timestamp=timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved in: {output_dir}")

    if args.train > 0:
        training_time_start = time.time()
        train_hcnn(
            projection_layer=hcnn.project,
            state=state,
            dataset_training=dataset_training,
            dataset_validation=dataset_validation,
            save_every=args.save_every,
            eval_every=args.eval_every,
            output_dir=output_dir,
            config_hcnn=config_hcnn,
            config_problem=config_dataset["problem"],
        )
        training_time = time.time() - training_time_start
        print(f"Training time: {training_time:.5f} seconds")

    # Evaluate the (trained) model on the test set
    raise NotImplementedError(
        "Evaluation is not implemented yet."
    )