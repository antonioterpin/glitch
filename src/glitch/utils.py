"""Module containing common constants and functions."""

import numpy as np
import jax.numpy as jnp
import timeit
import logging
import signal
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from ruamel.yaml import YAML
import wandb
import matplotlib.pyplot as plt

DEBUG = False
JAX_DEBUG_JIT = False

def plotting(
    trainig_losses, validation_losses, eqcvs, ineqcvs
):
    """Plot training curves.
    
    Args:
        trainig_losses: Training losses.
        validation_losses: Validation losses.
        eqcvs: Equality constraint violations.
        ineqcvs: Inequality constraint violations.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.plot(trainig_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.semilogy(eqcvs, label="Equality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.semilogy(ineqcvs, label="Inequality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_hcnn(
    loader,
    state,
    n_iter,
    batched_objective,
    A,
    lb,
    ub,
    prefix,
    time_evals=10,
    print_res=True,
    cv_tol=1e-3,
    single_instance=True,
):
    """Evaluate the performance of the HCNN."""
    opt_obj = []
    hcnn_obj = []
    eq_cv = []
    ineq_cv = []
    for X, obj in loader:
        X_full = jnp.concatenate(
            (X, jnp.zeros((X.shape[0], A.shape[1] - X.shape[1], 1))), axis=1
        )
        predictions = state.apply_fn(
            {"params": state.params},
            X[:, :, 0],
            X_full,
            100000,
            n_iter=n_iter,
        )
        opt_obj.append(obj)
        hcnn_obj.append(batched_objective(predictions))
        # Equality Constraint Violation
        eq_cv_batch = jnp.abs(
            A[0].reshape(1, A.shape[1], A.shape[2])
            @ predictions.reshape(X.shape[0], A.shape[2], 1)
            - X_full,
        )
        eq_cv_batch = jnp.max(eq_cv_batch, axis=1)
        eq_cv.append(eq_cv_batch)
        # Inequality Constraint Violation
        ineq_cv_batch_ub = jnp.maximum(
            predictions.reshape(X.shape[0], A.shape[2], 1) - ub, 0
        )
        ineq_cv_batch_lb = jnp.maximum(
            lb - predictions.reshape(X.shape[0], A.shape[2], 1), 0
        )
        # Compute the maximum and normalize by the size
        ineq_cv_batch = jnp.maximum(ineq_cv_batch_ub, ineq_cv_batch_lb) / ub
        ineq_cv_batch = jnp.max(ineq_cv_batch, axis=1)
        ineq_cv.append(ineq_cv_batch)
    # Objectives
    opt_obj = jnp.concatenate(opt_obj, axis=0)
    opt_obj_mean = opt_obj.mean()
    hcnn_obj_mean = jnp.concatenate(hcnn_obj, axis=0).mean()
    # Equality Constraints
    eq_cv = jnp.concatenate(eq_cv, axis=0)
    eq_cv_mean = eq_cv.mean()
    eq_cv_max = eq_cv.max()
    # Inequality Constraints
    ineq_cv = jnp.concatenate(ineq_cv, axis=0)
    ineq_cv_mean = ineq_cv.mean()
    ineq_cv_max = ineq_cv.max()
    ineq_perc = (1 - jnp.mean(ineq_cv > cv_tol)) * 100
    # Inference time (assumes all the data in one batch)
    if single_instance:
        X_inf = X[:1, :, :]
        X_inf_full = jnp.concatenate(
            (X_inf, jnp.zeros((X_inf.shape[0], A.shape[1] - X_inf.shape[1], 1))), axis=1
        )
    else:
        X_inf = X
        X_inf_full = X_full
    times = timeit.repeat(
        lambda: state.apply_fn(
            {"params": state.params},
            X_inf[:, :, 0],
            X_inf_full,
            100000,
            n_iter=n_iter,
        ).block_until_ready(),
        repeat=time_evals,
        number=1,
    )
    eval_time = np.mean(times)
    eval_time_std = np.std(times)
    if print_res:
        print(f"=========== {prefix} performance ===========")
        print("Mean objective                : ", f"{hcnn_obj_mean:.5f}")
        print(
            "Mean|Max eq. cv               : ",
            f"{eq_cv_mean:.5f}",
            "|",
            f"{eq_cv_max:.5f}",
        )
        print(
            "Mean|Max normalized ineq. cv  : ",
            f"{ineq_cv_mean:.5f}",
            "|",
            f"{ineq_cv_max:.5f}",
        )
        print(
            "Perc of valid cv. tol.        : ",
            f"{ineq_perc:.3f}%",
        )
        print("Time for evaluation [s]       : ", f"{eval_time:.5f}")
        print("Optimal mean objective        : ", f"{opt_obj_mean:.5f}")

    return (opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, eval_time, eval_time_std)


def evaluate_instance(
    problem_idx,
    loader,
    state,
    n_iter,
    batched_objective,
    A,
    lb,
    ub,
    prefix,
):
    """Evaluate performance on single problem instance."""
    X = loader.dataset.dataset.x0sets[
        loader.dataset.indices[problem_idx : problem_idx + 1]
    ]
    X_full = jnp.concatenate(
        (X, jnp.zeros((X.shape[0], A.shape[1] - X.shape[1], 1))), axis=1
    )
    predictions = state.apply_fn(
        {"params": state.params},
        X[:, :, 0],
        X_full,
        100000,
        n_iter=n_iter,
    )

    objective_val_hcnn = batched_objective(predictions).item()
    eqcv_val_hcnn = jnp.abs(
        A[0].reshape(1, A.shape[1], A.shape[2])
        @ predictions.reshape(X.shape[0], A.shape[2], 1)
        - X_full,
    ).max()
    ineqcv_ub_val_hcnn = jnp.maximum(predictions.reshape(1, -1, 1) - ub, 0).max()
    ineqcv_lb_val_hcnn = jnp.maximum(lb - predictions.reshape(1, -1, 1), 0).max()
    ineqcv_val_hcnn = jnp.maximum(ineqcv_ub_val_hcnn, ineqcv_lb_val_hcnn)
    print(f"=========== {prefix} individual performance ===========")
    print("HCNN")
    print(f"Objective:  \t{objective_val_hcnn:.5e}")
    print(f"Eq. cv:     \t{eqcv_val_hcnn:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val_hcnn:.5e}")

    objective_val = loader.dataset.dataset.objectives[
        loader.dataset.indices[problem_idx]
    ]
    eqcv_val = jnp.abs(
        A[0].reshape(1, A.shape[1], A.shape[2])
        @ loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]].reshape(
            X.shape[0], A.shape[2], 1
        )
        - X_full
    ).max()
    ineqcv_ub_val = jnp.maximum(
        loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]].reshape(
            1, -1, 1
        )
        - ub,
        0,
    ).max()
    ineqcv_lb_val = jnp.maximum(
        lb
        - loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]].reshape(
            1, -1, 1
        ),
        0,
    ).max()
    ineqcv_val = jnp.maximum(ineqcv_ub_val, ineqcv_lb_val)

    print("Optimal Solution")
    print(f"Objective:  \t{objective_val:.5e}")
    print(f"Eq. cv:     \t{eqcv_val:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val:.5e}")


# Create a logger instance
logger = logging.getLogger(__name__)

# Configure the logging format
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GracefulShutdown:
    """A context manager for graceful shutdowns."""

    stop = False

    def __init__(self, exit_message: Optional[str] = None):
        """Initializes the GracefulShutdown context manager.

        Args:
            exit_message (str): The message to log upon shutdown.
        """
        self.exit_message = exit_message

    def __enter__(self):
        """Register the signal handler."""

        def handle_signal(signum, frame):
            self.stop = True
            if self.exit_message:
                logger.info(self.exit_message)

        signal.signal(signal.SIGINT, handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Unregister the signal handler."""
        pass


def load_configuration(file_path: str):
    """Load YAML configuration from file."""
    yaml = YAML(typ="safe", pure=True)
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.load(file)


class Logger:
    """Encapsulates logging functionalities."""

    PROJECT_NAME = "fluids-estimation"

    def __init__(self, dataset: str) -> None:
        """Initializes the Logger and creates a new wandb run.

        Args:
            dataset (str): The name of the dataset to be logged.
        """
        wandb.login()
        self.dataset = dataset
        self.run = wandb.init(project=self.PROJECT_NAME)
        self.run.name = dataset + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def __enter__(self) -> "Logger":
        """Enters the runtime context for Logger.

        Returns:
            Logger: The current instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exits the runtime context and finishes the wandb run."""
        wandb.finish()

    class Timer:
        """A context manager for timing code execution and logging the results."""

        def __init__(
            self,
            label: str,
            t: int,
            log_vars: Optional[Callable[[], Dict[str, Any]]] = None,
        ) -> None:
            """Initializes the Timer context manager.

            Args:
                label (str): The label for the timed section.
                t (int): An indexing parameter (for example, the epoch).
                log_vars (Optional[Callable[[], Dict[str, Any]]], optional):
                    A callable that returns a dictionary of variable names and values
                    to log. Defaults to None.
            """
            self.label = label
            self.t = t
            self.log_vars = log_vars

        def __enter__(self) -> "Logger.Timer":
            """Starts the timer.

            Returns:
                Logger.Timer: The Timer instance.
            """
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            """Stops the timer and logs the elapsed time and optional variables."""
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            # Prepare log data with elapsed time and the provided integer parameter.
            log_data = {self.label: elapsed_time, "t": self.t}
            # If a callable to fetch additional variables is provided, update log data.
            if self.log_vars is not None:
                log_data.update(self.log_vars())
            wandb.log(log_data)

    def log(
        self,
        t: int,
        data: Dict[str, Any],
    ):
        """Logs data.

        Args:
            t (int): An indexing parameter (for example, the epoch).
            data (Dict[str, Any]): A dictionary of variable names and values to log.
        """
        # Add the integer parameter to the log data.
        data["t"] = t
        wandb.log(data)

    def timeit(
        self,
        label: str,
        t: int,
        log_vars: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> "Logger.Timer":
        """Creates a Timer context manager to time a code block and log its metrics.

        Args:
            label (str): The label for the timed section.
            t (int): An integer parameter (e.g., a batch index).
            log_vars (Optional[Callable[[], Dict[str, Any]]], optional):
                A callable that returns a dictionary of variable names and values
                to be logged. Defaults to None.

        Returns:
            Logger.Timer: A Timer context manager.
        """
        return Logger.Timer(label, t, log_vars)