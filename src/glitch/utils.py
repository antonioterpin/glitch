"""Module containing common constants and functions."""

import logging
import signal
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Optional
from ruamel.yaml import YAML
import wandb

DEBUG = False
JAX_DEBUG_JIT = False


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

    PROJECT_NAME = "glitch"

    def __init__(self, run_name: str) -> None:
        """Initializes the Logger and creates a new wandb run.

        Args:
            dataset (str): The name of the dataset to be logged.
        """
        wandb.login()
        self.run_name = run_name
        self.run = wandb.init(
            project=self.PROJECT_NAME,
            name=self.run_name,
            id=self.run_name,
            resume="allow")

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
    
    def log_figure(
        self,
        fig: plt.figure,
        key: str,
        t: Optional[int] = None,
    ):
        """Logs a figure.

        Args:
            t (int): An indexing parameter (for example, the epoch).
            fig (plt.figure): The figure to log.
            key (str): The key under which to log the figure.
        """
        # Add the integer parameter to the log data.
        wandb.log({key: wandb.Image(fig)}, step=t)
        