import jax
import jax.numpy as jnp
from typing import Optional, List

from glitch.definitions.dynamics import FleetStateInput

class TransitionsDataset():
    """Dataset for the transitions generation task."""
    def __init__(
            self, 
            n_states: int, 
            n_robots: int, 
            horizon: int,
            batch_size: int,
            keys: Optional[List[int]] = None,
            offset: int = 0):
        """Initialize the dataset.

        TODO: Allow customizing initial and terminal states.

        Args:
            n_states (int): Number of states.
            n_robots (int): Number of robots.
            horizon (int): Horizon length.
            keys (list, optional): List of keys for the dataset. Defaults to None.
                If None, the dataset will be unbounded.
            offset (int): Offset for the dataset. Defaults to 0.
        """
        super().__init__()
        self.n_states = n_states
        self.n_robots = n_robots
        self.horizon = horizon
        self.batch_size = batch_size
        self.keys = keys
        self.offset = offset

        # TODO: allow customizing initial and terminal states
        lb, ub = -4.0, 4.0
        positions_y = jnp.linspace(lb, ub, n_robots)
        positions_x = lb * jnp.ones((n_robots, 1))
        positions_nominal = jnp.concatenate((positions_x, positions_y[:, None]), axis=1)
        p = positions_nominal.reshape((1, n_robots, 2))
        v = jax.numpy.zeros((1, n_robots, 2))
        u = jax.numpy.zeros((0, n_robots, 2))
        shift = jnp.array([ub - lb, 0.0])

        def sample_initial_states(key):
            return FleetStateInput(p=p, v=v, u=u)
        
        def sample_final_states(key):
            pp = p + shift[None, None, :]
            # shuffle the final positions
            pp = jax.random.permutation(key, pp, axis=1)
            return FleetStateInput(p=pp, v=v, u=u)
        
        self.sample_initial_states = jax.vmap(sample_initial_states)
        self.sample_final_states = jax.vmap(sample_final_states)
        
        # initial_positions_box = jnp.concatenate((
        #     -4.0 * jnp.ones((n_states, 1))
        #     -3.0 * jnp.ones((n_states, 1)),
        # ), axis=1)
        # final_positions_box = jnp.concatenate((
        #     3.0 * jnp.ones((n_states, 1)),
        #     4.0 * jnp.ones((n_states, 1)),
        # ), axis=1)

        # self.sample_initial_states = jax.vmap(
        #     lambda k: sample_from_box(
        #         k, initial_positions_box, n_robots, zero_velocity=True
        #     )
        # )
        # self.sample_final_states = jax.vmap(
        #     lambda k: sample_from_box(
        #         k, final_positions_box, n_robots, zero_velocity=True
        #     )
        # )

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        if self.keys is None:
            return 2**31 - 1
        return len(self.keys)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the initial and final states.
        """
        # Note: this way offset mantains the same behavior in both cases
        idx = idx if self.keys is None else self.keys[idx]
        key = jax.random.PRNGKey(idx + self.offset)
        keyi, keyf = jax.random.split(key)
        keysi = jax.random.split(keyi, self.batch_size)
        keysf = jax.random.split(keyf, self.batch_size)

        return self.sample_initial_states(keysi), self.sample_final_states(keysf)


def create_dataloaders(config):
    """
    Create data loaders for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        tuple: Data loaders for training, validation, and testing.
    """
    dataset_size = config["dataset"].get("dataset_size", -1)

    # Split the dataset into training, validation, and test sets
    batch_size = config["dataset"].get("batch_size", 32)
    val_size = config["dataset"].get("val_size", 32)
    test_size = config["dataset"].get("test_size", 32)


    dataset_validation = TransitionsDataset(
        keys=[0],
        n_states=config["problem"]["n_states"],
        n_robots=config["problem"]["n_robots"],
        horizon=config["problem"]["horizon"],
        batch_size=val_size,
        offset=0
    )
    dataset_test = TransitionsDataset(
        keys=[1],
        n_states=config["problem"]["n_states"],
        n_robots=config["problem"]["n_robots"],
        horizon=config["problem"]["horizon"],
        batch_size=test_size,
        offset=0
    )
    dataset_train = TransitionsDataset(
        keys=None if dataset_size < 0 else jnp.arange(dataset_size),
        offset=2,
        batch_size=batch_size,
        n_states=config["problem"]["n_states"],
        n_robots=config["problem"]["n_robots"],
        horizon=config["problem"]["horizon"],
    )
    
    return dataset_train, dataset_validation, dataset_test