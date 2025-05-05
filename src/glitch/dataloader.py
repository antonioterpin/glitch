import jax
import jax.numpy as jnp
from typing import Optional, List

from glitch.definitions.dynamics import FleetStateInput
from glitch.configuration.random import sample_from_line, sample_from_circle

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

        if n_states > 2:
            raise ValueError("n_states must be 2 for the transitions dataset.")

        self.n_states = n_states
        self.n_robots = n_robots
        self.horizon = horizon
        self.batch_size = batch_size
        self.keys = keys
        self.offset = offset

        def sample_initial_states_from_line(key):
            p1 = jax.random.uniform(key, (1, 2), minval=-4.5, maxval=-3.5)
            p2_x = jax.random.uniform(key, (1, 1), minval=-4.5, maxval=-3.5)
            p2_y = jax.random.uniform(key, (1, 1), minval=3.5, maxval=4.5)
            p2 = jnp.concatenate((p2_x, p2_y), axis=1)
            return sample_from_line(p1, p2, n_robots)
        
        def sample_final_states_from_line(key):
            p3 = jax.random.uniform(key, (1, 2), minval=3.5, maxval=4.5)
            p4_x = jax.random.uniform(key, (1, 1), minval=3.5, maxval=4.5)
            p4_y = jax.random.uniform(key, (1, 1), minval=-4.5, maxval=-3.5)
            p4 = jnp.concatenate((p4_x, p4_y), axis=1)
            return sample_from_line(p3, p4, n_robots)
        
        def sample_initial_states_from_circle(key):
            center_x = jax.random.uniform(key, (1, 1), minval=-3.25, maxval=-3)
            center_y = jax.random.uniform(key, (1, 1), minval=-3.25, maxval=3.25)
            center = jnp.concatenate((center_x, center_y), axis=1)
            radius = jax.random.uniform(key, (), minval=0.5, maxval=1.5)
            return sample_from_circle(center, radius, n_robots)
        
        def sample_final_states_from_circle(key):
            center_x = jax.random.uniform(key, (1, 1), minval=3, maxval=3.25)
            center_y = jax.random.uniform(key, (1, 1), minval=-3.25, maxval=3.25)
            center = jnp.concatenate((center_x, center_y), axis=1)
            radius = jax.random.uniform(key, (), minval=0.5, maxval=1.5)
            return sample_from_circle(center, radius, n_robots)
        
        def sample_initial_states(key):
            fns = [sample_initial_states_from_line, sample_initial_states_from_circle]
            key_idx, key_fn = jax.random.split(key)
            # sample an integer
            idx = jax.random.randint(key_idx, (), 0, len(fns))
            # dispatch
            return jax.lax.switch(idx, fns, key_fn)
        
        def sample_final_states(key):
            fns = [sample_final_states_from_line, sample_final_states_from_circle]
            key_idx, key_fn = jax.random.split(key)
            # sample an integer
            idx = jax.random.randint(key_idx, (), 0, len(fns))
            # dispatch
            return jax.lax.switch(idx, fns, key_fn)
            
        
        self.sample_initial_states = jax.vmap(sample_initial_states)
        self.sample_final_states = jax.vmap(sample_final_states)
    

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