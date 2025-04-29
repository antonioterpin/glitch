import pytest
import jax
import jax.numpy as jnp
from jax import tree_util
from glitch.dynamics import FleetState

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def example_state():
    """Return a small, valid FleetState with two robots."""
    v = jnp.array([[1.0, -0.5], [0.3, 0.9]])  # shape (2, 2)
    p = jnp.array([[0.0, 1.0], [2.0, -1.0]])  # shape (2, 2)
    return FleetState(v, p)

# ----------------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------------

def test_valid_construction(example_state):
    """Constructor stores arrays unchanged when they satisfy all constraints."""
    fs = example_state
    assert jnp.array_equal(fs.v, jnp.array([[1.0, -0.5], [0.3, 0.9]]))
    assert jnp.array_equal(fs.p, jnp.array([[0.0, 1.0], [2.0, -1.0]]))


def test_dimension_mismatch():
    """v and p must have the same number of dimensions (ndim)."""
    v = jnp.array([[1.0, 2.0]])       # ndim = 2
    p = jnp.array([0.0, 1.0, 2.0])    # ndim = 1
    with pytest.raises(ValueError, match="same number of dimensions"):
        FleetState(v, p)


def test_not_2d():
    """v and p must be 2‑D arrays."""
    v = jnp.ones((2, 2, 2))  # ndim = 3
    p = jnp.ones((2, 2, 2))
    with pytest.raises(ValueError, match="must be 2D arrays"):
        FleetState(v, p)


def test_robot_count_mismatch():
    """v and p must refer to the same number of robots (same first dimension)."""
    v = jnp.ones((3, 2))  # 3 robots
    p = jnp.ones((2, 2))  # 2 robots
    with pytest.raises(ValueError, match="same number of robots"):
        FleetState(v, p)


# ----------------------------------------------------------------------------
# PyTree integration
# ----------------------------------------------------------------------------
def test_tree_roundtrip(example_state):
    """tree_flatten/unflatten round‑trip should reconstruct identical object."""
    leaves, aux = tree_util.tree_flatten(example_state)
    reconstructed = tree_util.tree_unflatten(aux, leaves)
    assert isinstance(reconstructed, FleetState)
    assert jnp.array_equal(reconstructed.v, example_state.v)
    assert jnp.array_equal(reconstructed.p, example_state.p)


# ----------------------------------------------------------------------------
# Dynamics matrix generation
# ----------------------------------------------------------------------------

def test_get_dynamics_single_robot():
    """Dynamics matrices match the expected continuous‑to‑discrete model for one robot."""
    v = jnp.zeros((1, 1))  # shape (1, 1)
    p = jnp.zeros((1, 1))
    fs = FleetState(v, p)
    h = 0.1
    A_exp = jnp.array([[1.0, h], [0.0, 1.0]])
    B_exp = jnp.array([[h ** 2 / 2], [h]])
    A, B = fs.get_dynamics(h)
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert jnp.allclose(A, A_exp)
    assert jnp.allclose(B, B_exp)

@pytest.mark.parametrize("h", [0.1, 0.2, 0.5])
def test_get_dynamics_single_robot_2d(h):
    """Dynamics matrices match the expected continuous‑to‑discrete model for one robot."""
    v = jnp.zeros((1, 2))  # shape (1, 2)
    p = jnp.zeros((1, 2))
    fs = FleetState(v, p)
    A_exp = jnp.array([
        [1.0, 0.0, h, 0.0], 
        [0.0, 1.0, 0.0, h],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    B_exp = jnp.array([
        [h ** 2 / 2, 0.0],
        [0.0, h ** 2 / 2],
        [h, 0.0],
        [0.0, h],
    ])
        
    A, B = fs.get_dynamics(h)
    assert A.shape == A_exp.shape, (
        f"A) Expected shape: {A_exp.shape}, got: {A.shape}"
    )
    assert B.shape == B_exp.shape, (
        f"B) Expected shape: {B_exp.shape}, got: {B.shape}"
    )
    assert jnp.allclose(A, A_exp), (
        f"A) Expected:\n{A_exp}, got:\n{A}"
    )
    assert jnp.allclose(B, B_exp), (
        f"B) Expected:\n{B_exp}, got:\n{B}"
    )

@pytest.mark.parametrize("h", [0.1, 0.2, 0.5])
def test_get_dynamics_multiple_robots(example_state, h):
    A_exp = jnp.array([
        [1.0, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, h, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, h, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, h],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    B_exp = jnp.array([
        [h ** 2 / 2, 0.0, 0.0, 0.0],
        [0.0, h ** 2 / 2, 0.0, 0.0],
        [0.0, 0.0, h ** 2 / 2, 0.0],
        [0.0, 0.0, 0.0, h ** 2 / 2],
        [h, 0.0, 0.0, 0.0],
        [0.0, h, 0.0, 0.0],
        [0.0, 0.0, h, 0.0],
        [0.0, 0.0, 0.0, h],
    ])

    A, B = example_state.get_dynamics(h)

    # Shape checks
    assert A.shape == A_exp.shape, (
        f"A) Expected shape: {A_exp.shape}, got: {A.shape}"
    )
    assert B.shape == B_exp.shape, (
        f"B) Expected shape: {B_exp.shape}, got: {B.shape}"
    )

    # Value checks
    assert jnp.allclose(A, A_exp)
    assert jnp.allclose(B, B_exp)

def test_get_dynamics_over_horizon_simple(example_state):
    h = 1
    A_expected = jnp.array([
        # timestep 1
        # first agent position
        [1.0, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, h, 0.0, 0.0],
        # second agent position
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, h, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, h],
        # first agent velocity
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        # second agent velocity
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # timestep 2
        # first agent position
        [1.0, 0.0, 0.0, 0.0, 2 * h, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 2 * h, 0.0, 0.0],
        # second agent position
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2 * h, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2 * h],
        # first agent velocity
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        # second agent velocity
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    B_expected = jnp.array([
        # timestep 1
        # first agent position
        [h ** 2 / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, h ** 2 / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # second agent, position
        [0.0, 0.0, h ** 2 / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, h ** 2 / 2, 0.0, 0.0, 0.0, 0.0],
        # first agent velocity
        [h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # second agent, velocity
        [0.0, 0.0, h, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, 0.0],
        # timestep 2
        # first agent, position
        [3 * h ** 2 / 2, 0.0, 0.0, 0.0, h ** 2 / 2, 0.0, 0.0, 0.0],
        [0.0, 3 * h ** 2 / 2, 0.0, 0.0, 0.0, h ** 2 / 2, 0.0, 0.0],
        # second agent, position
        [0.0, 0.0, 3 * h ** 2 / 2, 0.0, 0.0, 0.0, h ** 2 / 2, 0.0],
        [0.0, 0.0, 0.0, 3 * h ** 2 / 2, 0.0, 0.0, 0.0, h ** 2 / 2],
        # first agent, velocity
        [h, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0],
        [0.0, h, 0.0, 0.0, 0.0, h, 0.0, 0.0],
        # second agent, velocity
        [0.0, 0.0, h, 0.0, 0.0, 0.0, h, 0.0],
        [0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, h],
    ])

    A_horizon, B_horizon = example_state.get_dynamics_over_horizon(h, 2)
    assert A_horizon.shape == A_expected.shape, (
        f"A) Expected shape: {A_expected.shape}, got: {A_horizon.shape}"
    )
    assert B_horizon.shape == B_expected.shape, (
        f"B) Expected shape: {B_expected.shape}, got: {B_horizon.shape}"
    )
    assert jnp.allclose(A_horizon, A_expected), (
        f"A) Expected:\n{A_expected}, got:\n{A_horizon}"
    )
    assert jnp.allclose(B_horizon, B_expected), (
        f"B) Expected:\n{B_expected}, got:\n{B_horizon}"
    )

# ----------------------------------------------------------------------------
# FleetState flattening
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("p, p_flatten, v, v_flatten", [
    (
        jnp.array([[0.0, 1.0], [2.0, -1.0]]),
        jnp.array([[0.0], [1.0], [2.0], [-1.0]]),
        jnp.array([[1.0, -0.5], [0.3, 0.9]]),
        jnp.array([[1.0], [-0.5], [0.3], [0.9]])
    ),
    (
        jnp.array([[0.0, 1.0], [2.0, -1.0], [3.0, 4.0]]),
        jnp.array([[0.0], [1.0], [2.0], [-1.0], [3.0], [4.0]]),
        jnp.array([[1.0, -0.5], [0.3, 0.9], [5.0, 6.0]]),
        jnp.array([[1.0], [-0.5], [0.3], [0.9], [5.0], [6.0]])
    )
])
def test_get_flatten_state(p, p_flatten, v, v_flatten):
    # Random sample of states
    fs = FleetState(v, p)

    n_robots, n_dim = p.shape
    expected_shape = (2 * n_dim * n_robots, 1)
    flattened = fs.get_flatten_state()
    assert flattened.shape == expected_shape, (
        f"Flattened state does not match expected shape. "
        f"Expected:\n{expected_shape}, got:\n{flattened.shape}"
    )

    # Construct flatten state manually
    expected_flattened = jnp.concatenate((
        p_flatten,
        v_flatten
    ), axis=0)

    assert jnp.array_equal(flattened, expected_flattened), (
        f"Flattened state does not match expected values. "
        f"p: {fs.p}, v: {fs.v}, "
        f"Expected: {expected_flattened}, got: {flattened}."
    )

@pytest.mark.parametrize("n_robots", [1, 2])
@pytest.mark.parametrize("n_dim", [1, 2])
@pytest.mark.parametrize("seed", [0])
def test_recover_state_from_flatten(n_robots, n_dim, seed):
    # Random sample of states
    rng = jax.random.PRNGKey(seed)
    keyv, keyp = jax.random.split(rng)
    p = jax.random.uniform(keyp, (n_robots, n_dim))
    v = jax.random.uniform(keyv, (n_robots, n_dim))
    fs = FleetState(v, p)

    # Flatten the state
    flattened = fs.get_flatten_state()

    # Recover the state from the flattened state
    recovered_fs = fs.unpack_flatten(flattened)

    assert jnp.array_equal(recovered_fs.v, fs.v), (
        f"Recovered velocity does not match original. "
        f"p: {fs.p}, v: {fs.v}, "
        f"Flattened state: {flattened}, "
        f"Expected: {fs.v}, got: {recovered_fs.v}."
    )
    assert jnp.array_equal(recovered_fs.p, fs.p), (
        f"Recovered position does not match original. "
        f"Expected: {fs.p}, got: {recovered_fs.p}."
    )

# ----------------------------------------------------------------------------
# FleetState input flattening
# ----------------------------------------------------------------------------
def test_get_flatten_input(example_state):
    """Flatten the input of the fleet: [u1; u2; ...]."""
    u = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    u_flatten = example_state.get_flatten_input(u)
    expected_shape = (4, 1)
    assert u_flatten.shape == expected_shape, (
        f"Flattened input does not match expected shape. "
        f"Expected:\n{expected_shape}, got:\n{u_flatten.shape}"
    )

    # Construct flatten input manually
    expected_flattened = jnp.array([1.0, 2.0, 3.0, 4.0])[..., None]  # shape (4, 1)

    assert jnp.array_equal(u_flatten, expected_flattened), (
        f"Flattened input does not match expected values. "
        f"Expected: {expected_flattened}, got: {u_flatten}."
    )

def test_get_flatten_input_horizon(example_state):
    u = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # shape (2, 2, 2)
    u_flatten = example_state.get_flatten_input_horizon(u)
    expected_shape = (8, 1)
    assert u_flatten.shape == expected_shape, (
        f"Flattened input does not match expected shape. "
        f"Expected:\n{expected_shape}, got:\n{u_flatten.shape}"
    )
    # Construct flatten input manually
    expected_flattened = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])[..., None]  # shape (8, 1)
    assert jnp.array_equal(u_flatten, expected_flattened), (
        f"Flattened input does not match expected values. "
        f"Expected: {expected_flattened}, got: {u_flatten}."
    )

@pytest.mark.parametrize("n_robots", [1, 2, 3])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("seed", [0])
def test_get_flatten_input_horizon_multiple_robots(n_robots, n_dim, horizon, seed):
    # Random sample of states
    rng = jax.random.PRNGKey(0)
    keyv, keyp = jax.random.split(rng)
    v = jax.random.uniform(keyv, (n_robots, n_dim))
    p = jax.random.uniform(keyp, (n_robots, n_dim))
    fs = FleetState(v, p)

    # Input
    u = jnp.arange(horizon * n_robots * n_dim).reshape((horizon, n_robots, n_dim))

    # Flatten the input
    u_flatten = fs.get_flatten_input_horizon(u)

    expected_shape = (horizon * n_robots * n_dim, 1)
    assert u_flatten.shape == expected_shape, (
        f"Flattened input does not match expected shape. "
        f"Expected:\n{expected_shape}, got:\n{u_flatten.shape}"
    )

    # Construct flatten input manually
    expected_flattened = jnp.arange(horizon * n_robots * n_dim).reshape((-1, 1))
    assert jnp.array_equal(u_flatten, expected_flattened), (
        f"Flattened input does not match expected values. "
        f"Expected: {expected_flattened}, got: {u_flatten}."
    )

@pytest.mark.parametrize("n_robots", [1, 2, 3])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("seed", [0])
def test_unflatten_input(n_robots, n_dim, horizon, seed):
    # Random sample of states
    rng = jax.random.PRNGKey(seed)
    keyv, keyp, keyu = jax.random.split(rng, 3)
    v = jax.random.uniform(keyv, (n_robots, n_dim))
    p = jax.random.uniform(keyp, (n_robots, n_dim))
    fs = FleetState(v, p)

    # Input
    u = jax.random.uniform(keyu, (horizon, n_robots, n_dim))
    u_unflatten = fs.get_unflatten_input(
        fs.get_flatten_input_horizon(u)
    )

    # Check the shape of the unflattened input
    assert u_unflatten.shape == (horizon, n_robots, n_dim), (
        f"Unflattened input does not match expected shape. "
        f"Expected:\n{(horizon, n_robots, n_dim)}, got:\n{u_unflatten.shape}"
    )
    # Check the values of the unflattened input
    assert jnp.array_equal(u_unflatten, u), (
        f"Unflattened input does not match expected values. "
        f"Expected: {u}, got: {u_unflatten}."
    )


# ----------------------------------------------------------------------------
# Closed‑loop propagation test with known inputs
# ----------------------------------------------------------------------------

def test_propagation_known_inputs():
    n_robots = 2
    h = 1.0
    steps = 2

    # Initial state: all zeros, 1‑D motion -> arrays of shape (n, 1)
    v0 = jnp.zeros((n_robots, 1))
    p0 = jnp.zeros((n_robots, 1))
    fs = FleetState(v0, p0)

    # Dynamics matrices
    A, B = fs.get_dynamics(h)

    # Constant inputs (per robot)
    u = jnp.array([1.0, 2.0])[..., None]  # shape (n,1)

    x = fs.get_flatten_state()  # shape (2n, 1)

    for _ in range(steps):
        x = A @ x + B @ u

    # Recover per‑robot positions and velocities
    fs = fs.unpack_flatten(x)

    # Analytical expectations derived from A, B structure
    expected_p = 2 * u
    expected_v = 2 * u

    assert jnp.allclose(fs.p, expected_p)
    assert jnp.allclose(fs.v, expected_v)

@pytest.mark.parametrize("h", [0.1, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("n_robots", [1, 2, 3, 10])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [0])
def test_propagation_over_horizon(
    h, n_robots, n_dim, horizon, seed
):
    # Random sample of initial states and inputs
    rng = jax.random.PRNGKey(seed)
    keyv, keyp, keyu = jax.random.split(rng, 3)
    v = jax.random.uniform(keyv, (n_robots, n_dim))
    p = jax.random.uniform(keyp, (n_robots, n_dim))
    fs = FleetState(v, p)

    u = jax.random.uniform(keyu, (horizon, n_robots * n_dim))
    u_flatten = u.reshape((-1, 1), order='C')

    # Dynamics matrices
    A, B = fs.get_dynamics(h)
    A_horizon, B_horizon = fs.get_dynamics_over_horizon(h, horizon)

    # Initial state flattening
    x = fs.get_flatten_state()
    xk = x

    # Propagation over the horizon
    xks = A_horizon @ x + B_horizon @ u_flatten
    for i in range(horizon):
        xk = A @ xk + B @ u[i, ..., None]
        # Check the shape of the propagated state
        xk_from_xks = xks[i * (2 * n_dim * n_robots):(i + 1) * (2 * n_dim * n_robots)]
        assert xk.shape == xk_from_xks.shape, (
            f"Propagated state does not match expected shape. "
            f"Expected:\n{xk_from_xks.shape}, got:\n{xk.shape}"
        )
        # Check the values of the propagated state
        assert jnp.allclose(xk, xk_from_xks), (
            f"Propagated state does not match expected values. "
            f"Expected:\n{xk_from_xks}, got:\n{xk}"
        )
