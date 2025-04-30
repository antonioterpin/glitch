import pytest
import jax
import jax.numpy as jnp
from jax import tree_util
from glitch.dynamics import (
    FleetStateInput,
    get_dynamics,
    get_position_mask,
    get_velocity_mask,
    get_input_mask,
)

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def example_fsu():
    """Return a small, valid FleetState with two robots."""
    v = jnp.array([[[1.0, -0.5], [0.3, 0.9]]])  # shape (1, 2, 2)
    p = jnp.array([[[0.0, 1.0], [2.0, -1.0]]])  # shape (1, 2, 2)
    u = jnp.zeros((0, 2, 2))  # shape (0, 2, 2)
    return FleetStateInput(
        p=p,
        v=v,
        u=u
    )

# ----------------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------------

def test_valid_construction(example_fsu):
    """Constructor stores arrays unchanged when they satisfy all constraints."""
    assert jnp.array_equal(example_fsu.v, jnp.array([[[1.0, -0.5], [0.3, 0.9]]]))
    assert jnp.array_equal(example_fsu.p, jnp.array([[[0.0, 1.0], [2.0, -1.0]]]))
    assert jnp.array_equal(example_fsu.u, jnp.zeros((0, 2, 2)))


def test_dimension_mismatch():
    """v and p must have the same number of dimensions (ndim)."""
    v = jnp.array([[[1.0, 2.0]]])       # ndim = 2
    p = jnp.array([[0.0, 1.0, 2.0]])    # ndim = 1
    u = jnp.zeros((0, 2, 2))  # shape (0, 2, 2)
    with pytest.raises(ValueError, match="same number of dimensions"):
        FleetStateInput(v, p, u)

# TODO: add more tests for constructor


def test_robot_count_mismatch():
    """v and p must refer to the same number of robots (same first dimension)."""
    v = jnp.ones((1, 3, 2))  # 3 robots
    p = jnp.ones((1, 2, 2))  # 2 robots
    u = jnp.zeros((0, 2, 2))  # shape (0, 2, 2)
    with pytest.raises(ValueError, match="same number of robots"):
        FleetStateInput(v, p, u)


# ----------------------------------------------------------------------------
# PyTree integration
# ----------------------------------------------------------------------------
def test_tree_roundtrip(example_fsu):
    """tree_flatten/unflatten round‑trip should reconstruct identical object."""
    leaves, aux = tree_util.tree_flatten(example_fsu)
    reconstructed = tree_util.tree_unflatten(aux, leaves)
    assert isinstance(reconstructed, FleetStateInput)
    assert jnp.array_equal(reconstructed.v, example_fsu.v)
    assert jnp.array_equal(reconstructed.p, example_fsu.p)
    assert jnp.array_equal(reconstructed.u, example_fsu.u)


# ----------------------------------------------------------------------------
# Dynamics matrix generation
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("h", [0.1, 0.2, 0.5])
def test_get_dynamics_single_robot(h):
    A_exp = jnp.array([[1.0, h], [0.0, 1.0]])
    B_exp = jnp.array([[h ** 2 / 2], [h]])
    A, B = get_dynamics(
        horizon=1,
        n_states=1,
        n_robots=1,
        h=h,
    )
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert jnp.allclose(A, A_exp)
    assert jnp.allclose(B, B_exp)

@pytest.mark.parametrize("h", [0.1, 0.2, 0.5])
def test_get_dynamics_single_robot_2d(h):
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
        
    A, B = get_dynamics(
        horizon=1,
        n_states=2,
        n_robots=1,
        h=h,
    )
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
def test_get_dynamics_multiple_robots(h):
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

    A, B = get_dynamics(
        horizon=1,
        n_states=2,
        n_robots=2,
        h=h,
    )

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

@pytest.mark.parametrize("h", [0.1, 0.2, 0.5])
def test_get_dynamics_over_horizon_simple(h):
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

    A_horizon, B_horizon = get_dynamics(
        horizon=2,
        n_states=2,
        n_robots=2,
        h=h,
    )
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
# Flattening
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("p, p_flatten, v, v_flatten", [
    (
        jnp.array([[[0.0, 1.0], [2.0, -1.0]]]),
        jnp.array([[0.0], [1.0], [2.0], [-1.0]]),
        jnp.array([[[1.0, -0.5], [0.3, 0.9]]]),
        jnp.array([[1.0], [-0.5], [0.3], [0.9]])
    ),
    (
        jnp.array([[[0.0, 1.0], [2.0, -1.0], [3.0, 4.0]]]),
        jnp.array([[0.0], [1.0], [2.0], [-1.0], [3.0], [4.0]]),
        jnp.array([[[1.0, -0.5], [0.3, 0.9], [5.0, 6.0]]]),
        jnp.array([[1.0], [-0.5], [0.3], [0.9], [5.0], [6.0]])
    )
])
def test_get_flatten_state(p, p_flatten, v, v_flatten):
    # Random sample of states
    fs = FleetStateInput(v, p, jnp.zeros((0, p.shape[1], p.shape[2])))

    horizon, n_robots, n_dim = p.shape
    expected_shape = (
        2 * horizon * n_dim * n_robots + n_dim * n_robots * (horizon - 1), 1)
    flattened = fs.flatten()
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
    p = jax.random.uniform(keyp, (1, n_robots, n_dim))
    v = jax.random.uniform(keyv, (1, n_robots, n_dim))
    fs = FleetStateInput(v, p, jnp.zeros((0, n_robots, n_dim)))

    # Flatten the state
    flattened = fs.flatten()

    # Recover the state from the flattened state
    recovered_fs = fs.unpack(flattened)

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

def test_get_flatten_input():
    """Flatten the input of the fleet: [u1; u2; ...]."""
    u = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
    fsu = FleetStateInput(
        p=jnp.zeros((2, u.shape[1], u.shape[2])),
        v=jnp.zeros((2, u.shape[1], u.shape[2])), 
        u=u
    )
    u_flatten = fsu.flatten()[-u.shape[0] * u.shape[1] * u.shape[2]:]

    # Construct flatten input manually
    expected_flattened = jnp.array([1.0, 2.0, 3.0, 4.0])[..., None]  # shape (4, 1)

    assert jnp.array_equal(u_flatten, expected_flattened), (
        f"Flattened input does not match expected values. "
        f"Expected: {expected_flattened}, got: {u_flatten}."
    )

def test_get_flatten_input_horizon():
    u = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # shape (2, 2, 2)
    fsu = FleetStateInput(
        p=jnp.zeros((u.shape[0] + 1, u.shape[1], u.shape[2])),
        v=jnp.zeros((u.shape[0] + 1, u.shape[1], u.shape[2])),
        u=u
    )
    u_flatten = fsu.flatten()[-8:]
    # Construct flatten input manually
    expected_flattened = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])[..., None]  # shape (8, 1)
    assert jnp.array_equal(u_flatten, expected_flattened), (
        f"Flattened input does not match expected values. "
        f"Expected: {expected_flattened}, got: {u_flatten}."
    )

@pytest.mark.parametrize("n_robots", [1, 2, 3])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("horizon", [2, 3])
@pytest.mark.parametrize("seed", [0])
def test_get_flatten_input_horizon_multiple_robots(n_robots, n_dim, horizon, seed):
    # Random sample of states
    rng = jax.random.PRNGKey(0)
    keyv, keyp = jax.random.split(rng)
    v = jax.random.uniform(keyv, (horizon, n_robots, n_dim))
    p = jax.random.uniform(keyp, (horizon, n_robots, n_dim))
    u = jnp.arange((horizon - 1) * n_robots * n_dim).reshape(
        ((horizon - 1), n_robots, n_dim))
    fsu = FleetStateInput(v, p, u)
    # Flatten the input
    u_flatten = fsu.flatten()[-(horizon - 1) * n_robots * n_dim:]

    # Construct flatten input manually
    expected_flattened = jnp.arange((horizon - 1) * n_robots * n_dim).reshape((-1, 1))
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
    v = jax.random.uniform(keyv, (horizon, n_robots, n_dim))
    p = jax.random.uniform(keyp, (horizon, n_robots, n_dim))
    u = jax.random.uniform(keyu, (horizon - 1, n_robots, n_dim))
    fsu = FleetStateInput(v, p, u)

    u_unflatten = fsu.unpack(
        fsu.flatten()
    ).u

    # Check the shape of the unflattened input
    assert u_unflatten.shape == (horizon - 1, n_robots, n_dim), (
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
    v0 = jnp.zeros((1, n_robots, 1))
    p0 = jnp.zeros((1, n_robots, 1))
    fs = FleetStateInput(v0, p0, jnp.zeros((0, n_robots, 1)))

    # Dynamics matrices
    A, B = get_dynamics(
        horizon=1, # iterative propagation
        n_states=1,
        n_robots=n_robots,
        h=h,
    )

    # Constant inputs (per robot)
    u = jnp.array([1.0, 2.0])[..., None]  # shape (n,1)

    x = fs.flatten()[:n_robots * 2]

    for _ in range(steps):
        x = A @ x + B @ u

    # Recover per‑robot positions and velocities
    fs = fs.unpack(x)

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
    v = jax.random.uniform(keyv, (1, n_robots, n_dim))
    p = jax.random.uniform(keyp, (1, n_robots, n_dim))
    fs = FleetStateInput(v, p, jnp.zeros((0, n_robots, n_dim)))

    u = jax.random.uniform(keyu, (horizon, n_robots * n_dim))
    u_flatten = u.reshape((-1, 1), order='C')

    # Dynamics matrices
    A, B = get_dynamics(
        horizon=1,
        n_states=n_dim,
        n_robots=n_robots,
        h=h,
    )
    A_horizon, B_horizon = get_dynamics(
        horizon=horizon,
        n_states=n_dim,
        n_robots=n_robots,
        h=h,
    )

    # Initial state flattening
    x = fs.flatten()
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

# ----------------------------------------------------------------------------
# FleetStateInput masks
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("n_robots", [1, 2, 3])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("seed", [0])
def test_position_mask(n_robots, n_dim, horizon, seed):
    key = jax.random.PRNGKey(seed)
    keyp, keyv, keyu = jax.random.split(key, 3)
    p = jax.random.uniform(keyp, (horizon, n_robots, n_dim))
    v = jax.random.uniform(keyv, (horizon, n_robots, n_dim))
    u = jax.random.uniform(keyu, (horizon - 1, n_robots, n_dim))
    fsu = FleetStateInput(v, p, u)

    mask = get_position_mask(
        horizon=horizon,
        n_robots=n_robots,
        n_states=n_dim,
    )
    x = fsu.flatten()
    assert mask.shape == x.shape, (
        f"Position mask does not match expected shape. "
        f"Expected:\n{x.shape}, got:\n{mask.shape}"
    )

    # Zero out the position part of the mask
    x_masked = jnp.where(mask > 0, 0, x)
    fsu = fsu.unpack(x_masked)

    assert jnp.allclose(fsu.p, jnp.zeros_like(fsu.p)), (
        f"Position mask did not affect the position part. "
        f"Expected:\n{jnp.zeros_like(fsu.p)}, got:\n{fsu.p}"
    )
    assert jnp.allclose(fsu.v, v), (
        f"Position mask did affect the velocity part. "
        f"Expected:\n{v}, got:\n{fsu.v}"
    )
    assert jnp.allclose(fsu.u, u), (
        f"Position mask did affect the input part. "
        f"Expected:\n{u}, got:\n{fsu.u}"
    )

@pytest.mark.parametrize("n_robots", [1, 2, 3])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("seed", [0])
def test_velocity_mask(n_robots, n_dim, horizon, seed):
    key = jax.random.PRNGKey(seed)
    keyp, keyv, keyu = jax.random.split(key, 3)
    p = jax.random.uniform(keyp, (horizon, n_robots, n_dim))
    v = jax.random.uniform(keyv, (horizon, n_robots, n_dim))
    u = jax.random.uniform(keyu, (horizon - 1, n_robots, n_dim))
    fsu = FleetStateInput(v, p, u)

    mask = get_velocity_mask(
        horizon=horizon,
        n_robots=n_robots,
        n_states=n_dim,
    )
    x = fsu.flatten()
    assert mask.shape == x.shape, (
        f"Position mask does not match expected shape. "
        f"Expected:\n{x.shape}, got:\n{mask.shape}"
    )

    # Zero out the position part of the mask
    x_masked = jnp.where(mask > 0, 0, x)
    fsu = fsu.unpack(x_masked)

    assert jnp.allclose(fsu.p, p), (
        f"Position mask did affect the position part. "
        f"Expected:\n{p}, got:\n{fsu.p}"
    )
    assert jnp.allclose(fsu.v, jnp.zeros_like(fsu.v)), (
        f"Position mask did not affect the velocity part. "
        f"Expected:\n{jnp.zeros_like(fsu.v)}, got:\n{fsu.v}"
    )
    assert jnp.allclose(fsu.u, u), (
        f"Position mask did affect the input part. "
        f"Expected:\n{u}, got:\n{fsu.u}"
    )

@pytest.mark.parametrize("n_robots", [1, 2, 3])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("seed", [0])
def test_input_mask(n_robots, n_dim, horizon, seed):
    key = jax.random.PRNGKey(seed)
    keyp, keyv, keyu = jax.random.split(key, 3)
    p = jax.random.uniform(keyp, (horizon, n_robots, n_dim))
    v = jax.random.uniform(keyv, (horizon, n_robots, n_dim))
    u = jax.random.uniform(keyu, (horizon - 1, n_robots, n_dim))
    fsu = FleetStateInput(v, p, u)

    mask = get_input_mask(
        horizon=horizon,
        n_robots=n_robots,
        n_states=n_dim,
    )
    x = fsu.flatten()
    assert mask.shape == x.shape, (
        f"Position mask does not match expected shape. "
        f"Expected:\n{x.shape}, got:\n{mask.shape}"
    )

    # Zero out the position part of the mask
    x_masked = jnp.where(mask > 0, 0, x)
    fsu = fsu.unpack(x_masked)

    assert jnp.allclose(fsu.p, p), (
        f"Position mask did affect the position part. "
        f"Expected:\n{jnp.zeros_like(fsu.p)}, got:\n{fsu.p}"
    )
    assert jnp.allclose(fsu.v, v), (
        f"Position mask did affect the velocity part. "
        f"Expected:\n{v}, got:\n{fsu.v}"
    )
    assert jnp.allclose(fsu.u, jnp.zeros_like(fsu.u)), (
        f"Position mask did not affect the input part. "
        f"Expected:\n{jnp.zeros_like(fsu.u)}, got:\n{fsu.u}"
    )
