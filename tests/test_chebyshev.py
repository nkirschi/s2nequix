import jax
import pytest
from jax import config
import e3nn_jax as e3nn
import jax.numpy as jnp
from e3nn_jax.utils import assert_equivariant
import equinox as eqx

from nequix.model import EquivariantChebyshevLayer


config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "float32")
config.update("jax_platform_name", "cpu")


def make_random_graph(key, num_nodes, num_edges, node_offset=0):
    """Helper to generate random sparse graph edges and weights."""
    k1, k2, k3 = jax.random.split(key, 3)
    # Generate random edges within the node range, shifted by the batch offset
    senders = jax.random.randint(k1, (num_edges,), 0, num_nodes) + node_offset
    receivers = jax.random.randint(k2, (num_edges,), 0, num_nodes) + node_offset
    # Random normalized weights in [0, 1]
    norm_weights = jax.random.uniform(k3, (num_edges,))
    return senders, receivers, norm_weights


@pytest.mark.quick
def test_chebyshev_layer_initially_outputs_zero():
    n, e = 10, 20
    k = 6
    input_irreps = 7 * e3nn.Irreps("0e + 1o + 2e")
    output_irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    x = e3nn.IrrepsArray(input_irreps, jax.random.normal(key, (n, input_irreps.dim)))
    senders, receivers, norm_weights = make_random_graph(key, n, e)

    # init_to_zero is True by default
    layer = EquivariantChebyshevLayer(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        K=k,
        key=key,
    )

    out = layer(x, norm_weights, senders, receivers)

    # Verify the output is strictly zero
    assert jnp.allclose(out.array, 0.0)


@pytest.mark.quick
def test_chebyshev_layer_is_equivariant():
    n, e = 10, 20
    k = 6
    input_irreps = 7 * e3nn.Irreps("0e + 1o + 2e")
    output_irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    k_x, k_g = jax.random.split(key)

    x = e3nn.IrrepsArray(input_irreps, jax.random.normal(k_x, (n, input_irreps.dim)))
    senders, receivers, norm_weights = make_random_graph(k_g, n, e)

    layer = EquivariantChebyshevLayer(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        K=k,
        key=key,
        init_to_zero=False,  # note: zero initialisation would make the test trivial
    )

    # assert_equivariant will apply Wigner D-matrices to `x`
    # and leave the sparse graph arrays untouched.
    function_to_test = lambda x: layer(x, norm_weights, senders, receivers)

    assert_equivariant(function_to_test, key, x)  # type: ignore


@pytest.mark.quick
def test_chebyshev_layer_is_equivariant_with_dynamic_batching():
    ns = [10, 6, 8]  # number of nodes in each graph
    es = [24, 12, 18]  # number of edges in each graph
    k = 6
    input_irreps = 7 * e3nn.Irreps("0e + 1o + 2e")
    output_irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    k_x, k_g = jax.random.split(key)

    x = e3nn.concatenate(
        [e3nn.IrrepsArray(input_irreps, jax.random.normal(k_x, (n, input_irreps.dim))) for n in ns],
        axis=0,
    )

    # Build block-diagonal sparse graph indices simulating jraph dynamic batching
    senders_list, receivers_list, weights_list = [], [], []
    node_offset = 0
    for n, e in zip(ns, es):
        k_g, k_graph = jax.random.split(k_g)
        s, r, w = make_random_graph(k_graph, n, e, node_offset=node_offset)
        senders_list.append(s)
        receivers_list.append(r)
        weights_list.append(w)
        node_offset += n

    senders = jnp.concatenate(senders_list, axis=0)
    receivers = jnp.concatenate(receivers_list, axis=0)
    norm_weights = jnp.concatenate(weights_list, axis=0)

    layer = EquivariantChebyshevLayer(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        K=k,
        key=key,
        init_to_zero=False,  # note: zero initialisation would make the test trivial
    )
    function_to_test = lambda x: layer(x, norm_weights, senders, receivers)

    assert_equivariant(function_to_test, key, x)  # type: ignore


@pytest.mark.quick
def test_chebyshev_layer_exact_message_passing():
    n = 2
    k = 2
    input_irreps = e3nn.Irreps("2x0e + 1x1o")
    output_irreps = e3nn.Irreps("2x0e + 1x1o")

    key = jax.random.key(42)
    x = e3nn.IrrepsArray(input_irreps, jax.random.normal(key, (n, input_irreps.dim)))

    # 2-node bidirectional graph with weight 1.0
    senders = jnp.array([0, 1])
    receivers = jnp.array([1, 0])
    norm_weights = jnp.array([1.0, 1.0])

    layer = EquivariantChebyshevLayer(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        K=k,
        key=key,
        init_to_zero=False,  # note: zero initialisation would make the test trivial
    )
    num_channels = output_irreps.num_irreps

    # Helper to hot-swap coefficients and run the layer
    def run_with_coeffs(c0, c1, c2):
        c = jnp.array([[c0] * num_channels, [c1] * num_channels, [c2] * num_channels])
        temp_layer = eqx.tree_at(lambda m: m.filter_coeffs, layer, c)
        return temp_layer(x, norm_weights, senders, receivers)

    out_t0 = run_with_coeffs(1.0, 0.0, 0.0)
    out_t1 = run_with_coeffs(0.0, 1.0, 0.0)
    out_t2 = run_with_coeffs(0.0, 0.0, 1.0)

    assert not jnp.allclose(out_t0.array, 0.0), "T0 output should not be zero"

    # Assert T1 correctly swapped and negated the node features
    assert jnp.allclose(out_t1.array, -out_t0.array[::-1], atol=1e-6)

    # Assert T2 correctly collapsed back to the original T0 features
    assert jnp.allclose(out_t2.array, out_t0.array, atol=1e-6)
