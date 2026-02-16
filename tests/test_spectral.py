import jax
import pytest
from jax import config
import e3nn_jax as e3nn
import jax.numpy as jnp
from e3nn_jax.utils import assert_equivariant

from nequix.model import (
    EquivariantSpectralLayer,
)


config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "float32")
config.update("jax_platform_name", "cpu")


@pytest.mark.quick
def test_spectral_layer_initially_outputs_zero():
    n = 10  # number of nodes
    k = 6  # number of eigenvalues
    input_irreps = 7 * e3nn.Irreps("0e + 1o + 2e")
    output_irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    x = e3nn.IrrepsArray(input_irreps, jax.random.normal(key, (n, input_irreps.dim)))
    eigvals, eigvecs = jnp.linalg.eigh(jax.random.normal(key, (n, n)))
    eigvals = eigvals[..., None]  # eigenvalues as column vector
    eigvecs = eigvecs[:, :k]  # use only first k eigenvectors
    batch_index = jnp.zeros(n, dtype=jnp.int32)

    layer = EquivariantSpectralLayer(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        model_type="s2nequix",
        key=key,
    )
    out = layer(x, eigvals, eigvecs, batch_index)

    assert jnp.allclose(out.array, 0.0)  # type: ignore


@pytest.mark.quick
def test_spectral_layer_is_equivariant():
    n = 10  # number of nodes
    k = 6  # number of eigenvalues
    input_irreps = 7 * e3nn.Irreps("0e + 1o + 2e")
    output_irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    x = e3nn.IrrepsArray(input_irreps, jax.random.normal(key, (n, input_irreps.dim)))
    eigvals, eigvecs = jnp.linalg.eigh(jax.random.normal(key, (n, n)))
    eigvals = eigvals[..., None]  # eigenvalues as column vector
    eigvecs = eigvecs[:, :k]  # use only first k eigenvectors
    batch_index = jnp.zeros(n, dtype=jnp.int32)

    layer = EquivariantSpectralLayer(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        model_type="s2nequix",
        key=key,
        init_last_layer_to_zero=False,  # note: zero initialisation would make the test trivial
    )
    function_to_test = lambda x: layer(x, eigvals, eigvecs, batch_index)

    assert_equivariant(function_to_test, key, x)  # type: ignore


@pytest.mark.quick
def test_spectral_layer_is_equivariant_with_dynamic_batching():
    ns = [10, 6, 8]  # number of nodes in each graph
    k = 6  # number of eigenvalues
    input_irreps = 7 * e3nn.Irreps("0e + 1o + 2e")
    output_irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    x = e3nn.concatenate(
        [e3nn.IrrepsArray(input_irreps, jax.random.normal(key, (n, input_irreps.dim))) for n in ns],
        axis=0,
    )
    eigdecomps = [jnp.linalg.eigh(jax.random.normal(key, (n, n))) for n in ns]
    eigvals = jnp.concatenate([val[..., None] for val, _ in eigdecomps], axis=0)
    eigvecs = jnp.concatenate([vec[:, :k] for _, vec in eigdecomps], axis=0)
    batch_index = jnp.concatenate([jnp.full((n,), i) for i, n in enumerate(ns)], axis=0)

    layer = EquivariantSpectralLayer(
        input_irreps=input_irreps,
        output_irreps=output_irreps,
        model_type="s2nequix",
        key=key,
        init_last_layer_to_zero=False,  # note: zero initialisation would make the test trivial
    )
    function_to_test = lambda x: layer(x, eigvals, eigvecs, batch_index)

    assert_equivariant(function_to_test, key, x)  # type: ignore
