import jax
import pytest
from jax import config
import e3nn_jax as e3nn
import jax.numpy as jnp
from e3nn_jax.utils import assert_equivariant

from nequix.model import EquivariantSpectralLayer


config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "float32")
config.update("jax_platform_name", "cpu")


@pytest.mark.quick
def test_spectral_layer_initially_outputs_zero():
    n = 10  # number of nodes
    k = 6  # number of eigenvalues
    irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    x = e3nn.IrrepsArray(irreps, jax.random.normal(key, (n, irreps.dim)))
    eigvals, eigvecs = jnp.linalg.eigh(jax.random.normal(key, (n, n)))
    eigvals = eigvals = eigvals[:k][None, :]  # batch of single graph
    eigvecs = eigvecs[:, :k]  # use only first k eigenvectors
    batch_index = jnp.zeros(n, dtype=jnp.int32)

    layer = EquivariantSpectralLayer(
        key=key,
        irreps=irreps,
        pretransform_feats=False,
        init_to_zero=True,
    )
    out = layer(x, eigvals, eigvecs, batch_index)

    assert jnp.allclose(out.array, 0.0)  # type: ignore


@pytest.mark.quick
def test_spectral_layer_is_equivariant():
    n = 10  # number of nodes
    k = 6  # number of eigenvalues
    irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    x = e3nn.IrrepsArray(irreps, jax.random.normal(key, (n, irreps.dim)))
    eigvals, eigvecs = jnp.linalg.eigh(jax.random.normal(key, (n, n)))
    eigvals = eigvals = eigvals[:k][None, :]  # batch of single graph
    eigvecs = eigvecs[:, :k]  # use only first k eigenvectors
    batch_index = jnp.zeros(n, dtype=jnp.int32)

    layer = EquivariantSpectralLayer(
        irreps=irreps,
        key=key,
        pretransform_feats=True,
        init_to_zero=False,  # note: zero initialisation would make the test trivial
    )
    function_to_test = lambda x: layer(x, eigvals, eigvecs, batch_index)

    assert_equivariant(function_to_test, key, x)  # type: ignore


@pytest.mark.quick
def test_spectral_layer_is_equivariant_with_dynamic_batching():
    ns = [10, 6, 8]  # number of nodes in each graph
    k = 6  # number of eigenvalues
    irreps = 11 * e3nn.Irreps("4x0e + 2x1o + 1x2e")

    key = jax.random.key(42)
    x = e3nn.concatenate(
        [e3nn.IrrepsArray(irreps, jax.random.normal(key, (n, irreps.dim))) for n in ns],
        axis=0,
    )
    eigdecomps = [jnp.linalg.eigh(jax.random.normal(key, (n, n))) for n in ns]
    eigvals = jnp.stack([val[:k] for val, _ in eigdecomps], axis=0)
    eigvecs = jnp.concatenate([vec[:, :k] for _, vec in eigdecomps], axis=0)
    batch_index = jnp.concatenate([jnp.full((n,), i) for i, n in enumerate(ns)], axis=0)

    layer = EquivariantSpectralLayer(
        irreps=irreps,
        key=key,
        pretransform_feats=True,
        init_to_zero=False,  # note: zero initialisation would make the test trivial
    )
    function_to_test = lambda x: layer(x, eigvals, eigvecs, batch_index)

    assert_equivariant(function_to_test, key, x)  # type: ignore
