"""Benchmark cuequivariance vs e3nn-jax for Nequix operations on H200.

Tests the two operations from nequix.py that cuequivariance could accelerate:
  1. Channelwise tensor product  (NequixConvolution message-passing step)
  2. Indexed linear              (per-species skip connection)

Also benchmarks the full Nequix forward pass to estimate end-to-end impact.
"""

# Preload JAX's bundled libcublas (provides cublasGemmGroupedBatchedEx which
# the system libcublas lacks) and the cue_ops native lib so that cuequivariance
# can load without the user having to set LD_LIBRARY_PATH.
import ctypes
import os
import sysconfig

_site = sysconfig.get_paths()["purelib"]
for _sub in ("nvidia/cublas/lib/libcublas.so.12", "cuequivariance_ops/lib/libcue_ops.so"):
    _p = os.path.join(_site, _sub)
    if os.path.exists(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

import time
import math
from typing import Callable

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import equinox as eqx
import jraph

import cuequivariance as cue
import cuequivariance_jax as cuex

from nequix.nequix import Nequix, bessel_basis, polynomial_cutoff

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_NODES = 1_000          # nodes in synthetic graph
N_EDGES = 20_000         # edges (avg 20 neighbours)
N_SPECIES = 10
CHANNELS = 128
LMAX = 3
HIDDEN_IRREPS = f"{CHANNELS}x0e + {CHANNELS}x1o + {CHANNELS}x2e + {CHANNELS}x3o"
N_WARMUP = 5
N_REPEATS = 20

jax.config.update("jax_default_matmul_precision", "float32")

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def benchmark(fn: Callable, n_warmup: int = N_WARMUP, n_repeats: int = N_REPEATS) -> dict:
    """Run fn(), warm-up then time n_repeats calls. Returns ms stats."""
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(out)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e3)

    arr = jnp.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms":  float(arr.std()),
        "min_ms":  float(arr.min()),
    }


def fmt(stats: dict) -> str:
    return f"{stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms  (min {stats['min_ms']:.2f} ms)"


# ---------------------------------------------------------------------------
# Synthetic graph
# ---------------------------------------------------------------------------
rng = jax.random.PRNGKey(0)

positions = jax.random.normal(rng, (N_NODES, 3))
species   = jax.random.randint(rng, (N_NODES,), 0, N_SPECIES)
senders   = jax.random.randint(rng, (N_EDGES,), 0, N_NODES)
receivers = jax.random.randint(rng, (N_EDGES,), 0, N_NODES)

displacements = positions[senders] - positions[receivers]
r_norm = jnp.linalg.norm(displacements, axis=-1)

CUTOFF = 5.0
radial_basis = bessel_basis(r_norm, 8, CUTOFF) * polynomial_cutoff(r_norm, CUTOFF, 2.0)[:, None]

sh_irreps = e3nn.s2_irreps(LMAX)
sh = e3nn.spherical_harmonics(sh_irreps, displacements, normalize=True, normalization="component")

hidden_irreps = e3nn.Irreps(HIDDEN_IRREPS)
features_e3nn = e3nn.IrrepsArray(hidden_irreps, jax.random.normal(rng, (N_NODES, hidden_irreps.dim)))

print(f"Graph: {N_NODES} nodes, {N_EDGES} edges")
print(f"Hidden irreps: {hidden_irreps}  (dim={hidden_irreps.dim})")
print(f"SH irreps:     {sh_irreps}  (dim={sh_irreps.dim})")
print()

# ---------------------------------------------------------------------------
# 1. Channelwise tensor product
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. CHANNELWISE TENSOR PRODUCT (message passing core)")
print("=" * 70)

tp_filter_irreps = e3nn.tensor_product(hidden_irreps, sh_irreps, filter_ir_out=hidden_irreps)
n_tp_paths = tp_filter_irreps.num_irreps  # number of radial weights per edge

# Fake radial weights for TP
radial_weights = jax.random.normal(rng, (N_EDGES, n_tp_paths))

# -- e3nn baseline --
msg_e3nn = features_e3nn[senders]

@jax.jit
def tp_e3nn():
    msgs = e3nn.tensor_product(msg_e3nn, sh, filter_ir_out=hidden_irreps)
    return (msgs * radial_weights).array

stats_tp_e3nn = benchmark(tp_e3nn)
print(f"  e3nn   : {fmt(stats_tp_e3nn)}")

# -- cuequivariance --
try:
    cue_irreps_in  = cue.Irreps("O3", str(hidden_irreps))
    cue_irreps_sh  = cue.Irreps("O3", str(sh_irreps))
    cue_irreps_out = cue.Irreps("O3", str(tp_filter_irreps))

    ctp_desc = cue.descriptors.channelwise_tensor_product(
        cue_irreps_in, cue_irreps_sh, cue_irreps_out
    )
    ctp_poly = ctp_desc.polynomial

    feat_arr = features_e3nn[senders].array
    sh_arr   = sh.array

    # cuex channelwise TP expects operand order [weights, features, sh_feats]
    #   a[uv] (weights, 23⨯128⨯1), b[iu] (features, 128ch), c[jv] (sh, 1ch)
    # Radial-weights buffer needs to match operand-0 flat size.
    n_weight_elems = ctp_poly.operands[0].size
    radial_weights_cuex = jax.random.normal(rng, (N_EDGES, n_weight_elems))

    out_sdf = [jax.ShapeDtypeStruct((N_EDGES, ctp_poly.operands[-1].size), jnp.float32)]

    @jax.jit
    def tp_cuex():
        out, = cuex.segmented_polynomial(
            ctp_poly,
            [radial_weights_cuex, feat_arr, sh_arr],
            out_sdf,
            method="uniform_1d",
        )
        return out

    stats_tp_cuex = benchmark(tp_cuex)
    print(f"  cuex   : {fmt(stats_tp_cuex)}")
    speedup_tp = stats_tp_e3nn["mean_ms"] / stats_tp_cuex["mean_ms"]
    print(f"  speedup: {speedup_tp:.2f}x")

except Exception as e:
    print(f"  cuex   : FAILED — {e}")
    speedup_tp = None

print()

# ---------------------------------------------------------------------------
# 2. Indexed linear (per-species skip connection)
# ---------------------------------------------------------------------------
print("=" * 70)
print("2. INDEXED LINEAR (per-species skip connection)")
print("=" * 70)

# Build a small model just to get a real skip weight tensor
_key = jax.random.PRNGKey(42)
skip_e3nn = e3nn.equinox.Linear(
    irreps_in=hidden_irreps,
    irreps_out=hidden_irreps,
    linear_type="indexed",
    num_indexed_weights=N_SPECIES,
    force_irreps_out=True,
    key=_key,
)

@jax.jit
def skip_e3nn_fn():
    return skip_e3nn(species, features_e3nn)

stats_skip_e3nn = benchmark(skip_e3nn_fn)
print(f"  e3nn   : {fmt(stats_skip_e3nn)}")

try:
    cue_irreps_hid = cue.Irreps("O3", str(hidden_irreps))
    lin_desc = cue.descriptors.linear(cue_irreps_hid, cue_irreps_hid)
    lin_poly = lin_desc.polynomial

    # Per-species counts: sort nodes by species so consecutive chunks share a species,
    # which is what cuex's indexed_linear kernel expects (IndexingMode.REPEATED).
    sort_idx = jnp.argsort(species)
    species_sorted = species[sort_idx]
    species_counts = jnp.bincount(species_sorted, length=N_SPECIES).astype(jnp.int32)
    feat_arr_nodes = features_e3nn.array[sort_idx]  # (N_NODES, dim)

    # Weight buffer matches first operand of the linear polynomial
    w_cuex = jax.random.normal(rng, (N_SPECIES, lin_poly.operands[0].size))
    out_sdf_skip = [jax.ShapeDtypeStruct((N_NODES, hidden_irreps.dim), jnp.float32)]

    @jax.jit
    def skip_cuex_fn():
        out, = cuex.segmented_polynomial(
            lin_poly,
            [w_cuex, feat_arr_nodes],
            out_sdf_skip,
            indices=[cuex.Repeats(species_counts), None, None],
            method="indexed_linear",
        )
        return out

    stats_skip_cuex = benchmark(skip_cuex_fn)
    print(f"  cuex   : {fmt(stats_skip_cuex)}")
    speedup_skip = stats_skip_e3nn["mean_ms"] / stats_skip_cuex["mean_ms"]
    print(f"  speedup: {speedup_skip:.2f}x")

except Exception as e:
    import traceback
    print(f"  cuex   : FAILED — {type(e).__name__}: {e}")
    traceback.print_exc()
    speedup_skip = None

print()

# ---------------------------------------------------------------------------
# 3. Full Nequix forward pass
# ---------------------------------------------------------------------------
print("=" * 70)
print("3. FULL NEQUIX FORWARD PASS  (end-to-end, 5 layers)")
print("=" * 70)

n_node = jnp.array([N_NODES, 1])
n_edge = jnp.array([N_EDGES, 0])
dummy_pos = jnp.concatenate([positions, jnp.zeros((1, 3))], axis=0)
dummy_spe = jnp.concatenate([species, jnp.zeros(1, dtype=jnp.int32)], axis=0)
dummy_snd = jnp.concatenate([senders, jnp.array([N_NODES])], axis=0)
dummy_rcv = jnp.concatenate([receivers, jnp.array([N_NODES])], axis=0)

graph = jraph.GraphsTuple(
    nodes={"positions": dummy_pos, "species": dummy_spe, "eigvecs": None},
    edges={},
    globals={"cell": None, "eigvals": None},
    senders=dummy_snd,
    receivers=dummy_rcv,
    n_node=n_node,
    n_edge=n_edge,
)

model = Nequix(
    key=jax.random.PRNGKey(0),
    n_species=N_SPECIES,
    lmax=LMAX,
    hidden_irreps=HIDDEN_IRREPS,
    n_layers=5,
    spatial_cutoff=CUTOFF,
    radial_basis_size=8,
    radial_mlp_size=64,
    radial_mlp_layers=3,
    avg_n_neighbors=20.0,
)

@jax.jit
def nequix_fwd():
    return model(graph)

stats_nequix = benchmark(nequix_fwd)
print(f"  Nequix (e3nn, 5 layers): {fmt(stats_nequix)}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Channelwise TP  — e3nn: {stats_tp_e3nn['mean_ms']:.2f} ms", end="")
if speedup_tp:
    print(f"  |  cuex: {stats_tp_cuex['mean_ms']:.2f} ms  |  {speedup_tp:.2f}x")
else:
    print()
print(f"  Indexed linear  — e3nn: {stats_skip_e3nn['mean_ms']:.2f} ms", end="")
if speedup_skip:
    print(f"  |  cuex: {stats_skip_cuex['mean_ms']:.2f} ms  |  {speedup_skip:.2f}x")
else:
    print()
print(f"  Nequix fwd (full, 5 layers): {stats_nequix['mean_ms']:.2f} ms")

# Estimate end-to-end speedup from profiling fractions
# TP dominates message passing, skip is small; spectral is irrelevant to cuex
print()
print("  Note: cuequivariance has no equivalent for gated activations,")
print("  spectral layers (EVD/Chebyshev), or spherical harmonics in JAX.")
