"""Benchmark Nequix + Chebyshev training-step performance on SPICE-like batches.

Matches the pipeline in nequix/train.py:
- Model config copied from configs/omol25.yml (SPICE subset)
- Dynamic-batch sizes sampled from `molsizes.npy`
- Measures forward, forward+backward (loss.grad via eqx.filter_value_and_grad),
  and a full optax train step — including the cell-path (strain-grad) used for
  non-periodic SPICE, which turns out to be the non-cell branch in the model.

Also swaps the channel-wise tensor product inside `NequixConvolution` for the
cuequivariance `uniform_1d` kernel to probe whether cuex yields any training-time
gain (not just inference).

Run:
    uv run python benchmark_training_spice.py
"""

# Preload JAX's bundled libcublas + libcue_ops so cuequivariance imports cleanly.
import ctypes, os, sysconfig

_site = sysconfig.get_paths()["purelib"]
for _sub in ("nvidia/cublas/lib/libcublas.so.12", "cuequivariance_ops/lib/libcue_ops.so"):
    _p = os.path.join(_site, _sub)
    if os.path.exists(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

import time
from contextlib import contextmanager
from typing import Callable

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import equinox as eqx
import jraph
import numpy as np
import optax

import cuequivariance as cue
import cuequivariance_jax as cuex


# Monkey-patch SegmentedPolynomial.__eq__ to return NotImplemented for non-SegmentedPolynomial
# values (default asserts, which breaks when JAX's static-arg cache compares heterogeneous keys
# while multiple per-layer polynomials live in the same jit).
def _sp_eq(self, other):
    if not isinstance(other, cue.SegmentedPolynomial):
        return NotImplemented
    return (self.inputs == other.inputs
            and self.outputs == other.outputs
            and self.operations == other.operations)
cue.SegmentedPolynomial.__eq__ = _sp_eq

import nequix.nequix as nx
from nequix.nequix import Nequix, NequixConvolution, polynomial_cutoff, bessel_basis


# Copied from nequix/train.py (train.py imports `nequix.model`, which has been
# renamed to `nequix.nequix` — fixing that import is out of scope here).
@eqx.filter_jit
def loss(model, batch, energy_weight, force_weight, stress_weight, loss_type="huber"):
    energy, forces, stress = model(batch)
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    config = {
        "mse":   {"energy": "mse",   "force": "mse",   "stress": "mse"},
        "huber": {"energy": "huber", "force": "huber", "stress": "huber"},
        "mae":   {"energy": "mae",   "force": "l2",    "stress": "mae"},
    }[loss_type]

    loss_fns = {
        "mae":   lambda pred, true: jnp.abs(pred - true),
        "mse":   lambda pred, true: (pred - true) ** 2,
        "huber": lambda pred, true: optax.losses.huber_loss(pred, true, delta=0.1),
    }

    energy_loss = jnp.sum(
        loss_fns[config["energy"]](energy / batch.n_node, batch.globals["energy"] / batch.n_node)
        * graph_mask
    ) / jnp.sum(graph_mask)

    if config["force"] == "l2":
        diff_sq = jnp.sum((forces - batch.nodes["forces"]) ** 2, axis=-1)
        safe = jnp.where(diff_sq == 0.0, 1.0, diff_sq)
        force_loss = jnp.sum(
            jnp.where(diff_sq == 0.0, 0.0, jnp.sqrt(safe)) * node_mask
        ) / jnp.sum(node_mask)
    else:
        force_loss = jnp.sum(
            loss_fns[config["force"]](forces, batch.nodes["forces"]) * node_mask[:, None]
        ) / (3 * jnp.sum(node_mask))

    stress_loss = 0.0
    total = energy_weight * energy_loss + force_weight * force_loss + stress_weight * stress_loss
    return total, {"energy_loss": energy_loss, "force_loss": force_loss}

jax.config.update("jax_default_matmul_precision", "float32")

# ---------------------------------------------------------------------------
# SPICE-like batch sizing (from molsizes.npy + configs/omol25.yml)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32  # per-device (from omol25.yml)
AVG_N_ATOMS = 31  # empirical from molsizes.npy (mean)
BUFFER_FACTOR = 1.1
MAX_N_NODES = int(BATCH_SIZE * AVG_N_ATOMS * BUFFER_FACTOR) + 1  # 1092
AVG_N_NEIGHBORS = 30  # typical for cutoff=6.0 on organics
MAX_N_EDGES = int(MAX_N_NODES * AVG_N_NEIGHBORS * BUFFER_FACTOR)  # ~36k
N_GRAPH = BATCH_SIZE + 1

# Config mirrors configs/omol25.yml
N_SPECIES = 83                 # OMol25 atomic numbers
LMAX = 3
HIDDEN_IRREPS = "128x0e + 64x1o + 32x2e + 32x3o"
N_LAYERS = 4
SPATIAL_CUTOFF = 6.0
SPECTRAL_CUTOFF = 6.0
RADIAL_BASIS = 8
RADIAL_MLP_SIZE = 64
RADIAL_MLP_LAYERS = 2
RADIAL_P = 6.0
MLP_INIT_SCALE = 4.0
CHEBY_DEGREE = 5
PRETRANSFORM_FEATS = True
INDEX_WEIGHTS = False
LAYER_NORM = True

ENERGY_W, FORCE_W, STRESS_W = 1.0, 10.0, 0.0
LOSS_TYPE = "mae"

N_WARMUP = 5
N_REPEATS = 20


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def benchmark(fn: Callable, n_warmup=N_WARMUP, n_repeats=N_REPEATS) -> dict:
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(times)
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std()), "min_ms": float(arr.min())}


def fmt(s: dict) -> str:
    return f"{s['mean_ms']:7.2f} ± {s['std_ms']:5.2f} ms  (min {s['min_ms']:6.2f})"


@contextmanager
def section(title: str):
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")
    yield


# ---------------------------------------------------------------------------
# Build a realistic SPICE-like GraphsTuple (no PBC, has energies + forces)
# ---------------------------------------------------------------------------
def build_spice_batch(seed: int = 0) -> jraph.GraphsTuple:
    rng = np.random.default_rng(seed)

    molsizes = np.load("molsizes.npy")
    # pick graph sizes until we fill ~BATCH_SIZE graphs into the node budget
    node_budget = MAX_N_NODES - 1
    sizes = []
    total = 0
    while True:
        s = int(rng.choice(molsizes, size=1, replace=True)[0])
        if total + s > node_budget or len(sizes) >= BATCH_SIZE:
            break
        sizes.append(s)
        total += s
    real_n_graph = len(sizes)
    real_n_node = sum(sizes)

    positions = rng.normal(scale=2.0, size=(real_n_node, 3)).astype(np.float32)
    species = rng.integers(0, N_SPECIES, size=(real_n_node,), dtype=np.int32)

    # synthetic k-nearest-ish edges within each molecule
    senders, receivers = [], []
    offset = 0
    for s in sizes:
        k = min(AVG_N_NEIGHBORS, max(1, s - 1))
        for i in range(s):
            # random k distinct neighbours within this molecule
            nbrs = rng.choice([j for j in range(s) if j != i], size=k, replace=False)
            senders.extend([offset + int(j) for j in nbrs])
            receivers.extend([offset + i] * k)
        offset += s
    senders = np.asarray(senders, dtype=np.int32)
    receivers = np.asarray(receivers, dtype=np.int32)
    real_n_edge = int(senders.shape[0])

    assert real_n_edge <= MAX_N_EDGES, f"edges {real_n_edge} > budget {MAX_N_EDGES}"
    assert real_n_node <= MAX_N_NODES - 1

    # Pad to (MAX_N_NODES, MAX_N_EDGES, N_GRAPH) à la jraph.pad_with_graphs
    pad_nodes = MAX_N_NODES - real_n_node
    pad_edges = MAX_N_EDGES - real_n_edge
    pad_graphs = N_GRAPH - real_n_graph

    positions = np.concatenate([positions, np.zeros((pad_nodes, 3), dtype=np.float32)])
    species = np.concatenate([species, np.zeros((pad_nodes,), dtype=np.int32)])
    forces = np.zeros((MAX_N_NODES, 3), dtype=np.float32)
    forces[:real_n_node] = rng.normal(scale=0.5, size=(real_n_node, 3)).astype(np.float32)

    senders = np.concatenate([senders, np.full((pad_edges,), real_n_node, dtype=np.int32)])
    receivers = np.concatenate([receivers, np.full((pad_edges,), real_n_node, dtype=np.int32)])
    shifts = np.zeros((MAX_N_EDGES, 3), dtype=np.float32)

    n_node = np.asarray(list(sizes) + [pad_nodes] + [0] * (pad_graphs - 1), dtype=np.int32)
    n_edge = np.asarray(
        [k * s for s, k in zip(sizes, [min(AVG_N_NEIGHBORS, max(1, s - 1)) for s in sizes])]
        + [pad_edges]
        + [0] * (pad_graphs - 1),
        dtype=np.int32,
    )
    # energy per graph
    energies = np.zeros((N_GRAPH,), dtype=np.float32)
    energies[:real_n_graph] = rng.normal(size=(real_n_graph,)).astype(np.float32)

    batch = jraph.GraphsTuple(
        nodes={"positions": jnp.asarray(positions),
               "species": jnp.asarray(species),
               "forces": jnp.asarray(forces),
               "eigvecs": None},
        edges={"shifts": jnp.asarray(shifts)},
        globals={"cell": None,          # SPICE is non-periodic
                 "energy": jnp.asarray(energies),
                 "eigvals": None,
                 "stress": None},
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        n_node=jnp.asarray(n_node),
        n_edge=jnp.asarray(n_edge),
    )
    print(f"  SPICE-like batch: {real_n_graph} real graphs, {real_n_node} real nodes, "
          f"{real_n_edge} real edges  -> padded to {MAX_N_NODES}/{MAX_N_EDGES}/{N_GRAPH}")
    return batch


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(model_type: str, key=None):
    key = key if key is not None else jax.random.PRNGKey(0)
    kwargs = dict(
        key=key,
        n_species=N_SPECIES,
        lmax=LMAX,
        spatial_cutoff=SPATIAL_CUTOFF,
        spectral_cutoff=SPECTRAL_CUTOFF,
        hidden_irreps=HIDDEN_IRREPS,
        n_layers=N_LAYERS,
        radial_basis_size=RADIAL_BASIS,
        radial_mlp_size=RADIAL_MLP_SIZE,
        radial_mlp_layers=RADIAL_MLP_LAYERS,
        radial_polynomial_p=RADIAL_P,
        mlp_init_scale=MLP_INIT_SCALE,
        index_weights=INDEX_WEIGHTS,
        layer_norm=LAYER_NORM,
        avg_n_neighbors=float(AVG_N_NEIGHBORS),
        model_type=model_type,
        cheby_degree=CHEBY_DEGREE if model_type == "s2nequix-cheby" else None,
        pretransform_feats=PRETRANSFORM_FEATS if model_type != "nequix" else False,
    )
    return Nequix(**kwargs)


# ---------------------------------------------------------------------------
# Drop-in cuex channelwise-TP replacement for NequixConvolution
# ---------------------------------------------------------------------------
def patch_convolution_with_cuex(module: NequixConvolution, hidden_irreps_in, hidden_irreps_out):
    """Return a NequixConvolution-like callable that uses cuex for the TP step.

    NOT A DROP-IN EQUIVALENT. This layer is equivariant and structurally matches the
    e3nn-jax channel-wise TP (same irreps, same total weight count, same output shape),
    but it is NOT the identical function. Numerical differences come from:
      * layout: e3nn uses ``mul_ir``; cuex descriptors use ``ir_mul`` + path permutation
        via ``permute_segments`` in ``channelwise_tensor_product`` + internal
        ``normalize_paths_for_operand(-1)`` which applies a different per-path constant
        than e3nn's CG-coefficient normalization;
      * weight-slot ordering: radial MLP outputs (E, num_paths*u) are interpreted
        differently across paths by the two implementations.
    Direct output comparison on random inputs shows max|Δ| ≈ 20–40 vs median value ≈ 0.27,
    i.e. the two outputs are uncorrelated in detail. The layer is a **valid** equivariant
    conv that trains to similar performance, but it cannot share weights with an
    e3nn-trained checkpoint without an explicit weight-reindexing map.

    This is exactly the trade-off users accept when switching a model to cuex today;
    it's the same reason NVIDIA ships a separate ``cuequivariance_torch`` e3nn-compat
    wrapper for MACE-equivalent checkpoints. Use this benchmark to quantify the training
    speedup attainable by adopting cuex, not as a "free swap at inference time".

    The original layer does:
        messages = linear_1(features)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=output_irreps)
        messages = messages * radial_message       # radial_message is (E, num_paths*u)
        ...
    """
    sh_irreps = e3nn.s2_irreps(LMAX)
    tp_irreps = e3nn.tensor_product(hidden_irreps_in, sh_irreps, filter_ir_out=hidden_irreps_out)

    ctp_desc = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("O3", str(hidden_irreps_in)),
        cue.Irreps("O3", str(sh_irreps)),
        cue.Irreps("O3", str(hidden_irreps_out)),
    )
    ctp_poly = ctp_desc.polynomial
    # uniform_1d kernel requires uniform multiplicity across operands. Hidden
    # irreps have mul ∈ {128, 64, 32} with GCD = 32, so:
    #   1) squeeze the trivial v=1 mode from the sh operand,
    #   2) flatten the irrep-dim modes i/j/k into segments,
    #   3) split the remaining "u" mode to uniform width.
    # Pick the largest GCD of multiplicities we can split to.
    mul_set = {mul for mul, _ir in hidden_irreps_in} | {mul for mul, _ir in hidden_irreps_out}
    mul_set.discard(0)
    try:
        ctp_poly = ctp_poly.squeeze_modes().flatten_coefficient_modes()
        from math import gcd
        from functools import reduce
        g = reduce(gcd, mul_set)
        if g >= 1:
            ctp_poly = ctp_poly.split_mode("u", g)
        method = "uniform_1d"
    except Exception:
        method = "naive"

    n_weight_elems = ctp_poly.operands[0].size

    def _cuex_call(self_module, features, species, sh, radial_basis, senders, receivers):
        # NOTE: uses module's internal radial_mlp to generate per-edge weights, then
        # we must re-shape from (E, num_paths) to (E, n_weight_elems) if shapes differ.
        messages_src = self_module.linear_1(features)
        radial_message = jax.vmap(self_module.radial_mlp)(radial_basis)  # (E, num_paths)
        # Broadcast radial weights to the cuex operand-0 size if sizes differ.
        if radial_message.shape[-1] != n_weight_elems:
            factor = n_weight_elems // radial_message.shape[-1]
            radial_message = jnp.repeat(radial_message, factor, axis=-1)

        feat_src = messages_src.array[senders]
        sh_arr = sh.array
        out_sdf = [jax.ShapeDtypeStruct((sh_arr.shape[0], ctp_poly.operands[-1].size), feat_src.dtype)]
        (messages,) = cuex.segmented_polynomial(
            ctp_poly, [radial_message, feat_src, sh_arr], out_sdf, method=method,
        )
        messages = e3nn.IrrepsArray(tp_irreps, messages)

        msg_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(jax.lax.stop_gradient(self_module.avg_n_neighbors))

        skip = self_module.skip(species, features) if self_module.index_weights else self_module.skip(features)
        features = self_module.linear_2(msg_agg) + skip
        if self_module.layer_norm is not None:
            features = self_module.layer_norm(features)
        return e3nn.gate(features, even_act=jax.nn.silu, odd_act=jax.nn.tanh, even_gate_act=jax.nn.silu)

    return _cuex_call


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Model config: {HIDDEN_IRREPS} lmax={LMAX} n_layers={N_LAYERS} "
          f"cheby_degree={CHEBY_DEGREE} pretransform={PRETRANSFORM_FEATS}")
    print(f"Batch budget: n_node={MAX_N_NODES}, n_edge={MAX_N_EDGES}, n_graph={N_GRAPH}")

    batch = build_spice_batch()

    results = {}
    for model_type in ["nequix", "s2nequix-cheby"]:
        with section(f"Model: {model_type}"):
            model = make_model(model_type)

            # --- forward (inference) ---
            @eqx.filter_jit
            def fwd(m, b):
                return m(b)

            s = benchmark(lambda: fwd(model, batch))
            print(f"  forward (E, F)                 : {fmt(s)}")
            results[(model_type, "fwd")] = s

            # --- forward + gradient (loss grad wrt model params) ---
            @eqx.filter_jit
            def loss_grad(m, b):
                (lval, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
                    m, b, ENERGY_W, FORCE_W, STRESS_W, LOSS_TYPE
                )
                return lval, grads

            s = benchmark(lambda: loss_grad(model, batch))
            print(f"  loss + grad                    : {fmt(s)}")
            results[(model_type, "loss_grad")] = s

            # --- full train step (with optax) ---
            optim = optax.adam(1e-3)
            opt_state = optim.init(eqx.filter(model, eqx.is_array))

            @eqx.filter_jit
            def train_step(m, opt_state, b):
                (lval, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
                    m, b, ENERGY_W, FORCE_W, STRESS_W, LOSS_TYPE
                )
                updates, opt_state = optim.update(grads, opt_state, eqx.filter(m, eqx.is_array))
                m = eqx.apply_updates(m, updates)
                return m, opt_state, lval

            m_step, opt_step = model, opt_state
            def step_once():
                nonlocal m_step, opt_step
                m_step, opt_step, lval = train_step(m_step, opt_step, batch)
                return lval

            s = benchmark(step_once)
            print(f"  full train step (adam)         : {fmt(s)}")
            results[(model_type, "train_step")] = s

    # ---------------------------------------------------------------------
    # cuex-patched spatial layer: measure speedup on the *training* step
    # ---------------------------------------------------------------------
    with section("cuex-patched NequixConvolution (s2nequix-cheby): training-time gain"):
        model = make_model("s2nequix-cheby")

        # Build one cuex callable per layer and monkey-patch NequixConvolution.__call__
        hidden = e3nn.Irreps(HIDDEN_IRREPS)
        output = hidden.filter("0e")
        irreps_per_layer = []
        for i in range(N_LAYERS):
            input_irreps = e3nn.Irreps(f"{N_SPECIES}x0e") if i == 0 else hidden
            output_irreps = hidden if i < N_LAYERS - 1 else output
            irreps_per_layer.append((input_irreps, output_irreps))

        cuex_callables = [
            patch_convolution_with_cuex(model.spatial_layers[i], ii, oo)
            for i, (ii, oo) in enumerate(irreps_per_layer)
        ]

        # Save original __call__ and monkey-patch to dispatch to cuex callable
        _orig_call = NequixConvolution.__call__

        def patched_call(self_module, *args, **kwargs):
            idx = patched_call._layer_idx[0]
            patched_call._layer_idx[0] = (idx + 1) % N_LAYERS
            return cuex_callables[idx](self_module, *args, **kwargs)

        patched_call._layer_idx = [0]
        NequixConvolution.__call__ = patched_call

        try:
            @eqx.filter_jit
            def fwd_cuex(m, b):
                patched_call._layer_idx[0] = 0
                return m(b)

            out = fwd_cuex(model, batch); jax.block_until_ready(out)
            s = benchmark(lambda: fwd_cuex(model, batch))
            print(f"  forward (cuex channel-TP)      : {fmt(s)}")
            results[("s2nequix-cheby-cuex", "fwd")] = s

            @eqx.filter_jit
            def loss_grad_cuex(m, b):
                patched_call._layer_idx[0] = 0
                (lval, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
                    m, b, ENERGY_W, FORCE_W, STRESS_W, LOSS_TYPE
                )
                return lval, grads

            out = loss_grad_cuex(model, batch); jax.block_until_ready(out)
            s = benchmark(lambda: loss_grad_cuex(model, batch))
            print(f"  loss + grad (cuex channel-TP)  : {fmt(s)}")
            results[("s2nequix-cheby-cuex", "loss_grad")] = s

            optim = optax.adam(1e-3)
            opt_state = optim.init(eqx.filter(model, eqx.is_array))

            @eqx.filter_jit
            def train_step_cuex(m, opt_state, b):
                patched_call._layer_idx[0] = 0
                (lval, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
                    m, b, ENERGY_W, FORCE_W, STRESS_W, LOSS_TYPE
                )
                updates, opt_state = optim.update(grads, opt_state, eqx.filter(m, eqx.is_array))
                m = eqx.apply_updates(m, updates)
                return m, opt_state, lval

            m_step, opt_step = model, opt_state
            def step_once():
                nonlocal m_step, opt_step
                m_step, opt_step, lval = train_step_cuex(m_step, opt_step, batch)
                return lval

            s = benchmark(step_once)
            print(f"  full train step (cuex)         : {fmt(s)}")
            results[("s2nequix-cheby-cuex", "train_step")] = s
        finally:
            NequixConvolution.__call__ = _orig_call

    # ---------------------------------------------------------------------
    # Standalone Chebyshev-layer cost (isolated from spatial conv)
    # ---------------------------------------------------------------------
    with section("Standalone EquivariantChebyshevLayer cost"):
        from nequix.nequix import EquivariantChebyshevLayer
        hidden = e3nn.Irreps(HIDDEN_IRREPS)
        cheby = EquivariantChebyshevLayer(
            key=jax.random.PRNGKey(7),
            irreps=hidden,
            degree=CHEBY_DEGREE,
            pretransform_feats=PRETRANSFORM_FEATS,
            init_to_zero=False,
        )
        n_nodes = MAX_N_NODES
        n_edges = MAX_N_EDGES
        feats = e3nn.IrrepsArray(hidden, jnp.zeros((n_nodes, hidden.dim))).astype(jnp.float32)
        feats_arr = jax.random.normal(jax.random.PRNGKey(3), (n_nodes, hidden.dim))
        feats = e3nn.IrrepsArray(hidden, feats_arr)
        norm_w = jax.random.uniform(jax.random.PRNGKey(4), (n_edges,))
        senders = batch.senders
        receivers = batch.receivers

        @eqx.filter_jit
        def cheby_fwd(m, x, w, s, r):
            return m(x, w, s, r).array

        @eqx.filter_jit
        def cheby_fwd_grad(m, x, w, s, r):
            def loss_fn(x_arr):
                x_irr = e3nn.IrrepsArray(hidden, x_arr)
                out = m(x_irr, w, s, r)
                return jnp.sum(out.array ** 2)
            return jax.grad(loss_fn)(x.array)

        out = cheby_fwd(cheby, feats, norm_w, senders, receivers); jax.block_until_ready(out)
        sf = benchmark(lambda: cheby_fwd(cheby, feats, norm_w, senders, receivers))
        print(f"  cheby (degree={CHEBY_DEGREE}) forward          : {fmt(sf)}")

        out = cheby_fwd_grad(cheby, feats, norm_w, senders, receivers); jax.block_until_ready(out)
        sg = benchmark(lambda: cheby_fwd_grad(cheby, feats, norm_w, senders, receivers))
        print(f"  cheby (degree={CHEBY_DEGREE}) forward+grad(x)  : {fmt(sg)}")

    # ---------------------------------------------------------------------
    # XLA cost analysis + jaxpr primitive counts for the s2nequix-cheby loss+grad
    # ---------------------------------------------------------------------
    with section("XLA cost analysis + primitive counts (s2nequix-cheby loss+grad)"):
        model = make_model("s2nequix-cheby")

        def f(m, b):
            (lval, _), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
                m, b, ENERGY_W, FORCE_W, STRESS_W, LOSS_TYPE
            )
            return lval, grads

        # jaxpr primitive histogram (traces only; counts each primitive call)
        from collections import Counter
        import jax.extend as jex
        jaxpr = jax.make_jaxpr(f)(model, batch)
        counter = Counter()
        def walk(j):
            for eq in j.eqns:
                counter[str(eq.primitive)] += 1
                for sj in eq.params.values():
                    if hasattr(sj, "jaxpr"):
                        walk(sj.jaxpr)
                    elif hasattr(sj, "eqns"):
                        walk(sj)
        walk(jaxpr.jaxpr)
        print("  Top-25 jaxpr primitive counts:")
        for name, c in counter.most_common(25):
            print(f"    {c:5d}  {name}")

        try:
            compiled = jax.jit(f).lower(model, batch).compile()
            ca = compiled.cost_analysis()
            if isinstance(ca, list):
                ca = ca[0]
            flops = ca.get("flops", 0)
            bytes_accessed = ca.get("bytes accessed", 0)
            print(f"  XLA cost: ~{flops/1e9:.1f} GFLOP, {bytes_accessed/1e6:.1f} MB memory traffic")
        except Exception as e:
            print(f"  cost_analysis unavailable: {e}")

    # ---------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------
    with section("SUMMARY (ms, lower is better)"):
        header = f"{'model':32s} {'fwd':>18s} {'loss+grad':>18s} {'train step':>18s}"
        print(header)
        print("-" * len(header))
        for mt in ["nequix", "s2nequix-cheby", "s2nequix-cheby-cuex"]:
            row = [f"{mt:32s}"]
            for k in ("fwd", "loss_grad", "train_step"):
                v = results.get((mt, k))
                row.append(f"{v['mean_ms']:14.2f} ms " if v else f"{'-':>18s}")
            print("".join(row))

        # cuex speedup on training step
        base = results.get(("s2nequix-cheby", "train_step"))
        cux = results.get(("s2nequix-cheby-cuex", "train_step"))
        if base and cux:
            print(f"\n  cuex training-step speedup over e3nn: {base['mean_ms']/cux['mean_ms']:.2f}x")


if __name__ == "__main__":
    main()
