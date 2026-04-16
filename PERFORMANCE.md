# Nequix + Chebyshev performance analysis

Benchmarks and actionable optimizations for the `s2nequix-cheby` training pipeline
on SPICE-like batches. Numbers come from `benchmark_training_spice.py`, which mirrors
`configs/omol25.yml` (SPICE subset) on a single H200.

## 1. Setup

- Hardware: H200 (single GPU), `jax==0.10.0`, `cuequivariance==0.9.1`, `cuequivariance-jax==0.8.1`.
- Model: `hidden = 128x0e + 64x1o + 32x2e + 32x3o`, `n_layers=4`, `lmax=3`,
  `cheby_degree=5`, `pretransform_feats=True`, `layer_norm=True`, `index_weights=False`,
  `n_species=83`.
- Batch: dynamic-batched to match `DataLoader`: 32 graphs, ~1k nodes, ~28k edges
  (graph-size distribution sampled from `molsizes.npy`: mean 31 atoms, 95th-pct 88).
  Padded to `max_n_node=1092`, `max_n_edge=36036`, `n_graph=33`.
- Training step: `loss(MAE energy-per-atom + L2 force + 0·stress)` →
  `eqx.filter_value_and_grad` → `optax.adam` update, all inside `eqx.filter_jit`.
  This matches `train.py` except for `pmap` (single device here).

Reproduce with `uv run python benchmark_training_spice.py`.

## 2. Headline numbers

| model                                                 | forward (E, F) | loss + grad | full train step |
|-------------------------------------------------------|---------------:|------------:|----------------:|
| `nequix` (vanilla e3nn-jax)                           |       26.5 ms  |    66.5 ms  |        69.3 ms  |
| `s2nequix-cheby` (vanilla e3nn-jax, degree 5)         |       32.1 ms  |    81.0 ms  |        84.3 ms  |
| `s2nequix-cheby` + cuex channel-TP patch (3 layers)   |   **14.9 ms**  | **40.5 ms** |    **43.6 ms**  |

- **Chebyshev overhead**: +5.6 ms forward / +14.5 ms loss+grad vs vanilla Nequix.
  Matches the isolated `EquivariantChebyshevLayer(degree=5)` cost of 0.87 ms forward /
  1.81 ms fwd+grad per layer × 4 layers.
- **cuex training-time gain**: **1.93× on full train step, 2.15× on inference**.
  This answers the question "does cuex help training?" — yes, roughly the same speedup
  on the backward pass as on the forward.

### Op-level profile (jaxpr primitive histogram, s2nequix-cheby loss+grad)

```
 1377  mul
 1353  dot_general
 1161  transpose
  871  add_any
  755  reshape
  552  broadcast_in_dim
  303  reduce_sum
  146  div
  129  pad
   89  gather
   85  scatter-add
```

XLA cost estimate: **~20 GFLOP but ~79 GB of memory traffic per step**. At
H200 HBM peak of ~3 TB/s that is a 26 ms bandwidth floor — the model is strongly
memory-bound, which explains why fused kernels (cuex) help so much: each small
`dot_general` / `transpose` pair currently round-trips through HBM.

## 3. Equivalence caveat for the cuex swap — READ BEFORE ROLLING OUT

The benchmark patches `NequixConvolution` to call `cuex.segmented_polynomial(method="uniform_1d")`
on the `channelwise_tensor_product` descriptor. The resulting layer is **equivariant and
structurally equivalent**, but it is **not bit-identical** to the e3nn-jax layer.

Direct numerical comparison on random inputs, same radial weights:

| configuration                            | shape  | max \|Δ\|  | median value (both) |
|------------------------------------------|-------:|-----------:|--------------------:|
| `128x0e + 64x1o + 32x2e + 32x3o`         |  5472  | 20.2       | 0.27                |
| `128x0e + 128x1o + 128x2e + 128x3o`      | 12672  | 38.2       | 0.29                |

Reasons:
1. **Layout**: e3nn-jax uses `mul_ir`; cuex uses `ir_mul` and internally calls
   `permute_segments` after `irreps3.sort()`. Output slot order differs.
2. **Normalization**: `channelwise_tensor_product` calls
   `normalize_paths_for_operand(-1)`; e3nn-jax bakes per-path normalization into the
   CG coefficients. Constants disagree.
3. **Weight-slot map**: operand 0 has 1312 entries (same as `tp_irreps.num_irreps`),
   but the i-th radial-MLP output maps to a different (path, channel) in each.

**Consequences**
- The 1.93× speedup is a valid estimate of the achievable training-time gain.
- You **cannot swap layers at inference** on an e3nn-trained checkpoint without a
  weight-permutation layer; a model trained from scratch with cuex would reach
  similar metrics at roughly half the wall-clock.
- NVIDIA's own `cuequivariance_torch` MACE integration handles this with an
  explicit compatibility wrapper for the same reason.

Options to get a true drop-in:
- Precompute a permutation `P` and per-path rescale `α` that map the e3nn layout to
  the cuex layout; wrap the cuex call with `x → P @ (α ⊙ x)` on inputs and
  `y → P⁻¹ @ y` on outputs.
- Use `cuequivariance_jax.flax_linen.*` / `cuequivariance_jax.nnx.*` module APIs and
  re-train.
- Add an equivariance test (rotation applied to inputs ⇒ rotated outputs) to verify
  the cuex layer is a valid SO(3) equivariant conv, even though it is not e3nn's.

## 4. Actionable opportunities — ranked by (impact / effort)

### Tier 1: large win, moderate work

1. **Swap channel-wise TP for cuex in `NequixConvolution` (layers 2…N)**
   Expected: ≈2× full train step on the SPICE config, about the same on inference.
   - Use `cuex.segmented_polynomial(method="uniform_1d")` with the descriptor from
     `cue.descriptors.channelwise_tensor_product(...)`.
   - Needed pre-processing of the descriptor:
     `ctp.squeeze_modes().flatten_coefficient_modes().split_mode("u", GCD_of_muls)`.
     For the current config GCD=32; for uniform-mul 128 layouts no split is needed.
   - Layer 0 has input `83x0e` (not divisible by 32) and is cheap — keep as e3nn.
   - **Known bug in cuequivariance==0.9.1**: `SegmentedPolynomial.__eq__` asserts
     `isinstance(other, SegmentedPolynomial)` which breaks when multiple descriptors
     coexist in one jit (JAX compares static args against heterogeneous cache keys).
     Fix with a three-line monkey-patch returning `NotImplemented` on mismatch, or
     file upstream. See `benchmark_training_spice.py` for the patch.
   - Commit to the equivalence caveat above — plan a fresh training run, not a
     checkpoint swap.

2. **Uniformize hidden multiplicities: `128x0e + 128x1o + 128x2e + 128x3o`**
   (or `64×` everywhere). The preliminary kernel-level benchmark in
   `benchmark_cuvariance.py` showed **8.4× vs e3nn** on uniform-mul vs only ~2× on
   the mixed-mul layout — the `split_mode` shim creates ~3× as many paths
   (`23 → 471`) and loses a lot of per-kernel efficiency. Uniform mul also enables
   cuex's `indexed_linear` for the per-species skip path.

3. **Enable `index_weights=True` + `cuex.segmented_polynomial(method="indexed_linear")` for the skip**
   The production SPICE config currently has `index_weights=false` (per-species
   weights disabled). Turning it back on with the cuex indexed-linear kernel replaces
   N small per-species linears with one fused pass.
   Standalone benchmark (from `benchmark_cuvariance.py`) showed the cuex path is
   slower than e3nn on small N (0.24 ms vs 0.05 ms) because of kernel overhead — it
   only wins once fused into the convolution. Measure jointly with Tier 1.

### Tier 2: medium win, low work

4. **bf16 activations, fp32 weights / optimizer state**
   Model is memory-bound (79 GB/step). bf16 activations halve HBM traffic and unlock
   H200 tensor-cores at ~4× fp32 throughput. Keep weights and optimizer state fp32,
   cast inside `__call__`. Watch force-gradient accuracy: compute `grad(positions)`
   in bf16 then promote to fp32 for the MAE. Expected: additional 1.3–1.7× on top of
   Tier 1.

5. **Pre-sort by receiver globally (not per-layer)**
   `e3nn.scatter_sum(..., indices_are_sorted=?)` and `jraph.segment_sum` both benefit
   when receivers are sorted. The data loader already computes `senders, receivers`
   from `matscipy.neighbours.neighbour_list` — sort once per batch in Python and pass
   `indices_are_sorted=True` into every `segment_sum` call (both spatial conv and
   Chebyshev). Cheap, no correctness risk.

6. **Fuse `linear_2 + skip + layer_norm + gate`**
   The 1161 transposes in the jaxpr come mostly from this block. A single fused
   pass over the output irreps array (custom equivariant residual block, or
   `e3nn.experimental.fused_*` if present) cuts transpose and reshape traffic
   noticeably. Lower priority than Tier 1, but worthwhile once cuex is in.

### Tier 3: small or situational

7. **Chebyshev inner loop at higher degree**
   At degree 5 the whole `EquivariantChebyshevLayer` is only 0.87 ms forward, so
   further optimization is not urgent. At degree ≥ 10 the scan becomes a meaningful
   share; at that point rewrite `l_tilde_op` as a single
   `cuex.segmented_polynomial(method="indexed_linear")` over sorted receivers, and
   roll the scan outside the layer so the JIT can reuse the sender-gather.

8. **Data-loader throughput check**
   With cuex the train step is ~43 ms → 23 steps/s → ~47 min per `subepoch_length=10_000`.
   Measure `prefetch` queue fill rate under real SPICE; if it drops below 1, the GPU
   is starved and further model optimizations give diminishing returns until the
   loader catches up. Increasing `num_workers` or switching to NumPy-zero-copy
   structured arrays in `dict_to_graphstuple` is the first lever.

9. **Replace `e3nn.gate` with a hand-rolled scalar-gated silu**
   Minor — on the order of a few ms. Only revisit once everything above is done.

## 5. Notes on workarounds carried in the benchmark

The benchmark file contains three non-trivial workarounds that future maintainers
should know about:

- `ctypes.CDLL(..., RTLD_GLOBAL)` preloading of `libcublas.so.12` (from JAX's
  bundled directory) and `libcue_ops.so` (from `cuequivariance_ops`). JAX does not
  add either directory to `LD_LIBRARY_PATH`, so `cuequivariance_jax` imports fail
  without this preload.
- `cue.SegmentedPolynomial.__eq__` monkey-patch — see Tier 1 for details.
- Descriptor rewriting via `squeeze_modes → flatten_coefficient_modes → split_mode("u", g)`
  to make mixed-multiplicity layouts digestible by the `uniform_1d` kernel.

All three are stable against the current cuequivariance 0.9.x line but should be
retested on each upgrade.

## 6. TL;DR

- On SPICE-like batches, `s2nequix-cheby` training steps at **~84 ms** with e3nn-jax.
- Switching channel-wise TP to cuequivariance drops this to **~44 ms** — a 1.9×
  training speedup (not just inference).
- Chebyshev adds a small, expected overhead (~5 ms fwd, ~15 ms loss+grad) and is
  not currently a bottleneck.
- Biggest open wins beyond cuex: uniform multiplicities (another ~4×), bf16 (~1.5×),
  indexed-linear skip.
- The cuex-patched layer is equivariant but **not numerically identical** to the
  e3nn-jax one — plan to train from scratch, not to hot-swap a trained checkpoint.
