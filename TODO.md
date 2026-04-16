# Performance TODO

## cuEquivariance Integration

cuEquivariance (NVIDIA) provides CUDA-accelerated kernels for equivariant operations.
Benchmark data from NVIDIA docs and independent studies (2025):

### MACE Port (`nequix/mace.py`)

The MACE architecture concentrates compute in operations where cuEquivariance excels.
Expected end-to-end speedup: **~3x** (up to 10x for large systems).

| Operation | e3nn | cuEquivariance | Speedup | Notes |
|---|---|---|---|---|
| Symmetric contraction (128ch, corr=3, 1000 nodes) | 40.6 ms | 3.54 ms | 11.5x | Dominant cost in MACE; biggest win |
| Channelwise tensor product (message passing) | 0.98 ms | 0.17 ms | 5.6x | Per-layer convolution |
| Indexed linear (species-dependent skip) | 9.01 ms | 1.02 ms | 8.8x | Requires "uniform_1d" kernel |
| Fully connected tensor product (small irreps) | ~1.3 ms | ~1.3 ms | 1.0x | No benefit |

**Priority: high** -- symmetric contraction is the single largest cost contributor in MACE.
The `SymmetricContraction` class should be designed as a swappable module so
cuEquivariance can be dropped in without changing the rest of the model.

### Nequix (`nequix/nequix.py`)

Nequix lacks symmetric contraction, so gains are more modest.
Expected end-to-end speedup: **~1.5-2x**.

- Channelwise TP in `NequixConvolution`: ~2-5x speedup possible
- Indexed linear skip connection: ~5-9x on that op, but small fraction of total
- Spectral layers (EVD/Chebyshev): no cuEquivariance equivalent, pure JAX
- Gated activations: no benefit

### Caveats

- cuEquivariance requires NVIDIA GPU with CUDA 12+
- Naive kernel fallback can be *slower* than e3nn (0.87x) for non-uniform segments
- Spherical harmonics in cuEquivariance lack CUDA kernels (163x slower than e3nn)
- JAX bindings (`cuequivariance-jax`) require JAX 0.5.0+

## Other Performance Opportunities

### Mixed Precision

- BF16/FP16 for linear layers and tensor products yields ~4x additional speedup on
  top of cuEquivariance (per arXiv:2510.23621)
- Energies and thermodynamic observables remain stable in NVT/NPT MD
- Practical policy: FP32 by default, BF16 for linear layers when accuracy permits

### JAX-Specific

- e3nn-jax claims 44% faster training than PyTorch e3nn for MACE (XLA compilation)
- Buffer donation in `jax.jit` for reduced memory copies
- `jax.lax.scan` already used in Chebyshev recursion (good)
- Profile `segment_sum` in EVD spectral path for potential custom kernel
