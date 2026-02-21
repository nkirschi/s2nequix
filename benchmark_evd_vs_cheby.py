import jax
import jax.numpy as jnp
from jax import lax, jit
import time

# --- 1. STABLE COMPETITORS ---


@jit
def run_full_eigh(L, x, coeffs):
    """Exact spectral filtering via eigendecomposition."""
    vals, vecs = jnp.linalg.eigh(L)
    # Chebyshev polynomials are defined on [-1, 1].
    # Since eigenvalues of L_sym are in [0, 2], we shift: vals - 1.0
    shifted_vals = vals - 1.0

    # Evaluate polynomial for each eigenvalue
    # T_k(x) = cos(k * acos(x))
    f_vals = jnp.zeros_like(vals)
    for k, c in enumerate(coeffs):
        # Using JAX-native polynomial evaluation
        f_vals += c * jnp.cos(k * jnp.arccos(jnp.clip(shifted_vals, -1.0, 1.0)))

    return vecs @ (f_vals[:, None] * (vecs.T @ x))


@jit
def run_chebyshev(norm_w, senders, receivers, x, coeffs):
    """Sparse iterative filtering (Stable-ChebNet style)."""
    num_nodes = x.shape[0]

    def l_tilde_op(v):
        # L_tilde * v = (L_sym - I) * v
        # L_sym * v = v - (D^-1/2 W D^-1/2) * v
        # Therefore: (L_sym - I) * v = - (D^-1/2 W D^-1/2) * v
        return -jax.ops.segment_sum(norm_w[:, None] * v[senders], receivers, num_nodes)

    def body_fn(carry, c_k):
        t_curr, t_prev, total = carry
        # Standard Clenshaw/Chebyshev recurrence: T_k = 2 * x * T_{k-1} - T_{k-2}
        # Here x is the operator L_tilde
        t_next = 2 * l_tilde_op(t_curr) - t_prev
        return (t_next, t_curr, total + c_k * t_next), None

    t0 = x
    t1 = l_tilde_op(x)
    init_total = coeffs[0] * t0 + coeffs[1] * t1

    # Scan handles the recurrence from k=2 to K
    final_state, _ = lax.scan(body_fn, (t1, t0, init_total), coeffs[2:])
    return final_state[2]


# --- 2. BENCHMARK UTILITY ---


def get_error(y_true, y_pred):
    """Relative Frobenius norm error."""
    return jnp.linalg.norm(y_true - y_pred) / jnp.linalg.norm(y_true)


def benchmark_suite(N_list, K_list, density=0.05):
    print(
        f"{'N':>6} | {'K':>4} | {'Eigh(s)':>10} | {'Cheb(s)':>10} | {'Speedup':>8} | {'Rel. Error':>12}"
    )
    print("-" * 68)

    for N in N_list:
        for K in K_list:
            # Generate Data
            key = jax.random.PRNGKey(int(time.time()))
            x = jax.random.normal(key, (N, 32))
            coeffs = jnp.array([1.0] + [0.1] * K)  # Random-ish filter

            # Create sparse L_sym
            adj = (jax.random.uniform(key, (N, N)) < density).astype(jnp.float32)
            adj = (adj + adj.T) / 2  # Symmetric
            deg = adj.sum(axis=1) + 1e-6
            L = jnp.eye(N) - (adj / jnp.sqrt(deg[:, None] * deg[None, :]))

            # Sparse conversion
            senders, receivers = jnp.where(adj > 0)
            norm_w = adj[senders, receivers] / jnp.sqrt(deg[senders] * deg[receivers])

            # --- TIMING ---
            # Warmup and Compile
            _ = run_full_eigh(L, x, coeffs).block_until_ready()
            _ = run_chebyshev(norm_w, senders, receivers, x, coeffs).block_until_ready()

            # Eigh Time
            s = time.perf_counter()
            for _ in range(5):
                _ = run_full_eigh(L, x, coeffs).block_until_ready()
            t_eigh = (time.perf_counter() - s) / 5

            # Cheb Time
            s = time.perf_counter()
            for _ in range(5):
                _ = run_chebyshev(norm_w, senders, receivers, x, coeffs).block_until_ready()
            t_cheb = (time.perf_counter() - s) / 5

            # Accuracy
            y_eigh = run_full_eigh(L, x, coeffs)
            y_cheb = run_chebyshev(norm_w, senders, receivers, x, coeffs)
            err = get_error(y_eigh, y_cheb)

            print(
                f"{N:6d} | {K:4d} | {t_eigh:10.5f} | {t_cheb:10.5f} | {t_eigh / t_cheb:7.1f}x | {err:.2e}"
            )


print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Run for N up to 4096 (OOM limit for many GPUs on eigh)
benchmark_suite(N_list=[10, 100, 1000], K_list=[5, 10, 20, 50])
