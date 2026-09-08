"""
Benchmark: TensorNetAdapt gradient MPO caching vs. no caching.

The fix: eval_candidate_gradient now caches 2*H_mpo@A_mpo per pool operator
on first call and reuses it on subsequent iterations. This script quantifies
the speedup by timing rank_gradients (the gradient screening step) across
multiple ADAPT iterations.

Usage:
    cd /path/to/ceo-adapt-vqe
    python examples/benchmark_tn_caching.py
"""

import time
import numpy as np
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt
from adaptvqe.pools import GSD

# ── molecule ──────────────────────────────────────────────────────────────────
N = 4
R = 1.5
geometry = [["H", [0, 0, i * R]] for i in range(N)]
mol = MolecularData(geometry, "sto-3g", 1, 0, description=f"H{N}")
mol = run_pyscf(mol, run_fci=True, run_scf=True)
print(f"H{N} chain  HF={mol.hf_energy:.6f}  FCI={mol.fci_energy:.6f}\n")

MAX_MPO_BOND = 200
MAX_MPS_BOND = 16
N_ADAPT_ITER = 4


# ── "before" class: always recomputes the gradient MPO ───────────────────────
class TensorNetAdaptNoCaching(TensorNetAdapt):
    """Reproduces the original behaviour: recomputes 2*H@A every call."""

    def eval_candidate_gradient(self, index, coefficients=None, indices=None):
        operator = self.pool.get_mpo_op(index, len(self.hamiltonian_mpo.tensors))
        if len(operator.tensors) != len(self.hamiltonian_mpo.tensors):
            raise AssertionError(
                f"Observable has {len(operator.tensors)} tensors "
                f"and H has {len(self.hamiltonian_mpo.tensors)}"
            )
        observable = 2 * self.hamiltonian_mpo.apply(operator)
        return self.evaluate_observable(observable, coefficients, indices)


def build_adapt(cls):
    pool = GSD(mol)
    return cls(
        pool=pool,
        molecule=mol,
        max_adapt_iter=N_ADAPT_ITER + 1,
        recycle_hessian=False,
        tetris=False,
        verbose=False,
        threshold=1e-3,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=MAX_MPS_BOND,
    )


def run_with_timing(adapt, n_iter, label):
    """
    Patch rank_gradients to measure its wall time, run n_iter ADAPT iterations,
    return list of screening times.
    """
    screening_times = []
    original_rank = adapt.rank_gradients

    def timed_rank(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_rank(*args, **kwargs)
        screening_times.append(time.perf_counter() - t0)
        return result

    adapt.rank_gradients = timed_rank

    print(f"── {label} ──")
    adapt.initialize()
    for i in range(n_iter):
        finished = adapt.run_iteration()
        print(f"  iter {i+1}: screening={screening_times[-1]:.3f}s  energy={adapt.energy:.6f}")
        if finished:
            break
    print()
    return screening_times


no_cache = build_adapt(TensorNetAdaptNoCaching)
cached   = build_adapt(TensorNetAdapt)

times_no_cache = run_with_timing(no_cache, N_ADAPT_ITER, "NO CACHING (original)")
times_cached   = run_with_timing(cached,   N_ADAPT_ITER, "WITH CACHING (fixed)")

# ── report ────────────────────────────────────────────────────────────────────
print("=" * 62)
print("rank_gradients wall time (seconds)")
print("=" * 62)
header = f"{'Iter':>5}  {'No cache (s)':>12}  {'Cached (s)':>10}  {'Speedup':>8}"
print(header)
print("-" * 42)

n = min(len(times_no_cache), len(times_cached))
for i in range(n):
    nc, c = times_no_cache[i], times_cached[i]
    speedup = nc / c if c > 0 else float("inf")
    note = "  ← MPO build (both pay once)" if i == 0 else ""
    print(f"{i+1:>5}  {nc:>12.3f}  {c:>10.3f}  {speedup:>7.2f}x{note}")

if n > 1:
    mean_nc = np.mean(times_no_cache[1:])
    mean_c  = np.mean(times_cached[1:])
    print("-" * 42)
    print(f"{'mean':>5}  {mean_nc:>12.3f}  {mean_c:>10.3f}  {mean_nc/mean_c:>7.2f}x  (iter 2+, steady state)")

print()
print(f"Pool size: {no_cache.pool.size} operators")
print("No cache: recomputes all MPO products on every iteration")
print("Cached:   builds MPO products once (iter 1), reuses thereafter")
