"""
Benchmark: analytical recycling vs. parallel PSR vs. parallel FD gradient computation.

For a fixed ansatz of depth d, estimate_gradients costs:
  Recycling (serial):      O(d · n · χ³)     — one sweep, O(1) amortised state work/position
  Parallel PSR (p workers): O(d²/p · n · χ³) — d independent pairs of energy evals, parallelised
  Parallel FD  (p workers): O(d²/p · n · χ³) — same structure as PSR but forward differences

PSR is exact (like recycling) but parallelisable; FD has O(dx) gradient error.
Crossover with recycling occurs roughly when p ≳ 2d.

Usage:
    cd /path/to/ceo-adapt-vqe

    # Recommended: cap BLAS threads so Python threads get full cores
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python examples/benchmark_optim_gradients.py

    # Without BLAS tuning (may show oversubscription at high worker counts):
    .venv/bin/python examples/benchmark_optim_gradients.py
"""

import os
import time
import numpy as np
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt
from adaptvqe.pools import GSD

# ── config ────────────────────────────────────────────────────────────────────
N = 4
R = 1.5
geometry = [["H", [0, 0, i * R]] for i in range(N)]
MAX_MPS_BOND  = 16
MAX_MPO_BOND  = 200
WORKER_COUNTS = [1, 2, 4, 8]
ANSATZ_DEPTHS = [1, 2, 4, 6, 8, 10]
N_REPEATS     = 3   # median over this many repetitions

print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS', 'not set')}\n")

mol = MolecularData(geometry, "sto-3g", 1, 0, description=f"H{N}")
mol = run_pyscf(mol, run_fci=True, run_scf=True)
print(f"H{N}  HF={mol.hf_energy:.8f}  FCI={mol.fci_energy:.8f}\n")

# ── build reference ansatz ────────────────────────────────────────────────────
print(f"Building reference ansatz up to depth {max(ANSATZ_DEPTHS)} via ADAPT...")
pool_ref = GSD(mol)
runner = TensorNetAdapt(
    pool=pool_ref, molecule=mol,
    max_adapt_iter=max(ANSATZ_DEPTHS) + 1,
    recycle_hessian=False, tetris=False, verbose=False,
    threshold=1e-6, max_mpo_bond=MAX_MPO_BOND, max_mps_bond=MAX_MPS_BOND,
    n_optim_workers=1,
)
runner.initialize()
while len(runner.indices) < max(ANSATZ_DEPTHS):
    if runner.run_iteration():
        break

ansatz_coefficients = runner.coefficients
ansatz_indices      = runner.indices
print(f"Built ansatz of depth {len(ansatz_indices)}.\n")


# ── adapt instances (one per worker count) ────────────────────────────────────
def build_adapt(n_workers):
    pool = GSD(mol)
    a = TensorNetAdapt(
        pool=pool, molecule=mol,
        max_adapt_iter=1,
        recycle_hessian=False, tetris=False, verbose=False,
        threshold=1e-6, max_mpo_bond=MAX_MPO_BOND, max_mps_bond=MAX_MPS_BOND,
        n_optim_workers=n_workers,
    )
    a.initialize()
    return a

adapt_instances = {w: build_adapt(w) for w in WORKER_COUNTS}


def time_grads(adapt, coeffs, idxs, method, n_repeats):
    times, grads_last = [], None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        grads_last = adapt.estimate_gradients(
            coefficients=list(coeffs), indices=list(idxs), method=method
        )
        times.append(time.perf_counter() - t0)
    return np.median(times), grads_last


# ── timing runs ───────────────────────────────────────────────────────────────
# Configs: (label, n_workers, method)
configs = [
    ("recycling (serial)", 1, "an"),
    *[(f"PSR  ({w} workers)", w, "an") for w in WORKER_COUNTS[1:]],
    *[(f"FD   ({w} workers)", w, "fd") for w in WORKER_COUNTS[1:]],
]

results = {}   # (label) -> {depth -> (time, grads)}

print("Timing estimate_gradients at each depth...")
for label, n_workers, method in configs:
    adapt = adapt_instances[n_workers]
    results[label] = {}
    for depth in ANSATZ_DEPTHS:
        if depth > len(ansatz_indices):
            break
        t, g = time_grads(adapt, ansatz_coefficients[:depth], ansatz_indices[:depth],
                          method, N_REPEATS)
        results[label][depth] = (t, g)
        print(f"  [{label:<24}] depth={depth:>2}: {t:.4f}s")
    print()

# ── correctness ───────────────────────────────────────────────────────────────
print("=" * 65)
print("Correctness vs. serial recycling")
print("  PSR should match to ~machine precision (exact)")
print("  FD  should match to ~1e-5 (O(dx=1e-8) error)")
print("=" * 65)
for label, n_workers, method in configs[1:]:
    print(f"\n{label}:")
    for depth in list(ANSATZ_DEPTHS)[:4]:
        if depth not in results[label] or depth not in results["recycling (serial)"]:
            continue
        g_ref = results["recycling (serial)"][depth][1]
        g_cmp = results[label][depth][1]
        diffs = [abs(a - b) for a, b in zip(g_ref, g_cmp)]
        max_d = max(diffs) if diffs else 0.0
        mean_d = np.mean(diffs) if diffs else 0.0
        ok = "✓" if max_d < (1e-8 if method == "an" else 1e-3) else "✗"
        print(f"  depth={depth}: max|Δg|={max_d:.2e}  mean|Δg|={mean_d:.2e}  {ok}")

# ── speedup table ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Speedup vs. serial recycling  (PSR rows = exact; FD rows = approx)")
print("=" * 65)

col = 18
header = f"{'depth':>6}" + "".join(f"  {lbl[:col]:>{col}}" for lbl, _, _ in configs[1:])
print(header[:min(len(header), 120)])
print("-" * min(len(header), 120))

for depth in ANSATZ_DEPTHS:
    if depth > len(ansatz_indices):
        break
    t_ref = results["recycling (serial)"].get(depth, (None,))[0]
    if t_ref is None:
        continue
    row = f"{depth:>6}"
    for label, _, _ in configs[1:]:
        t = results[label].get(depth, (None,))[0]
        if t is None:
            row += f"  {'—':>{col}}"
        else:
            row += f"  {t_ref/t:>{col}.2f}x"
    print(row[:120])

print()
print(f"max_mps_bond={MAX_MPS_BOND}. Increase it for a more realistic crossover point.")
print("Tip: PSR >1x with fewer workers than FD means exactness comes for free.")
