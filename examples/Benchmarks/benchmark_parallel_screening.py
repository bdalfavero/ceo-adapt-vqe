"""
Benchmark and correctness check: parallel gradient screening.

Runs TensorNetAdapt with n_screening_workers in {1, 2, 4} on H4/sto-3g and:
  1. Verifies that all parallel configs produce identical gradients and energies
     to the serial baseline (they should be bit-for-bit identical — threading
     only changes evaluation order, not results).
  2. Times rank_gradients per iteration so you can see the wall-clock speedup.

BLAS thread note
----------------
quimb / numpy use multi-threaded BLAS internally.  When n_screening_workers > 1
Python threads and BLAS threads compete for the same cores, which can cause
slowdown (oversubscription).  For best parallel efficiency, cap BLAS threads
to 1 before launching:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python examples/benchmark_parallel_screening.py

Usage:
    cd /path/to/ceo-adapt-vqe
    .venv/bin/python examples/benchmark_parallel_screening.py
"""

import os
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
print(f"H{N}  HF={mol.hf_energy:.8f}  FCI={mol.fci_energy:.8f}")
print(f"BLAS threads (OMP_NUM_THREADS): {os.environ.get('OMP_NUM_THREADS', 'not set')}\n")

MAX_MPS_BOND = 32
MAX_MPO_BOND = 200
N_ADAPT_ITER = 5
WORKER_COUNTS = [1, 2, 4]
ATOL = 1e-10   # gradients and energies should be identical across worker counts


def build(n_workers):
    pool = GSD(mol)
    return TensorNetAdapt(
        pool=pool,
        molecule=mol,
        max_adapt_iter=N_ADAPT_ITER + 1,
        recycle_hessian=False,
        tetris=False,
        verbose=False,
        threshold=1e-4,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=MAX_MPS_BOND,
        use_H_ket_screening=True,
        n_screening_workers=n_workers,
    )


def run_timed(adapt, n_iter, label):
    """
    Run n_iter iterations, record (screening_time, total_norm, sel_indices, energy) per iter.
    """
    records = []
    orig_rank = adapt.rank_gradients

    def timed_rank(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_rank(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        sel_indices, sel_gradients, total_norm, max_norm = result
        records.append({
            "screening_time": elapsed,
            "total_norm": total_norm,
            "sel_indices": list(sel_indices),
            "sel_gradients": list(sel_gradients),
        })
        return result

    adapt.rank_gradients = timed_rank
    adapt.initialize()

    print(f"── workers={adapt.n_screening_workers} ──")
    for i in range(n_iter):
        finished = adapt.run_iteration()
        r = records[-1]
        print(
            f"  iter {i+1}: screen={r['screening_time']:.3f}s  "
            f"‖g‖={r['total_norm']:.6f}  E={adapt.energy:.8f}"
        )
        if finished:
            break
    print()
    return records, adapt.energy


all_records = {}
all_energies = {}

for n_workers in WORKER_COUNTS:
    adapt = build(n_workers)
    records, final_energy = run_timed(adapt, N_ADAPT_ITER, f"workers={n_workers}")
    all_records[n_workers] = records
    all_energies[n_workers] = final_energy

# ── correctness check ─────────────────────────────────────────────────────────
print("=" * 60)
print("Correctness: parallel vs. serial (workers=1)")
print("=" * 60)
baseline = all_records[1]
all_pass = True

for n_workers in WORKER_COUNTS[1:]:
    records = all_records[n_workers]
    n = min(len(baseline), len(records))
    print(f"\nworkers={n_workers} vs. serial:")
    for i in range(n):
        b, r = baseline[i], records[i]

        norm_diff  = abs(b["total_norm"] - r["total_norm"])
        ops_match  = b["sel_indices"] == r["sel_indices"]
        grads_diff = max(
            (abs(a - c) for a, c in zip(b["sel_gradients"], r["sel_gradients"])),
            default=0.0,
        )
        ok = norm_diff < ATOL and ops_match and grads_diff < ATOL
        if not ok:
            all_pass = False
        print(
            f"  iter {i+1}: [{'PASS' if ok else 'FAIL'}]  "
            f"‖g‖ diff={norm_diff:.1e}  "
            f"ops={'match' if ops_match else 'MISMATCH'}  "
            f"max grad diff={grads_diff:.1e}"
        )

    e_diff = abs(all_energies[1] - all_energies[n_workers])
    e_ok = e_diff < ATOL
    if not e_ok:
        all_pass = False
    print(
        f"  final energy: serial={all_energies[1]:.10f}  "
        f"parallel={all_energies[n_workers]:.10f}  "
        f"diff={e_diff:.1e}  [{'PASS' if e_ok else 'FAIL'}]"
    )

print()
print(f"Overall correctness: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

# ── timing table ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Screening wall time (seconds) per iteration")
print("=" * 60)
col = 12
header = f"{'Iter':>4}" + "".join(f"  {'workers='+str(w):>{col}}" for w in WORKER_COUNTS)
print(header)
print("-" * len(header))
n_iters = min(len(all_records[w]) for w in WORKER_COUNTS)
for i in range(n_iters):
    row = f"{i+1:>4}"
    base_t = all_records[1][i]["screening_time"]
    for w in WORKER_COUNTS:
        t = all_records[w][i]["screening_time"]
        speedup = f"({base_t/t:.1f}x)" if w > 1 else ""
        row += f"  {t:>7.3f} {speedup:<4}"
    print(row)

mean_base = np.mean([r["screening_time"] for r in all_records[1]])
print("-" * len(header))
row = f"{'mean':>4}"
for w in WORKER_COUNTS:
    mt = np.mean([r["screening_time"] for r in all_records[w]])
    speedup = f"({mean_base/mt:.1f}x)" if w > 1 else ""
    row += f"  {mt:>7.3f} {speedup:<4}"
print(row)
print()
print("Tip: if parallel is slower, try OMP_NUM_THREADS=1 to avoid BLAS oversubscription.")
