"""
Comparison: TiledPauliPool (from N=4 source ops) vs FullPauliPool on XXZ N=8 chain.

Motivation: large-system runs (N=40..100) with TiledPauliPool stall at absolute
energy error ~1.0.  This script checks whether the tiled pool is the bottleneck
by running both pools on a small (N=8) instance where the exact ground energy is
known and LinAlgAdapt is tractable.

If FullPauliPool converges lower, the tiled pool is missing important operators.
If both converge to the same energy, the stalling must come from somewhere else
(bond-dimension truncation, optimizer, etc.).

Usage:
    cd /path/to/ceo-adapt-vqe
    .venv/bin/python examples/Scaling/xxz_pool_comparison.py
"""

import numpy as np
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.hamiltonians import XXZHamiltonian

# ── config ─────────────────────────────────────────────────────────────────────
N_SOURCE         = 4     # small system used to learn source operators
N_TARGET         = 8     # system we actually want to solve
J_XY, J_Z       = 1, 1
NUM_ITER         = 100   # ADAPT iterations for the main comparison
N_RANDOM_RUNS    = 50    # random restarts for source-op collection (N_SOURCE)
MAX_ADAPT_SMALL  = 8     # max ADAPT depth per small-system run
MAX_MPO_BOND     = 100

print(f"XXZ chain: J_xy={J_XY}  J_z={J_Z}")

# ── exact energy for N_TARGET ───────────────────────────────────────────────────
# XXZHamiltonian caches exact energies for N in {3,4,6,8,...}; this is ED.
h_target = XXZHamiltonian(J_XY, J_Z, N_TARGET)
exact_energy = h_target.ground_energy
print(f"Exact ground energy (N={N_TARGET}): {exact_energy:.10f}\n")

# ── Step 1: collect source operators from N_SOURCE via random ADAPT restarts ───
print(f"Collecting source operators from N={N_SOURCE} via {N_RANDOM_RUNS} random restarts...")
h_source = XXZHamiltonian(J_XY, J_Z, N_SOURCE)
pool_source = FullPauliPool(n=N_SOURCE, max_mpo_bond=MAX_MPO_BOND)

source_indices = []
for run in range(N_RANDOM_RUNS):
    adapt_small = LinAlgAdapt(
        pool=pool_source,
        custom_hamiltonian=h_source,
        verbose=False,
        threshold=1e-5,
        max_adapt_iter=MAX_ADAPT_SMALL,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
    )
    adapt_small.run()
    for idx in adapt_small.indices:
        if idx not in source_indices:
            source_indices.append(idx)
    if (run + 1) % 10 == 0:
        print(f"  run {run+1}/{N_RANDOM_RUNS}: {len(source_indices)} unique ops so far")

source_ops = [pool_source.operators[i].operator for i in source_indices]
print(f"Source pool: {len(source_ops)} unique operators from N={N_SOURCE} "
      f"(pool size = {pool_source.size})\n")

# ── Step 2: build both pools for N_TARGET ──────────────────────────────────────
print(f"Building pools for N={N_TARGET}...")
tiled_pool = TiledPauliPool(n=N_TARGET, source_ops=source_ops, max_mpo_bond=MAX_MPO_BOND)
full_pool  = FullPauliPool(n=N_TARGET, max_mpo_bond=MAX_MPO_BOND)
print(f"  TiledPauliPool size: {tiled_pool.size}")
print(f"  FullPauliPool  size: {full_pool.size}  (4^{N_TARGET} = {4**N_TARGET})\n")

# ── Step 3: run ADAPT-VQE with each pool ───────────────────────────────────────
def run_adapt(pool, label, num_iter):
    adapt = LinAlgAdapt(
        pool=pool,
        custom_hamiltonian=h_target,
        verbose=False,
        threshold=1e-6,
        max_adapt_iter=num_iter + 1,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
    )
    adapt.initialize()

    energies  = [adapt.energy]
    abs_errs  = [abs(adapt.energy - exact_energy)]
    depths    = [0]

    print(f"── {label} ──")
    print(f"  {'iter':>4}  {'energy':>16}  {'|E-E_exact|':>14}  {'depth':>6}")
    print(f"  {'0':>4}  {adapt.energy:>16.10f}  {abs_errs[0]:>14.6e}  {0:>6}")

    for i in range(1, num_iter + 1):
        finished = adapt.run_iteration()
        e   = adapt.energy
        err = abs(e - exact_energy)
        d   = len(adapt.indices)
        energies.append(e)
        abs_errs.append(err)
        depths.append(d)
        print(f"  {i:>4}  {e:>16.10f}  {err:>14.6e}  {d:>6}")
        if finished:
            print(f"  [Converged at iter {i}]")
            break

    print()
    return energies, abs_errs, depths


energies_tiled, errs_tiled, depths_tiled = run_adapt(tiled_pool, f"TiledPauliPool (N={N_SOURCE}→{N_TARGET})", NUM_ITER)
energies_full,  errs_full,  depths_full  = run_adapt(full_pool,  f"FullPauliPool  (N={N_TARGET})",            NUM_ITER)

# ── Step 4: summary ────────────────────────────────────────────────────────────
print("=" * 65)
print(f"Summary  (N={N_TARGET}, J_xy={J_XY}, J_z={J_Z})")
print(f"  Exact energy:               {exact_energy:.10f}")
print()
print(f"  TiledPauliPool ({len(source_ops)} source ops, {tiled_pool.size} tiled):")
print(f"    Final energy:             {energies_tiled[-1]:.10f}")
print(f"    Final |E - E_exact|:      {errs_tiled[-1]:.6e}")
print(f"    Final ansatz depth:       {depths_tiled[-1]}")
print()
print(f"  FullPauliPool ({full_pool.size} ops):")
print(f"    Final energy:             {energies_full[-1]:.10f}")
print(f"    Final |E - E_exact|:      {errs_full[-1]:.6e}")
print(f"    Final ansatz depth:       {depths_full[-1]}")
print()

gap = energies_tiled[-1] - energies_full[-1]
if abs(gap) < 1e-6:
    print("Verdict: Both pools converge to the same energy — tiling is NOT the bottleneck.")
elif gap > 0:
    print(f"Verdict: TiledPauliPool is {gap:.6e} Ha ABOVE FullPauliPool.")
    print("         The tiled pool lacks operators that FullPauliPool finds.")
    print("         Pool expressibility is likely the bottleneck in large-system runs.")
else:
    print(f"Verdict: TiledPauliPool is {abs(gap):.6e} Ha BELOW FullPauliPool (unexpected).")
print("=" * 65)
