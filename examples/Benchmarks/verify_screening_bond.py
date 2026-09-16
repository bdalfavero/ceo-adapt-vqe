"""
Verification: screening_mps_bond — does reducing the MPS bond dimension used
during gradient screening still lead to correct energy convergence?

This script runs TensorNetAdapt on H4/sto-3g at several screening bond
dimensions (the optimization always uses the full max_mps_bond).  It prints
per-iteration energies and the final error vs. FCI for each configuration, so
you can see which screening approximation level still achieves chemical accuracy.

Usage:
    cd /path/to/ceo-adapt-vqe
    .venv/bin/python examples/verify_screening_bond.py
"""

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
fci = mol.fci_energy
hf  = mol.hf_energy
print(f"H{N}  HF={hf:.8f}  FCI={fci:.8f}\n")

CHEM_ACCURACY = 1.6e-3   # Hartree

MAX_MPS_BOND = 32         # optimization bond dim, kept fixed for all runs
MAX_MPO_BOND = 200
N_ADAPT_ITER = 8

# screening_mps_bond=None means "use full MAX_MPS_BOND during screening" (exact baseline)
SCREENING_BONDS = [None, 16, 8, 4]


def run(screening_mps_bond, label):
    pool = GSD(mol)
    adapt = TensorNetAdapt(
        pool=pool,
        molecule=mol,
        max_adapt_iter=N_ADAPT_ITER + 1,
        recycle_hessian=False,
        tetris=False,
        verbose=False,
        threshold=1e-4,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=MAX_MPS_BOND,
        use_H_ket_screening=True,       # always on; screening_mps_bond is orthogonal
        screening_mps_bond=screening_mps_bond,
    )

    energies = []
    sel_ops  = []
    adapt.initialize()

    for i in range(N_ADAPT_ITER):
        finished = adapt.run_iteration()
        energies.append(adapt.energy)
        sel_ops.append(adapt.indices[-1] if adapt.indices else None)
        if finished:
            break

    return energies, sel_ops


# ── run all configurations ────────────────────────────────────────────────────
all_results = {}
for sb in SCREENING_BONDS:
    label = f"screening_bond={'full' if sb is None else sb:>4}"
    print(f"Running {label} ...")
    energies, sel_ops = run(sb, label)
    all_results[sb] = {"energies": energies, "sel_ops": sel_ops}
    print()

# ── energy table ─────────────────────────────────────────────────────────────
labels = {None: "full (baseline)"}
labels.update({sb: str(sb) for sb in SCREENING_BONDS if sb is not None})

col_w = 18
header = f"{'Iter':>4}" + "".join(f"  {labels[sb]:>{col_w}}" for sb in SCREENING_BONDS)
print("=" * len(header))
print("Energies (Hartree)")
print("=" * len(header))
print(header)
print("-" * len(header))

n_iters = max(len(r["energies"]) for r in all_results.values())
for i in range(n_iters):
    row = f"{i+1:>4}"
    for sb in SCREENING_BONDS:
        es = all_results[sb]["energies"]
        val = f"{es[i]:.8f}" if i < len(es) else "      —      "
        row += f"  {val:>{col_w}}"
    print(row)

print("-" * len(header))
row = f"{'FCI':>4}"
row += "".join(f"  {fci:>{col_w}.8f}" for _ in SCREENING_BONDS)
print(row)
print("=" * len(header))

# ── error and accuracy summary ────────────────────────────────────────────────
print()
print("Final error vs. FCI  |  Chem. accuracy = 1.6e-3 Ha")
print("-" * 55)
base_energies = all_results[None]["energies"]
for sb in SCREENING_BONDS:
    r = all_results[sb]
    final_e = r["energies"][-1]
    err = abs(final_e - fci)
    ok = "✓ chem. acc." if err < CHEM_ACCURACY else "✗ outside chem. acc."
    lbl = labels[sb]
    print(f"  {lbl:<20}  error={err:.2e} Ha  {ok}")

    if sb is not None:
        # Compare selected operators vs. baseline
        base_ops = base_energies
        this_ops = r["sel_ops"]
        base_seq = all_results[None]["sel_ops"]
        n = min(len(base_seq), len(this_ops))
        mismatches = sum(1 for a, b in zip(base_seq[:n], this_ops[:n]) if a != b)
        print(f"  {'':20}  op sequence mismatches vs. baseline: {mismatches}/{n}")

print()
print(f"Note: optimization always uses max_mps_bond={MAX_MPS_BOND}.")
print("      screening_mps_bond only affects which operators are selected.")
