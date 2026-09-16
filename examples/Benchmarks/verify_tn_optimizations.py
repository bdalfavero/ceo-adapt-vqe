"""
Verification: TensorNetAdapt optimization correctness check.

Runs three configurations side-by-side on the same molecule and checks that
gradient arrays, selected operators, and converged energies agree to within
floating-point tolerance:

  A. baseline  — use_H_ket_screening=False, compress_grad_mpos=False
                 (plain caching of H@A MPOs, no extra approximations)
  B. H_ket     — use_H_ket_screening=True,  compress_grad_mpos=False
                 (precompute H|ψ⟩ once per sweep; should be numerically exact)
  C. compressed — use_H_ket_screening=False, compress_grad_mpos=True
                 (compress H@A MPOs to max_mpo_bond; introduces small approximation)

Usage:
    cd /path/to/ceo-adapt-vqe
    python examples/verify_tn_optimizations.py
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
print(f"H{N} chain  HF={mol.hf_energy:.6f}  FCI={mol.fci_energy:.6f}\n")

MAX_MPO_BOND = 200
MAX_MPS_BOND = 64   # high bond dim so results are near-exact for all three
N_ADAPT_ITER = 5
ATOL = 1e-8         # tolerance for gradient / energy comparisons


def build(label, use_H_ket_screening, compress_grad_mpos):
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
        use_H_ket_screening=use_H_ket_screening,
        compress_grad_mpos=compress_grad_mpos,
    )
    return adapt


# ── run and collect per-iteration diagnostics ─────────────────────────────────

def run_collect(adapt, n_iter, label):
    """Run n_iter ADAPT iterations; return list of (sel_indices, sel_gradients, total_norm, energy)."""
    records = []
    original_rank = adapt.rank_gradients

    def instrumented_rank(*args, **kwargs):
        sel_indices, sel_gradients, total_norm, max_norm = original_rank(*args, **kwargs)
        records.append({
            "sel_indices": list(sel_indices),
            "sel_gradients": list(sel_gradients),
            "total_norm": total_norm,
        })
        return sel_indices, sel_gradients, total_norm, max_norm

    adapt.rank_gradients = instrumented_rank
    adapt.initialize()
    print(f"── {label} ──")
    for i in range(n_iter):
        finished = adapt.run_iteration()
        rec = records[-1]
        print(
            f"  iter {i+1}: sel_op={rec['sel_indices'][:3]}...  "
            f"‖g‖={rec['total_norm']:.6f}  E={adapt.energy:.8f}"
        )
        if finished:
            break
    print()
    return records, adapt.energy


configs = [
    ("A: baseline (H@A cache, no compress)", False, False),
    ("B: H_ket screening",                   True,  False),
    ("C: H@A cache + MPO compress",          False, True),
]

results = {}
for label, use_H_ket, compress in configs:
    adapt = build(label, use_H_ket, compress)
    records, final_energy = run_collect(adapt, N_ADAPT_ITER, label)
    results[label] = {"records": records, "final_energy": final_energy}

# ── compare ───────────────────────────────────────────────────────────────────
print("=" * 62)
print("Comparison vs. baseline (A)")
print("=" * 62)

baseline_label = "A: baseline (H@A cache, no compress)"
baseline = results[baseline_label]

all_passed = True
for label, _, _ in configs[1:]:
    r = results[label]
    n = min(len(baseline["records"]), len(r["records"]))
    print(f"\n{label}")

    for i in range(n):
        b_rec = baseline["records"][i]
        r_rec = r["records"][i]

        # Selected operators should be identical
        ops_match = b_rec["sel_indices"] == r_rec["sel_indices"]

        # Gradient norms should agree to ATOL
        norm_diff = abs(b_rec["total_norm"] - r_rec["total_norm"])
        norm_ok = norm_diff < ATOL

        # Top gradient magnitude
        if b_rec["sel_gradients"] and r_rec["sel_gradients"]:
            grad_diffs = [abs(a - b) for a, b in
                          zip(b_rec["sel_gradients"], r_rec["sel_gradients"])]
            max_grad_diff = max(grad_diffs)
        else:
            max_grad_diff = 0.0
        grads_ok = max_grad_diff < ATOL

        status = "PASS" if (ops_match and norm_ok and grads_ok) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(
            f"  iter {i+1}: [{status}]  "
            f"ops={'match' if ops_match else 'MISMATCH'}  "
            f"‖g‖ diff={norm_diff:.2e}  "
            f"max grad diff={max_grad_diff:.2e}"
        )

    energy_diff = abs(baseline["final_energy"] - r["final_energy"])
    energy_ok = energy_diff < ATOL
    if not energy_ok:
        all_passed = False
    print(
        f"  final energy: baseline={baseline['final_energy']:.8f}  "
        f"this={r['final_energy']:.8f}  diff={energy_diff:.2e}  "
        f"[{'PASS' if energy_ok else 'FAIL'}]"
    )

print()
print("=" * 62)
print(f"Overall: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
print("=" * 62)
print()
print("Note: config C (MPO compress) may show small diffs — that is expected")
print("      since it introduces a deliberate approximation. Configs A and B")
print("      should agree to machine precision.")
