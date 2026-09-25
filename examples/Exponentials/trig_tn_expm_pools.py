"""Compare the circuit-based tensor network exponentials (OperatorPool.tn_expm_mult_state) to the
trigonometric-formula ones e^(c A) = I + sin(c) A + (1 - cos(c)) A^2 (the pools' own tn_expm_mult_state)
for every pool type that implements the formula. The exact state from the sparse (LinAlg) exponentials
is used as a reference to get the infidelity of both tensor network methods."""

from time import perf_counter_ns
from copy import deepcopy
import numpy as np
import pandas as pd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from quimb.tensor.tensor_1d import MatrixProductState
from adaptvqe.chemistry import get_hf_det
from adaptvqe.matrix_tools import ket_to_sparse_vector
from adaptvqe.tensor_helpers import computational_basis_mps
from adaptvqe.pools import OperatorPool, ImplementationType, GSD, PairedGSD, QE, MVP_CEO, OVP_CEO, DVE_CEO

MAX_MPO_BOND = 1000
MAX_DEPTH = 20
SAMPLES = 10
# SD, QE1 and QE_All are left out: their circuit-based tensor network method is broken (see test_trig_tn_expm.py).
POOL_CLASSES = [GSD, PairedGSD, QE, MVP_CEO, OVP_CEO, DVE_CEO]


def circuit_compute_state(pool, indices, coefficients, ref_state, max_bond):
    state = ref_state.copy()
    for coefficient, index in zip(coefficients, indices):
        state = OperatorPool.tn_expm_mult_state(pool, coefficient, index, state, max_bond=max_bond, big_endian=False)
    return state


def trig_compute_state(pool, indices, coefficients, ref_state, max_bond):
    state = ref_state.copy()
    for coefficient, index in zip(coefficients, indices):
        state = pool.tn_expm_mult_state(coefficient, index, state, max_bond=max_bond, big_endian=False)
    return state


def exact_compute_state(pool, indices, coefficients, ref_state):
    state = ref_state.copy()
    for coefficient, index in zip(coefficients, indices):
        state = pool.expm_mult(coefficient, index, state)
    dense = np.asarray(state.todense() if hasattr(state, "todense") else state)
    return MatrixProductState.from_dense(dense.flatten())


def infidelity(mps, exact_mps):
    return 1. - abs(exact_mps.H @ mps) ** 2 / (mps.H @ mps).real


def benchmark_pool(pool_class, mol, chi, rng):
    pool = pool_class(mol)
    pool.max_mpo_bond = MAX_MPO_BOND
    sparse_pool = deepcopy(pool)
    sparse_pool.imp_type = ImplementationType.SPARSE
    tn_pool = deepcopy(pool)
    tn_pool.imp_type = ImplementationType.TENSORS

    ref_det = get_hf_det(mol.n_electrons, pool.n)
    sparse_ref_state = ket_to_sparse_vector(ref_det).astype(complex)
    tn_ref_state = computational_basis_mps(ref_det)

    # The MPOs are built lazily and cached, so build them before timing. This is a one-time cost per pool.
    start_time = perf_counter_ns()
    for index in range(tn_pool.size):
        tn_pool.get_mpo_op(index, tn_ref_state.L)
    mpo_build_time = perf_counter_ns() - start_time
    print(f"{pool_class.__name__}: {tn_pool.size} operators, MPOs built in {mpo_build_time / 1e6:.1f} ms")

    rows = []
    for depth in range(1, MAX_DEPTH):
        times_circuit = []
        times_trig = []
        infidelities_circuit = []
        infidelities_trig = []
        for _ in range(SAMPLES):
            indices = rng.choice(tn_pool.size, size=depth)
            coeffs = rng.random(depth)

            start_time = perf_counter_ns()
            mps_circuit = circuit_compute_state(tn_pool, indices, coeffs, tn_ref_state, chi)
            times_circuit.append(perf_counter_ns() - start_time)

            start_time = perf_counter_ns()
            mps_trig = trig_compute_state(tn_pool, indices, coeffs, tn_ref_state, chi)
            times_trig.append(perf_counter_ns() - start_time)

            exact_mps = exact_compute_state(sparse_pool, indices, coeffs, sparse_ref_state)
            infidelities_circuit.append(infidelity(mps_circuit, exact_mps))
            infidelities_trig.append(infidelity(mps_trig, exact_mps))
        rows.append({
            "pool": pool_class.__name__,
            "depths": depth,
            "avg_times_circuit": np.average(times_circuit),
            "std_times_circuit": np.std(times_circuit),
            "avg_times_trig": np.average(times_trig),
            "std_times_trig": np.std(times_trig),
            "speedup": np.average(times_circuit) / np.average(times_trig),
            "avg_infidelity_circuit": np.average(infidelities_circuit),
            "std_infidelity_circuit": np.std(infidelities_circuit),
            "avg_infidelity_trig": np.average(infidelities_trig),
            "std_infidelity_trig": np.std(infidelities_trig),
            "mpo_build_time": mpo_build_time,
        })
    return rows


if __name__ == "__main__":
    N = 4
    chi = 15

    r = 1.5
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
    mol = run_pyscf(mol, run_fci=False, run_ccsd=True, run_scf=True)

    rng = np.random.default_rng(0)
    rows = []
    for pool_class in POOL_CLASSES:
        rows.extend(benchmark_pool(pool_class, mol, chi, rng))

    df = pd.DataFrame(rows)
    df.set_index(["pool", "depths"], inplace=True)
    print(df[["speedup", "avg_infidelity_circuit", "avg_infidelity_trig"]].groupby("pool").agg(["mean", "max"]))
    df.to_csv("trig_tn_expm_pools_results.csv")
