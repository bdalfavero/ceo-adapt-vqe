"""Compare the current method of doing the exponentials (simulate one exponential circuit at a time)
to applying each exponential with the identity e^(c A) = I + sin(c) A + (1 - cos(c)) A^2,
which holds for GSD generators (A^3 = -A). The generators are applied as MPOs, so no circuit is needed."""

from time import perf_counter_ns
from random import sample
import numpy as np
import pandas as pd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from quimb.tensor.tensor_1d import MatrixProductState
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.pools import GSD

MAX_MPO_BOND = 1000
MAX_DEPTH = 20
SAMPLES = 10


def trig_mpo_expm_mult_state(
    adapt: TensorNetAdapt, coefficient: float, index: int, state: MatrixProductState, max_bond: int | None = None
) -> MatrixProductState:
    """Multiply the state by e^(coefficient * A), where A is the pool operator labeled by index.
    Uses e^(c A) = I + sin(c) A + (1 - cos(c)) A^2, valid when A^3 = -A (e.g. GSD)."""

    op_mpo = adapt.pool.get_mpo_op(index, state.L)
    op_state = op_mpo.apply(state)
    op2_state = op_mpo.apply(op_state)

    # There is a weird thing in quimb where we can't multiply an MPS by 0.
    # If a prefactor is near 0, replace it by 1e-18.
    sin_coeff = np.sin(coefficient)
    if abs(sin_coeff) <= 1e-18:
        sin_coeff = 1e-18
    cos_coeff = 1. - np.cos(coefficient)
    if abs(cos_coeff) <= 1e-18:
        cos_coeff = 1e-18

    mult_state = state + sin_coeff * op_state + cos_coeff * op2_state
    if max_bond is not None:
        mult_state.compress(max_bond=max_bond)
    return mult_state


def trig_mpo_compute_state(
    adapt: TensorNetAdapt, indices: list[int], coefficients: list[float], bra: bool=False,
    max_bond: int | None = None, ref_state=None
) -> MatrixProductState:
    """Compute the state by applying the exponential of each operator using the trigonometric identity."""

    if ref_state is None:
        ref_state = adapt.tn_ref_state
    if max_bond is None:
        max_bond = adapt.max_mps_bond

    if bra:
        coefficients = [-c for c in reversed(coefficients)]
        indices = reversed(indices)

    state = ref_state.copy()
    for coefficient, index in zip(coefficients, indices):
        state = trig_mpo_expm_mult_state(adapt, coefficient, index, state, max_bond=max_bond)
    if bra:
        state = state.H

    return state


if __name__ == "__main__":
    N = 4
    chi = 15
    num_iter = 10

    r = 1.5
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
    mol = run_pyscf(mol, run_fci=False, run_ccsd=True, run_scf=True)  # CCSD doesn't work here?

    # LinAlgAdapt and TensorNetAdapt set different implementation types on the pool, so each gets its own.
    pool = GSD(mol)
    la_pool = GSD(mol)

    la_adapt = LinAlgAdapt(
        pool=la_pool,
        molecule=mol,
        max_adapt_iter=num_iter + 1,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.1,
        skip_converged_rename=True,
    )
    la_adapt.initialize()

    tn_adapt = TensorNetAdapt(
        pool=pool,
        molecule=mol,
        max_adapt_iter=num_iter + 1,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.1,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=chi,
        skip_converged_rename=True,
    )
    tn_adapt.initialize()

    # Warm up the lazily-built operators (sparse matrices and MPOs), so they aren't included in the timing.
    all_indices = list(range(len(pool.operators)))
    all_coeffs = np.zeros(len(all_indices))
    la_adapt.compute_state(indices=all_indices, coefficients=all_coeffs)
    tn_adapt.compute_state(indices=all_indices, coefficients=all_coeffs)
    trig_mpo_compute_state(tn_adapt, indices=all_indices, coefficients=all_coeffs)

    # Test the exponentiation methods: Get timing and infidelity.
    avg_times_la = []
    std_times_la = []
    avg_times_tn = []
    std_times_tn = []
    avg_times_trig = []
    std_times_trig = []
    avg_infidelity = []
    std_infidelity = []
    depths = list(range(1, MAX_DEPTH))
    for depth in depths:
        times_la = []
        times_tn = []
        times_trig = []
        infidelities = []
        for _ in range(SAMPLES):
            indices = sample(range(len(pool.operators)), depth)
            coeffs = np.random.rand(len(indices))

            # Use the LinAlg compute_state as a reference.
            start_time = perf_counter_ns()
            _ = la_adapt.compute_state(indices=indices, coefficients=coeffs)
            end_time = perf_counter_ns()
            times_la.append(abs(end_time - start_time))

            # Use the default TensorNet compute_state.
            start_time = perf_counter_ns()
            mps_tn = tn_adapt.compute_state(indices=indices, coefficients=coeffs)
            end_time = perf_counter_ns()
            times_tn.append(abs(end_time - start_time))

            # Use the trigonometric identity compute_state.
            start_time = perf_counter_ns()
            mps_trig = trig_mpo_compute_state(tn_adapt, indices=indices, coefficients=coeffs)
            end_time = perf_counter_ns()
            times_trig.append(abs(end_time - start_time))

            # Get the infidelity.
            infidelity = 1. - np.abs(mps_trig.H @ mps_tn) ** 2
            infidelities.append(infidelity)
        avg_times_la.append(np.average(times_la))
        std_times_la.append(np.std(times_la))
        avg_times_tn.append(np.average(times_tn))
        std_times_tn.append(np.std(times_tn))
        avg_times_trig.append(np.average(times_trig))
        std_times_trig.append(np.std(times_trig))
        avg_infidelity.append(np.average(infidelities))
        std_infidelity.append(np.std(infidelities))

    output_dict = {
        "depths": depths,
        "avg_times_la": avg_times_la,
        "std_times_la": std_times_la,
        "avg_times_tn": avg_times_tn,
        "std_times_tn": std_times_tn,
        "avg_times_trig": avg_times_trig,
        "std_times_trig": std_times_trig,
        "avg_infidelity": avg_infidelity,
        "std_infidelity": std_infidelity
    }
    df = pd.DataFrame.from_dict(output_dict, orient='columns')
    df.set_index("depths", inplace=True)
    df.to_csv("trig_mpo_expm_results.csv")
