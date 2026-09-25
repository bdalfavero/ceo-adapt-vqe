"""Compare the current method of doing the exponentials (simulate one exponential circuit at a time)
to simulating the circuit for the whole Ansatz."""

from time import perf_counter_ns
from random import sample
import numpy as np
import pandas as pd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from qiskit.qasm2 import dumps
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.pools import DVE_CEO, GSD, PairedGSD

MAX_MPO_BOND = 1000
MAX_DEPTH = 15
SAMPLES = 10

def alternate_compute_state(
    adapt: TensorNetAdapt, indices: list[int], coefficients: list[float], bra: bool=False,
    max_bond: int=100, ref_state=None, big_endian=False
) -> MatrixProductState:
    """Compute the state by getting the whole circuit at once."""
    
    if ref_state is None:
        ref_state = adapt.tn_ref_state

    if bra:
        coefficients = [-c for c in reversed(coefficients)]
        indices = reversed(indices)

    evolution_circuit = adapt.pool.get_circuit(indices, coefficients, big_endian=big_endian)
    qasm_str = dumps(evolution_circuit)
    circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
        qasm_str, psi0=adapt.tn_ref_state.copy(), max_bond=max_bond, progbar=False
    )
    # OpenQASM 2 has no global phase, so reapply it here
    state = circuit_mps.psi * np.exp(1j * evolution_circuit.global_phase)
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

    pool = GSD(mol)

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

    # Test the exponentiation methods: Get timing and infidelity.
    avg_times_tn = []
    std_times_tn = []
    avg_times_alt = []
    std_times_alt = []
    avg_infidelity = []
    std_infidelity = []
    depths = list(range(1, MAX_DEPTH))
    for depth in depths:
        times_tn = []
        times_alt = []
        infidelities = []
        for _ in range(SAMPLES):
            indices = sample(range(len(pool.operators)), depth)
            coeffs = np.random.rand(len(indices))

            # Use the default compute_state.
            start_time = perf_counter_ns()
            mps_tn = tn_adapt.compute_state(indices=indices, coefficients=coeffs)
            end_time = perf_counter_ns()
            elapsed = abs(end_time - start_time)
            times_tn.append(elapsed)

            # Use the alternate compute_state.
            start_time = perf_counter_ns()
            mps_alt = alternate_compute_state(tn_adapt, indices=indices, coefficients=coeffs)
            end_time = perf_counter_ns()
            elapsed = abs(end_time - start_time)
            times_alt.append(elapsed)

            # Get the infidelity.
            infidelity = 1. - np.abs(mps_alt.H @ mps_tn) ** 2
            infidelities.append(infidelity)
        avg_times_tn.append(np.average(times_tn))
        std_times_tn.append(np.std(times_tn))
        avg_times_alt.append(np.average(times_alt))
        std_times_alt.append(np.std(times_alt))
        avg_infidelity.append(np.average(infidelities))
        std_infidelity.append(np.std(infidelities))

    output_dict = {
        "depths": depths,
        "avg_times_tn": avg_times_tn,
        "std_times_tn": std_times_tn,
        "avg_times_alt": avg_times_alt,
        "std_times_alt": std_times_alt,
        "avg_infidelity": avg_infidelity,
        "std_infidelity": std_infidelity
    }
    df = pd.DataFrame.from_dict(output_dict, orient='columns')
    df.set_index("depths", inplace=True)
    df.to_csv("alternate_expm_results.csv")