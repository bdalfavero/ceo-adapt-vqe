"""Do DMRG at multiple bond dimensions and get data about the circuit."""

import numpy as np
import pandas as pd
import quimb.tensor as qtn
from quimb.tensor import DMRG
import qiskit
from qiskit.qasm2 import dumps
from qiskit_ibm_runtime.fake_provider import FakeFez
from mps_to_circuit import mps_to_circuit
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.tensor_helpers import qubop_to_mpo

MAX_MPO_BOND = 1000


def circuit_energy(mpo, circuit, chi):
    ckt_qasm = dumps(circuit)
    circuit = qtn.CircuitMPS.from_openqasm2_str(ckt_qasm, max_bond=chi)
    final_mps = circuit.psi
    return (final_mps.H @ mpo @ final_mps).real


if __name__ == "__main__":
    chi_dmrg_large = 1_000
    l = 8
    j_xy = 1
    j_z = 1
    h = XXZHamiltonian(
        j_xy, j_z, l,
        store_ref_vector=False,
        diag_mode="quimb", max_mpo_bond=MAX_MPO_BOND, max_mps_bond=chi_dmrg_large
    )
    dmrg_energy_large = h.ground_energy

    ham_mpo = qubop_to_mpo(h.operator, MAX_MPO_BOND)

    backend = FakeFez()

    num_layers = 3

    chis = list(range(4, 31, 2))
    energies = []
    depths = []
    approx_depths = []
    errs = []
    exact_energies = [] # Energies from exact compilation.
    approx_energies = [] # Energies from approximate compilation.
    exact_errors = [] # Errors from exact compilation.
    approx_errors = [] # Errors from approximate compilation.
    for chi in chis:
        print(f"chi = {chi}")
        dmrg = DMRG(ham_mpo, bond_dims=chi)
        converged = dmrg.solve()
        if not converged:
            print("DMRG did not converge!")
        ground_energy = dmrg.energy.real
        mps = dmrg.state
        mps_arrays = mps.arrays

        qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")
        qc_transpiled = qiskit.transpile(qc, backend=backend)
        qc2 = mps_to_circuit(
            mps_arrays, method="approximate", shape="lpr",
            chi_max=chi, compress=True# , num_layers=num_layers
        )
        qc2_transpiled = qiskit.transpile(qc2, backend=backend)

        # Simulate the circuits to get their errors.
        exact_energy = circuit_energy(ham_mpo, qc_transpiled, 10)
        approx_energy = circuit_energy(ham_mpo, qc2_transpiled, chi_dmrg_large)

        errs.append(abs(ground_energy - dmrg_energy_large))
        energies.append(ground_energy)
        exact_energies.append(exact_energy)
        exact_errors.append(abs(exact_energy - dmrg_energy_large))
        approx_energies.append(approx_energy)
        approx_errors.append(abs(approx_energy - dmrg_energy_large))
        depths.append(qc_transpiled.depth())
        approx_depths.append(qc2_transpiled.depth())

    output_data = {
        "chi": chis, "energy": energies, "error": errs, "depths": depths, "approx_depths": approx_depths,
        "exact_energy": exact_energies, "exact_error": exact_errors,
        "approx_energy": approx_energies, "approx_error": approx_errors
    }
    df = pd.DataFrame.from_dict(output_data, orient='columns')
    df.to_csv("mps_to_circuit_results.csv", index=False)
