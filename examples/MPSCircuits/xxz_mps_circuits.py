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
    # return (final_mps.H @ mpo @ final_mps).real
    return (final_mps.H @ final_mps.gate_with_mpo(mpo)).real


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

    num_layers_vals = [10, 50, 100]
    chis = list(range(4, 31, 2))

    records = []
    for chi in chis:
        print(f"chi = {chi}")
        dmrg = DMRG(ham_mpo, bond_dims=chi)
        converged = dmrg.solve()
        if not converged:
            print("DMRG did not converge!")
        ground_energy = dmrg.energy.real
        dmrg_err = abs(dmrg_energy_large - ground_energy)
        records.append((chi, "DMRG", 0, ground_energy, dmrg_err, 0, 0))

        mps = dmrg.state
        mps_arrays = mps.arrays
        qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")
        qc_transpiled = qiskit.transpile(qc, backend=backend)
        exact_energy = circuit_energy(ham_mpo, qc, chi_dmrg_large)
        exact_error = abs(dmrg_energy_large - exact_energy)
        records.append((chi, "exact", 0, exact_energy, exact_error, qc.depth(), qc_transpiled.depth()))

        for num_layers in num_layers_vals:
            qc2 = mps_to_circuit(
                mps_arrays, method="approximate", shape="lpr",
                chi_max=chi, compress=True, num_layers=num_layers
            )
            qc2_transpiled = qiskit.transpile(qc2, backend=backend)
            approx_energy = circuit_energy(ham_mpo, qc2, chi_dmrg_large)
            approx_error = abs(dmrg_energy_large - approx_energy)
            records.append((chi, "approximate", num_layers, approx_energy, approx_error, qc2.depth(), qc2_transpiled.depth()))

    columns = ["chi", "method", "num_layers", "energy", "error", "pretrans_depth", "posttrans_depth"]
    df = pd.DataFrame.from_records(records, columns=columns)
    df.to_csv("mps_to_circuit_results.csv", index=False)
