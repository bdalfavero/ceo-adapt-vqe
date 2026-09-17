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

def get_circuit_mps(circuit, chi):
    ckt_qasm = dumps(circuit)
    circuit = qtn.CircuitMPS.from_openqasm2_str(ckt_qasm, max_bond=chi)
    final_mps = circuit.psi
    # return (final_mps.H @ mpo @ final_mps).real
    return final_mps


def mps_circuit_overlap(ckt1, ckt2, chi):
    mps1 = get_circuit_mps(ckt1, chi)
    mps2 = get_circuit_mps(ckt2, chi)
    return mps2 @ mps1

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
    ham_mpo = qubop_to_mpo(h.operator, MAX_MPO_BOND)

    backend = FakeFez()

    chi = 1_000
    dmrg = DMRG(ham_mpo, bond_dims=chi)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge!")
    ground_energy = dmrg.energy.real

    mps = dmrg.state
    mps_arrays = mps.arrays
    qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")

    num_layers_vals = [10, 50, 100, 500, 1000]
    square_overlaps = []
    for num_layers in num_layers_vals:
        qc2 = mps_to_circuit(
            mps_arrays, method="approximate", shape="lpr",
            chi_max=chi, compress=False, num_layers=num_layers
        )
        overlap = mps_circuit_overlap(qc, qc2, chi_dmrg_large)
        square_overlaps.append(np.abs(overlap) ** 2)
    
    output_data = {
        "num_layers": num_layers_vals, "square_overlaps": square_overlaps
    }
    df = pd.DataFrame.from_dict(output_data, orient="columns")
    df.to_csv("overlaps.csv", index=False)