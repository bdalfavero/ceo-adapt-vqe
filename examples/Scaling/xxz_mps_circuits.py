"""Do DMRG at multiple bond dimensions and get data about the circuit."""

import numpy as np
import pandas as pd
import quimb.tensor as qtn
from quimb.tensor import DMRG
import qiskit
from qiskit_ibm_runtime.fake_provider import FakeFez
from mps_to_circuit import mps_to_circuit
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.tensor_helpers import qubop_to_mpo

MAX_MPO_BOND = 1000

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

    chis = list(range(4, 31, 2))
    energies = []
    depths = []
    errs = []
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
        errs.append(abs(ground_energy - dmrg_energy_large))
        energies.append(ground_energy)
        depths.append(qc_transpiled.depth())

    output_data = {
        "chi": chis, "energy": energies, "error": errs, "depths": depths
    }
    df = pd.DataFrame.from_dict(output_data, orient='columns')
    df.to_csv("mps_to_circuit_results.csv", index=False)
