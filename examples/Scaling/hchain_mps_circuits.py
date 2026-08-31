"""Do DMRG at multiple bond dimensions and get data about the circuit."""

import numpy as np
import pandas as pd
import quimb.tensor as qtn
from quimb.tensor import DMRG
import qiskit
from qiskit_ibm_runtime.fake_provider import FakeFez
from mps_to_circuit import mps_to_circuit
from openfermion import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.tensor_helpers import qubop_to_mpo

MAX_MPO_BOND = 1000

if __name__ == "__main__":
    N = 4
    r = 1.5
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
    mol = run_pyscf(mol, run_fci=True, run_ccsd=False)  # CCSD doesn't work here?
    exact_energy = mol.fci_energy
    int_op = mol.get_molecular_hamiltonian()
    hamiltonian = get_fermion_operator(int_op)
    h = jordan_wigner(hamiltonian)
    ham_mpo = qubop_to_mpo(hamiltonian, MAX_MPO_BOND)
    print(f"FCI energy {exact_energy:5.4e}")

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
        errs.append(abs(ground_energy - exact_energy))
        energies.append(ground_energy)
        depths.append(qc_transpiled.depth())

    output_data = {
        "chi": chis, "energy": energies, "error": errs, "depths": depths
    }
    df = pd.DataFrame.from_dict(output_data, orient='columns')
    df.to_csv("hchain_mps_to_circuit_results.csv", index=False)
