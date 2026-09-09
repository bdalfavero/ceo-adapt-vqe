import pandas as pd
from openfermion import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.pools import DVE_CEO
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.tensor_helpers import qubop_to_mpo

MAX_MPO_BOND = 1000
NUM_ITER = 20

if __name__ == "__main__":
    N = 4
    r = 1.5
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
    mol = run_pyscf(mol, run_fci=True, run_ccsd=False)  # CCSD doesn't work here?
    pool = DVE_CEO(mol)
    exact_energy = mol.fci_energy
    int_op = mol.get_molecular_hamiltonian()
    hamiltonian = get_fermion_operator(int_op)
    h = jordan_wigner(hamiltonian)
    ham_mpo = qubop_to_mpo(hamiltonian, MAX_MPO_BOND)
    print(f"FCI energy {exact_energy:5.4e}")

    chis = list(range(4, 21, 2))
    # chis = list(range(4, 6, 2))
    energies = []
    depths = []
    errs = []
    for chi in chis:
        my_adapt = TensorNetAdapt(
            pool=pool,
            molecule=mol,
            max_adapt_iter=NUM_ITER,
            recycle_hessian=True,
            tetris=True,
            verbose=True,
            threshold=0.1,
            max_mpo_bond=MAX_MPO_BOND,
            max_mps_bond=chi,
            skip_converged_rename=True
        )
        my_adapt.run()
        data = my_adapt.data
        energy = data.evolution.energies[-1]
        qc = data.get_circuit(pool,include_ref=True)
        energies.append(energy)
        errs.append(abs(energy - exact_energy))
        depths.append(qc.depth())

    output_data = {
        "chi": chis, "energy": energies, "error": errs, "depths": depths
    }
    df = pd.DataFrame.from_dict(output_data, orient='columns')
    df.to_csv("hchain_adapt_bond_results.csv", index=False)
