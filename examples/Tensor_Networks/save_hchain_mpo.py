import argparse
import pickle
import openfermion as of
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.tensor_helpers import qubop_to_mpo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, help="Number of sites.")
    parser.add_argument("--max-mpo-bond", type=int, default=int(1e6), help="Maximum bond dimension of MPO.")
    args = parser.parse_args()
    N = args.N
    max_mpo_bond = args.max_mpo_bond

    r = 1.5
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
    mol = run_pyscf(mol, run_fci=False, run_ccsd=True, run_scf=True)  # CCSD doesn't work here?
    hf_energy = mol.hf_energy
    exact_energy = mol.ccsd_energy
    print(f"hf_energy = {hf_energy}")
    print(f"exact_energy = {exact_energy}")

    hamiltonian = mol.get_molecular_hamiltonian()
    if isinstance(hamiltonian, of.QubitOperator):
        hamiltonian_mpo = qubop_to_mpo(hamiltonian, max_mpo_bond)
    else:
        ham_jw = of.transforms.jordan_wigner(hamiltonian)
        hamiltonian_mpo = qubop_to_mpo(ham_jw, max_mpo_bond)

    with open(f"hchain_mpo_N{N}_chi{max_mpo_bond}.pkl", "wb") as f:
        pickle.dump(hamiltonian_mpo, f)
