from os.path import isfile
import pickle
from pyscf import scf
from pyscf.tools import fcidump  
import openfermion as of
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.hamiltonians import FermionicHamiltonian
from adaptvqe.pools import DVE_CEO, GSD, PairedGSD
from adaptvqe.utils import hamiltonian_from_fcidump
from adaptvqe.tensor_helpers import qubop_to_mpo

DMRG_MPS_BOND = 25
MAX_MPO_BOND = 200
NUM_ITER = 10

if __name__ == "__main__":
    chi = 15
    N = 8
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

    fcidump_fname = f"H{N}.fcidump"
    h1e = mol.one_body_integrals
    eri = mol.two_body_integrals  # in chemists' notation (pq|rs)
    norb = h1e.shape[0]
    nelec = mol.n_electrons
    spin = mol.multiplicity - 1  # 2S = multiplicity - 1
    fcidump.from_integrals(
        fcidump_fname,
        h1e, eri, norb, nelec,
        ms=spin
    )
    h_int, norb, nelec = hamiltonian_from_fcidump(fcidump_fname)
    h = FermionicHamiltonian(
        h_int, f"H{N}", nelec, diag_mode="quimb",
        max_mps_bond=DMRG_MPS_BOND, max_mpo_bond=MAX_MPO_BOND
    )
    print(f"DMRG energy: {h.ground_energy}")

    mpo_fname = f"hchain_mpo_N{N}_chi{MAX_MPO_BOND}.pkl" 
    if not isfile(mpo_fname):
        hamiltonian = mol.get_molecular_hamiltonian()
        if isinstance(hamiltonian, of.QubitOperator):
            hamiltonian_mpo = qubop_to_mpo(hamiltonian, MAX_MPO_BOND)
        else:
            ham_jw = of.transforms.jordan_wigner(hamiltonian)
            hamiltonian_mpo = qubop_to_mpo(ham_jw, MAX_MPO_BOND)

        with open(mpo_fname, "wb") as f:
            pickle.dump(hamiltonian_mpo, f)

    pool = GSD(mol)
    my_adapt = TensorNetAdapt(
        pool=pool,
        custom_hamiltonian=h,
        # molecule=mol,
        max_adapt_iter=NUM_ITER + 1,
        recycle_hessian=True,
        # tetris=True,
        verbose=True,
        threshold=0.1,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=chi,
        skip_converged_rename=True,
        mpo_filename=mpo_fname
    )
    my_adapt.run()

    data = my_adapt.data
    indices = data.evolution.indices
    coefficients = data.evolution.coefficients
    energies = data.evolution.energies
    
    output_dict = {
        "N": N,
        "hf_energy": hf_energy,
        "exact_energy": exact_energy,
        "energies": energies,
        "indices": indices,
        "coefficients": coefficients
    }

    with open("hchain_results.pkl", "wb") as f:
        pickle.dump(output_dict, f)

    print(exact_energy, energies[-1])