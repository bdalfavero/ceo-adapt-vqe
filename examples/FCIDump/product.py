"""Run ADAPT for the fcidump data at
https://github.com/rmlarose/calibrate-ibm/tree/main/experiments/L11-BE2-Fragments-Circuits/L11-BE2-Product-Circuits/L11_product_BE2_f13"""

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

MAX_MPO_BOND = 100
DMRG_MPS_BOND = 15
NUM_ITER = 20

if __name__ == "__main__":
    chi = 5
    fcidump_fname = "fcidump.txt"

    h_int, norb, nelec = hamiltonian_from_fcidump(fcidump_fname)
    print(f"Read Hamiltonian with {norb} orbitals and {nelec} electrons.")
    mpo_fname = f"product_mpo_chi{MAX_MPO_BOND}.pkl" 
    if not isfile(mpo_fname):
        if isinstance(h_int, of.QubitOperator):
            hamiltonian_mpo = qubop_to_mpo(h_int, MAX_MPO_BOND)
        else:
            ham_jw = of.transforms.jordan_wigner(h_int)
            hamiltonian_mpo = qubop_to_mpo(ham_jw, MAX_MPO_BOND)

        with open(mpo_fname, "wb") as f:
            pickle.dump(hamiltonian_mpo, f)
    print(f"Wrote Hamiltonian to {mpo_fname}")

    h = FermionicHamiltonian(
        h_int, "product", nelec, diag_mode="quimb",
        max_mps_bond=DMRG_MPS_BOND, max_mpo_bond=MAX_MPO_BOND
    )
    print(f"DMRG energy: {h.ground_energy}")

    pool = GSD(n=norb)
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
        "chi": chi,
        "dmrg_energy": h.ground_energy,
        "energies": energies,
        "indices": indices,
        "coefficients": coefficients
    }

    with open("product_results.pkl", "wb") as f:
        pickle.dump(output_dict, f)
