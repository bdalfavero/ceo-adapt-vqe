"""Run ADAPT for the fcidump data at
https://github.com/rmlarose/calibrate-ibm/tree/main/experiments/L11-BE2-Fragments-Circuits/L11-BE2-Product-Circuits/L11_product_BE2_f13"""

from os.path import isfile
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("fci_filename", type=str, help="fcidump filename for Hamiltonian.")
    parser.add_argument("mpo_filename", type=str, help="Pickle filename for Hamiltonian MPO.")
    parser.add_argument("--chi", type=int, default=10, help="Max MPS bond dimension")
    parser.add_argument("--chi_mpo", type=int, default=100, help="Max MPO bond dimension")
    parser.add_argument("--num-iter", type=int, default=50, help="Number of ADAPT iterations")
    args = parser.parse_args()
    chi = args.chi
    chi_mpo = args.chi_mpo
    num_iter = args.num_iter
    fcidump_fname = args.fci_filename
    mpo_fname = args.mpo_filename

    h_int, norb, nelec = hamiltonian_from_fcidump(fcidump_fname)
    with open(mpo_fname, "rb") as f:
        hamiltonian_mpo = pickle.load(f)

    h = FermionicHamiltonian(
        h_int, "product", nelec, diag_mode="quimb",
        max_mps_bond=chi_mpo, max_mpo_bond=chi_mpo,
        mpo_filename=mpo_fname
    )
    print(f"DMRG energy: {h.ground_energy}")

    pool = GSD(n=norb)
    my_adapt = TensorNetAdapt(
        pool=pool,
        custom_hamiltonian=h,
        # molecule=mol,
        max_adapt_iter=num_iter,
        recycle_hessian=True,
        # tetris=True,
        verbose=True,
        threshold=0.1,
        max_mpo_bond=chi_mpo,
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
