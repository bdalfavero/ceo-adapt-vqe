import argparse
from os.path import isfile
import pickle
import openfermion as of
from adaptvqe.utils import hamiltonian_from_fcidump
from adaptvqe.tensor_helpers import qubop_to_mpo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fci_filename", type=str, help="fcidump filename for Hamiltonian.")
    parser.add_argument("chi_mpo", type=int, default=100, help="Max MPO bond dimension.")
    parser.add_argument("--mpo-filename", type=str, default="mpo.pkl", help="Pickle filename for MPO.")
    args = parser.parse_args()
    chi_mpo = args.chi_mpo
    fcidump_fname = args.fci_filename
    mpo_fname = args.mpo_filename

    h_int, norb, nelec = hamiltonian_from_fcidump(fcidump_fname)
    print(f"Read Hamiltonian with {norb} orbitals and {nelec} electrons.")
    # mpo_fname = f"product_mpo_chi{chi_mpo}.pkl" 
    if not isfile(mpo_fname):
        if isinstance(h_int, of.QubitOperator):
            hamiltonian_mpo = qubop_to_mpo(h_int, chi_mpo)
        else:
            ham_jw = of.transforms.jordan_wigner(h_int)
            hamiltonian_mpo = qubop_to_mpo(ham_jw, chi_mpo)

        with open(mpo_fname, "wb") as f:
            pickle.dump(hamiltonian_mpo, f)
    print(f"Wrote Hamiltonian to {mpo_fname}")