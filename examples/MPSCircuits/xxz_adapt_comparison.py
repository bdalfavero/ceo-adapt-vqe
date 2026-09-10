import pandas as pd
import qiskit
from qiskit_ibm_runtime.fake_provider import FakeFez
from openfermion import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.pools import FullPauliPool
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.tensor_helpers import qubop_to_mpo

MAX_MPO_BOND = 1000
NUM_ITER = 30

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
    exact_energy = h.ground_energy
    print(f"DMRG energy = {exact_energy}")

    pool = FullPauliPool(n=h.n)
    chi = 31

    backend = FakeFez()
    
    energies = []
    depths = []
    errs = []
    my_adapt = TensorNetAdapt(
        pool=pool,
        custom_hamiltonian=h,
        max_adapt_iter=NUM_ITER,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.1,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=chi,
        skip_converged_rename=True
    )
    for _ in range(NUM_ITER):
        my_adapt.initialize()
        my_adapt.run_iteration()
        data = my_adapt.data
        qc = data.get_circuit(
            pool, indices=my_adapt.indices, coefficients=my_adapt.coefficients, include_ref=True
        )
        qc_transpiled = qiskit.transpile(qc, backend=backend)
        energies.append(my_adapt.energy)
        errs.append(abs(my_adapt.energy - exact_energy))
        depths.append(qc_transpiled.depth())


    output_data = {
        "energy": energies, "error": errs, "depths": depths
    }
    df = pd.DataFrame.from_dict(output_data, orient='columns')
    df.to_csv("xxz_adapt_bond_results.csv", index=False)
