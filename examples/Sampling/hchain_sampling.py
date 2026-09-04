import numpy as np
from scipy.sparse.linalg import expm, expm_multiply

from openfermion import get_sparse_operator
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from qiskit.quantum_info import Operator, process_fidelity

from adaptvqe.pools import FullPauliPool, TiledPauliPool, PairedGSD
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, SampledLinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy


N = 4
r = 1.5
geometry = [['H', [0, 0, i * r]] for i in range(N)]
basis = 'sto-3g'
multiplicity = 1
charge = 0
mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
mol = run_pyscf(mol, run_fci=True, run_ccsd=True, run_scf=True)  # CCSD doesn't work here?
hf_energy = mol.hf_energy
exact_energy = mol.fci_energy
print(f"hf_energy = {hf_energy}")
print(f"exact_energy = {exact_energy}")

pool = PairedGSD(mol)

my_adapt = SampledLinAlgAdapt(
    molecule=mol,
    pool=pool,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=10,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
)
my_adapt.run()
data = my_adapt.data

coefficients = data.result.ansatz.coefficients
indices = data.result.ansatz.indices

qc = data.get_circuit(pool,include_ref=True)
hamiltonian = mol.get_molecular_hamiltonian()
energy = get_circuit_energy(qc, hamiltonian)
print("\nEnergy from circuit: ", energy)
fci_err = np.abs(exact_energy - energy)
print(f"FCI error = {fci_err:5.4e}")
assert np.abs(energy-data.result.energy) < 10**-6