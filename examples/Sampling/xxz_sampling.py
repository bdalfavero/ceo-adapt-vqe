import numpy as np
from scipy.sparse.linalg import expm, expm_multiply

from openfermion import get_sparse_operator
from qiskit.quantum_info import Operator, process_fidelity

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, SampledLinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy

l = 4
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l)
pool = FullPauliPool(n=l)

my_adapt = SampledLinAlgAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=5,
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
energy = get_circuit_energy(qc,h.operator)
print("\nEnergy from circuit: ", energy)
assert np.abs(energy-data.result.energy) < 10**-6