"""Start optimization with LinAlgAdapt, then hand off to the sampler."""

import pickle

import numpy as np
from scipy.sparse.linalg import expm, expm_multiply

from qiskit.qasm2 import dump

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, SampledLinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy

CIRCUIT_DIR = "circuits/"
NUM_LINALG_ITER = 2
NUM_ADAPT_ITER = 5

l = 4
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l)
pool = FullPauliPool(n=l)

print("Running LinAlgAdapt")
my_adapt = LinAlgAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=NUM_LINALG_ITER,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
)
my_adapt.run()
data = my_adapt.data

print("Running SampledLinAlgAdapt")
sampled_adapt = SampledLinAlgAdapt(
    # custom_hamiltonian=h,
    previous_data=data,
    pool=pool,
    verbose=True,
    threshold=10**-5,
    max_adapt_iter=NUM_LINALG_ITER + NUM_ADAPT_ITER,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
)
sampled_adapt.load(data)
sampled_adapt.run()
data = sampled_adapt.data

coefficients = data.result.ansatz.coefficients
indices = data.result.ansatz.indices

qc = data.get_circuit(pool,include_ref=True)
energy = get_circuit_energy(qc,h.operator)
print("\nEnergy from circuit: ", energy)
assert np.abs(energy-data.result.energy) < 10**-6
energy_err = np.abs(h.ground_energy - energy)
print(f"Ground state energy error {energy_err}")

energies = []
iter_circuit_fname_dict = {}
for i, (indices, coeffs) in enumerate(zip(data.evolution.indices, data.evolution.coefficients)):
    qc = data.get_circuit(pool, indices, coeffs, include_ref=True)
    circuit_fname = CIRCUIT_DIR + f"xxz_circuit{i}.qasm"
    iter_circuit_fname_dict[i] = circuit_fname
    dump(qc, circuit_fname)

output_data = {
    "num_linalg_iter": NUM_LINALG_ITER,
    "num_adapt_iter": NUM_ADAPT_ITER,
    "energies": data.evolution.energies,
    "iteration_circuits": iter_circuit_fname_dict
}
with open("xxz_handoff_results.pkl", "wb") as f:
    pickle.dump(output_data, f)