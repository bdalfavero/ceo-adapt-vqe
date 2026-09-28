"""Train with a noisy simulator instead of an exact one."""

from typing import Optional
import pickle

import numpy as np
from scipy.sparse.linalg import expm, expm_multiply

from openfermion import get_sparse_operator

from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, SampledLinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy

CIRCUIT_DIR = "noisy_circuits/"

def simulate_exactly(qc: QuantumCircuit, qiskit_hamiltonian: SparsePauliOp, shots: Optional[int]) -> float:
    """Use an exact Aer simulator. See https://quantum.cloud.ibm.com/docs/en/guides/simulate-with-qiskit-aer"""

    # The circuit needs to be transpiled to the AerSimulator target
    pass_manager = generate_preset_pass_manager(0, AerSimulator())
    isa_circuit = pass_manager.run(qc)
    pub = (isa_circuit, qiskit_hamiltonian)
    # Use a noisy circuit simulator to get energies.
    noise_model = NoiseModel()
    cx_depolarizing_prob = 0.02
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(cx_depolarizing_prob, 2), ["cx"]
    )

    noisy_estimator = Estimator(
        options=dict(backend_options=dict(noise_model=noise_model))
    )
    job = noisy_estimator.run([pub])
    result = job.result()
    pub_result = result[0]
    noisy_value = float(pub_result.data.evs)
    return noisy_value


if __name__ == "__main__":
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
        max_adapt_iter=6,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
        custom_callback=simulate_exactly
    )
    my_adapt.run()
    data = my_adapt.data

    coefficients = data.result.ansatz.coefficients
    indices = data.result.ansatz.indices

    qc = data.get_circuit(pool,include_ref=True)
    energy = get_circuit_energy(qc,h.operator)
    print("\nEnergy from circuit: ", energy)
    # assert np.abs(energy-data.result.energy) < 10**-6
    energy_err = np.abs(h.ground_energy - energy)
    print(f"Ground state energy error {energy_err}")

    iter_circuit_fname_dict = {}
    for i, (indices, coeffs) in enumerate(zip(data.evolution.indices, data.evolution.coefficients)):
        qc = data.get_circuit(pool, indices, coeffs, include_ref=True)
        circuit_fname = CIRCUIT_DIR + f"xxz_circuit{i}.qasm"
        iter_circuit_fname_dict[i] = circuit_fname
        dump(qc, circuit_fname)

    energies = np.array(data.evolution.energies)
    output_data = {
        "energies": energies.tolist(),
        "iteration_circuits": iter_circuit_fname_dict,
        "indices": data.evolution.indices,
        "coefficients": data.evolution.coefficients
    }
    with open("noisy_results.pkl", "wb") as f:
        pickle.dump(output_data, f)