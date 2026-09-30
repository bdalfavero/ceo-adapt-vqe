from typing import Optional
import pickle

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.qasm2 import dump
from qiskit_ibm_runtime.fake_provider import FakeFez

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, SampledLinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy

def simulate_exactly(qc: QuantumCircuit, qiskit_hamiltonian: SparsePauliOp, shots: Optional[int]) -> float:
    """Use an exact Aer simulator. See https://quantum.cloud.ibm.com/docs/en/guides/simulate-with-qiskit-aer"""

    exact_estimator = Estimator()
    # The circuit needs to be transpiled to the AerSimulator target
    pass_manager = generate_preset_pass_manager(0, AerSimulator())
    isa_circuit = pass_manager.run(qc)
    pub = (isa_circuit, qiskit_hamiltonian)
    job = exact_estimator.run([pub])
    result = job.result()
    pub_result = result[0]
    exact_value = float(pub_result.data.evs)
    return exact_value


def simulate_noisily(
    qc: QuantumCircuit, qiskit_hamiltonian: SparsePauliOp, shots: Optional[int],
    use_depolarizing: bool=False
) -> float:
    """Use an exact Aer simulator. See https://quantum.cloud.ibm.com/docs/en/guides/simulate-with-qiskit-aer"""

    # The circuit needs to be transpiled to the AerSimulator target
    pass_manager = generate_preset_pass_manager(0, AerSimulator())
    isa_circuit = pass_manager.run(qc)
    pub = (isa_circuit, qiskit_hamiltonian)
    # Use a noisy circuit simulator to get energies.
    if use_depolarizing:
        noise_model = NoiseModel()
        cx_depolarizing_prob = 0.02
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(cx_depolarizing_prob, 2), ["cx"]
        )
        noisy_estimator = Estimator(
            options=dict(backend_options=dict(noise_model=noise_model))
        )
    else:
        device_backend = FakeFez()
        noisy_estimator = BackendEstimatorV2(backend=device_backend)
    job = noisy_estimator.run([pub])
    result = job.result()
    pub_result = result[0]
    noisy_value = float(pub_result.data.evs)
    return noisy_value


def build_adapt(callback):
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
        custom_callback=callback
    )
    return h, my_adapt, pool


def save_data(circuit_dir: str, fname, pool, data):
    assert '/' in circuit_dir

    iter_circuit_fname_dict = {}
    for i, (indices, coeffs) in enumerate(zip(data.evolution.indices, data.evolution.coefficients)):
        qc = data.get_circuit(pool, indices, coeffs, include_ref=True)
        circuit_fname = circuit_dir + f"xxz_circuit{i}.qasm"
        iter_circuit_fname_dict[i] = circuit_fname
        dump(qc, circuit_fname)

    energies = np.array(data.evolution.energies)
    output_data = {
        "energies": energies.tolist(),
        "iteration_circuits": iter_circuit_fname_dict,
        "indices": data.evolution.indices,
        "coefficients": data.evolution.coefficients
    }
    with open(fname, "wb") as f:
        pickle.dump(output_data, f)