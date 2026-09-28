"""The directory circuits/ contains circuits for the XXZ model. Simulate them with a
noisy simulator to see how well they do."""

import pickle
import numpy as np
from qiskit.qasm2 import load
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.op_conv import to_qiskit_operator

CIRCUIT_DIR = "circuits/"

if __name__ == "__main__":
    l = 4
    j_xy = 1
    j_z = 1
    h = XXZHamiltonian(j_xy, j_z, l)
    qiskit_hamiltonian = to_qiskit_operator(h.operator)

    with open("exact_results.pkl", "rb") as f:
        handoff_data = pickle.load(f)

    noisy_energies = []
    for i, f in handoff_data["iteration_circuits"].items():
        qc = load(f)

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
        noisy_energies.append(noisy_value)
    noisy_energies = np.array(noisy_energies)
    np.save("noisy_energies", noisy_energies)