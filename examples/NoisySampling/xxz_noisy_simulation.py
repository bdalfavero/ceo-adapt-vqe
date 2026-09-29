"""The directory circuits/ contains circuits for the XXZ model. Simulate them with a
noisy simulator to see how well they do."""

import sys
import pickle
import numpy as np
from qiskit.qasm2 import load
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.op_conv import to_qiskit_operator
from common import simulate_noisily

CIRCUIT_DIR = "circuits/"
SHOTS = None

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
        noisy_value = simulate_noisily(qc, qiskit_hamiltonian, SHOTS)
        noisy_energies.append(noisy_value)
    noisy_energies = np.array(noisy_energies)
    np.save("noisy_energies", noisy_energies)