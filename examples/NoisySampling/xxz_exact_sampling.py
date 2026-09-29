import sys
import pickle
import numpy as np
from qiskit.qasm2 import dump
from adaptvqe.circuits import get_circuit_energy
sys.path.append(".")
from common import *

CIRCUIT_DIR = "exact_circuits/"
OUTPUT_FNAME = "exact_results.pkl"

if __name__ == "__main__":
    h, my_adapt, pool = build_adapt(simulate_exactly)
    my_adapt.run()
    data = my_adapt.data

    # coefficients = data.result.ansatz.coefficients
    # indices = data.result.ansatz.indices

    # qc = data.get_circuit(pool, include_ref=True)
    # energy = get_circuit_energy(qc, h.operator)
    # print("\nEnergy from circuit: ", energy)
    # assert np.abs(energy-data.result.energy) < 10**-6
    # energy_err = np.abs(h.ground_energy - energy)
    # print(f"Ground state energy error {energy_err}")

    save_data(CIRCUIT_DIR, OUTPUT_FNAME, pool, data)