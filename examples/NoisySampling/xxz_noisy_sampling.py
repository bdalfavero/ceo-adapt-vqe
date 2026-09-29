"""Train with a noisy simulator instead of an exact one."""

from typing import Optional
import sys
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

from common import *

CIRCUIT_DIR = "noisy_circuits/"
OUTPUT_FNAME = "noisy_results.pkl"

if __name__ == "__main__":
    h, my_adapt, pool = build_adapt(simulate_noisily)
    my_adapt.run()
    data = my_adapt.data
    save_data(CIRCUIT_DIR, OUTPUT_FNAME, pool, data)