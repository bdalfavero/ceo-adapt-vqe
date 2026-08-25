"""Test contracting a circuit on GPU."""

import qiskit
from qiskit.qasm2 import dumps
import quimb.tensor as qtn
import torch

def to_backend(x, device_name="cuda"):
    return torch.tensor(x, dtype=torch.complex64, device=device_name)

max_bond = 5

circuit = qiskit.QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
qasm_str = dumps(circuit)

circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
    qasm_str, max_bond=max_bond, progbar=False,
    to_backend=lambda x: to_backend(x, device_name="mps")
)