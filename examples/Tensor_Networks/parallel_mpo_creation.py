from typing import Optional
from os import listdir
from itertools import batched
import pickle
# from multiprocessing import Pool
from pathos.pools import ProcessPool as Pool
import numpy as np
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct
from adaptvqe.tensor_helpers import qubop_to_mpo, tensor_product_mpo
from adaptvqe.hamiltonians import XXZHamiltonian


def qubop_to_mpo_parallel(
    qubop: of.QubitOperator, max_bond: int, nq: Optional[int]=None,
    n_workers: int=4
) -> MatrixProductOperator:
    """Convert an openfermion QubitOperator to an MPO."""

    if nq is None:
        nq = of.utils.count_qubits(qubop)

    def worker_callback(keys):
        for i, key in enumerate(keys):
            coeff = qubop.terms[key]
            matrices = [np.eye(2).astype(complex) for _ in range(nq)]
            for idx, pauli in key:
                if pauli == 'X':
                    matrices[idx] = np.array([[0., 1.], [1., 0.]]).astype(complex)
                elif pauli == 'Y':
                    matrices[idx] = np.array([[0., -1j], [1j, 0.]])
                else: # By default this is a Z.
                    matrices[idx] = np.array([[1., 0.], [0, -1.]]).astype(complex)
            term_mpo = tensor_product_mpo(matrices)
            if i == 0:
                total_mpo = coeff * term_mpo
            else:
                # total_mpo += coeff * term_mpo
                if len(total_mpo.tensors) == 1 and len(term_mpo.tensors) == 1:
                    m1 = total_mpo.tensors[0].data
                    m2 = coeff * term_mpo.tensors[0].data
                    total_mpo = MatrixProductOperator.from_dense(m1 + m2)
                else:
                    total_mpo += coeff * term_mpo
                tensor_network_1d_compress_direct(total_mpo, max_bond=max_bond, inplace=True)
        return total_mpo
    
    # TODO add a batch-and-checkpoint system where we save subsets of terms to files. Parallelize the chunks individually.
    keys = sorted(list(qubop.terms.keys()))
    # return worker_callback(keys)
    keys_per_batch = int(len(keys) / n_workers)
    batched_keys = list(batched(keys, keys_per_batch))
    pool = Pool(n_workers)
    all_mpos = pool.map(worker_callback, batched_keys)
    if len(all_mpos) == 1:
        return all_mpos[0]
    else:
        return sum(all_mpos[1:], start=all_mpos[0])


if __name__ == "__main__":
    j_xy = 1
    j_z = 1
    l = 4
    max_mpo_bond = 100
    dmrg_mps_bond = 14
    h = XXZHamiltonian(j_xy, j_z, l, diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=dmrg_mps_bond)
    qubop = h.operator

    standard_mpo = qubop_to_mpo(qubop, max_mpo_bond)
    parallel_mpo = qubop_to_mpo_parallel(qubop, max_mpo_bond)
    difference_mpo = standard_mpo - parallel_mpo
    print(difference_mpo.norm())