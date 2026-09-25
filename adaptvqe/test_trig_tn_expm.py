"""Confirm that the trigonometric-formula tn_expm_mult_state (MPO-based) gives the same states as the sparse
exponentials (LinAlg) and as the circuit-based tensor network exponentials."""

import unittest
from copy import deepcopy
import numpy as np
from quimb.tensor.tensor_1d import MatrixProductState
from adaptvqe.molecules import create_h2, create_h4
from adaptvqe.chemistry import get_hf_det
from adaptvqe.matrix_tools import ket_to_sparse_vector
from adaptvqe.tensor_helpers import computational_basis_mps
from adaptvqe.pools import (OperatorPool, ImplementationType, GSD, PairedGSD, SD, QE, QE_All, QE1, MVP_CEO,
                            OVP_CEO, DVG_CEO, DVE_CEO)

# Pools whose tn_expm_mult_state uses the trigonometric formula (directly or through inheritance).
POOL_CLASSES = [GSD, PairedGSD, SD, QE, QE_All, QE1, MVP_CEO, OVP_CEO, DVG_CEO, DVE_CEO]
# Pools left out of the comparison with the circuit-based method:
#  - SD and QE1: get_circuit takes no big_endian argument, so the circuit-based tensor network method fails.
#  - QE_All: for some operators (e.g. index 6 for H2) get_circuit doesn't match the sparse exponential,
#    so the circuit is wrong, not the trigonometric formula (which does match the sparse exponential).
NO_CIRCUIT_TN = [SD, QE1, QE_All]
TOL = 1e-10


def make_pools(pool_class, molecule):
    """Return a sparse and a tensor network copy of the pool, plus the matching Hartree-Fock reference states."""

    pool = pool_class(molecule)
    sparse_pool = deepcopy(pool)
    sparse_pool.imp_type = ImplementationType.SPARSE
    tn_pool = deepcopy(pool)
    tn_pool.imp_type = ImplementationType.TENSORS

    ref_det = get_hf_det(molecule.n_electrons, pool.n)
    sparse_ref_state = ket_to_sparse_vector(ref_det).astype(complex)
    tn_ref_state = computational_basis_mps(ref_det)
    return sparse_pool, tn_pool, sparse_ref_state, tn_ref_state


def fidelity(mps_1, mps_2):
    return abs(mps_1.H @ mps_2) ** 2 / ((mps_1.H @ mps_1).real * (mps_2.H @ mps_2).real)


class TestTrigTNExpm(unittest.TestCase):

    def check_against_sparse(self, molecule, depth, seed=0):
        """Apply a random ansatz with both methods and compare the final states (no truncation)."""

        rng = np.random.default_rng(seed)
        for pool_class in POOL_CLASSES:
            with self.subTest(pool=pool_class.__name__):
                sparse_pool, tn_pool, sparse_state, tn_state = make_pools(pool_class, molecule)
                indices = rng.choice(sparse_pool.size, size=depth)
                coefficients = rng.uniform(-np.pi, np.pi, size=depth)
                for coefficient, index in zip(coefficients, indices):
                    sparse_state = sparse_pool.expm_mult(coefficient, index, sparse_state)
                    tn_state = tn_pool.tn_expm_mult_state(coefficient, index, tn_state)

                dense = np.asarray(sparse_state.todense() if hasattr(sparse_state, "todense") else sparse_state)
                sparse_mps = MatrixProductState.from_dense(dense.flatten())
                # The exponentials are unitary, so the norm must be preserved as well.
                self.assertAlmostEqual((tn_state.H @ tn_state).real, 1., delta=TOL)
                self.assertGreater(fidelity(sparse_mps, tn_state), 1 - TOL)

    def test_h2_against_sparse(self):
        self.check_against_sparse(create_h2(1.5), depth=6)

    def test_h4_against_sparse(self):
        self.check_against_sparse(create_h4(1.5), depth=10)

    def test_each_operator_against_circuit(self):
        """Every operator of the pool, one at a time, against the circuit-based method."""

        molecule = create_h2(1.5)
        rng = np.random.default_rng(1)
        for pool_class in POOL_CLASSES:
            if pool_class in NO_CIRCUIT_TN:
                continue
            with self.subTest(pool=pool_class.__name__):
                _, tn_pool, _, tn_ref_state = make_pools(pool_class, molecule)
                for index in range(tn_pool.size):
                    coefficient = rng.uniform(-np.pi, np.pi)
                    trig_state = tn_pool.tn_expm_mult_state(coefficient, index, tn_ref_state.copy(), big_endian=False)
                    circuit_state = OperatorPool.tn_expm_mult_state(
                        tn_pool, coefficient, index, tn_ref_state.copy(), max_bond=None, big_endian=False
                    )
                    # Compare the overlap itself (not only its modulus) to also check the global phase.
                    self.assertAlmostEqual(circuit_state.H @ trig_state, 1., delta=1e-8)

    def test_zero_coefficient(self):
        """quimb can't multiply an MPS by 0, check that the workaround returns the input state."""

        _, tn_pool, _, tn_ref_state = make_pools(GSD, create_h2(1.5))
        state = tn_pool.tn_expm_mult_state(0., 0, tn_ref_state.copy())
        self.assertGreater(fidelity(state, tn_ref_state), 1 - TOL)

    def test_big_endian_not_supported(self):
        _, tn_pool, _, tn_ref_state = make_pools(QE, create_h2(1.5))
        with self.assertRaises(NotImplementedError):
            tn_pool.tn_expm_mult_state(0.1, 0, tn_ref_state, big_endian=True)


if __name__ == "__main__":
    unittest.main()
