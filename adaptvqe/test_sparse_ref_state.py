import unittest
import numpy as np
from scipy.sparse import csc_matrix

from adaptvqe.matrix_tools import ket_to_vector, ket_to_sparse_vector
from adaptvqe.chemistry import get_hf_det


def dense_ref_state(ket):
    """Replicate the original two-step construction from hamiltonians.py:343."""
    return csc_matrix(ket_to_vector(ket)).transpose()


class TestKetToSparseVector(unittest.TestCase):

    def _assert_equivalent(self, ket):
        expected = dense_ref_state(ket)
        result = ket_to_sparse_vector(ket)
        self.assertEqual(result.shape, expected.shape)
        diff = (result - expected)
        self.assertAlmostEqual(diff.nnz, 0)

    # --- Hartree-Fock states ---

    def test_hf_2e_4q(self):
        self._assert_equivalent(get_hf_det(2, 4))

    def test_hf_4e_8q(self):
        self._assert_equivalent(get_hf_det(4, 8))

    def test_hf_6e_12q(self):
        self._assert_equivalent(get_hf_det(6, 12))

    # --- Néel states (alternating 1010... and 0101...) ---

    def test_neel_4q_1010(self):
        neel = [1, 0, 1, 0]
        self._assert_equivalent(neel)

    def test_neel_4q_0101(self):
        neel = [0, 1, 0, 1]
        self._assert_equivalent(neel)

    def test_neel_6q(self):
        neel = [1, 0, 1, 0, 1, 0]
        self._assert_equivalent(neel)

    # --- Edge cases ---

    def test_all_zeros(self):
        self._assert_equivalent([0, 0, 0, 0])

    def test_all_ones(self):
        self._assert_equivalent([1, 1, 1, 1])

    def test_single_qubit_zero(self):
        self._assert_equivalent([0])

    def test_single_qubit_one(self):
        self._assert_equivalent([1])

    def test_two_qubits_10(self):
        self._assert_equivalent([1, 0])

    def test_two_qubits_01(self):
        self._assert_equivalent([0, 1])

    def test_arbitrary_8q(self):
        self._assert_equivalent([1, 0, 0, 1, 1, 0, 1, 0])

    # --- Little-endian flag ---

    def test_little_endian_matches_reversed_big_endian(self):
        ket = [1, 0, 1, 1]
        big_endian_result = ket_to_sparse_vector(ket[::-1])
        little_endian_result = ket_to_sparse_vector(ket, little_endian=True)
        diff = (big_endian_result - little_endian_result)
        self.assertAlmostEqual(diff.nnz, 0)

    def test_little_endian_hf(self):
        ket = get_hf_det(2, 4)
        big_result = ket_to_sparse_vector(ket)
        little_result = ket_to_sparse_vector(ket[::-1], little_endian=True)
        diff = (big_result - little_result)
        self.assertAlmostEqual(diff.nnz, 0)

    # --- Structural properties ---

    def test_result_is_csc_matrix(self):
        result = ket_to_sparse_vector([1, 0, 1, 0])
        self.assertIsInstance(result, csc_matrix)

    def test_single_nonzero_entry(self):
        for ket in [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]:
            result = ket_to_sparse_vector(ket)
            self.assertEqual(result.nnz, 1)

    def test_correct_index_for_known_ket(self):
        # |1010> in big-endian = index 1*8 + 0*4 + 1*2 + 0*1 = 10
        result = ket_to_sparse_vector([1, 0, 1, 0])
        dense = result.toarray().flatten()
        self.assertEqual(np.argmax(dense), 10)
        self.assertAlmostEqual(dense[10], 1.0)

    def test_norm_is_one(self):
        for ket in [[1, 0, 0, 0], [0, 0, 1, 1], get_hf_det(3, 6)]:
            result = ket_to_sparse_vector(ket)
            self.assertAlmostEqual(np.linalg.norm(result.toarray()), 1.0)


if __name__ == "__main__":
    unittest.main()
