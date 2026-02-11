import unittest

import numpy as np

from adaptvqe.pools import FullPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

class TestParallelRankGradients(unittest.TestCase):

    def test_xxz(self):
        """Rank gradients in serial and in parallel and test that the results are the same."""

        l = 4
        j_xy = 1
        j_z = 1
        h = XXZHamiltonian(j_xy, j_z, l)
        pool = FullPauliPool(n=l)
        my_adapt = LinAlgAdapt(
            pool=pool,
            custom_hamiltonian=h,
            verbose=False,
            threshold=10**-5,
            max_adapt_iter=5,
            max_opt_iter=10000,
            sel_criterion="gradient",
            recycle_hessian=False,
            rand_degenerate=True,
        )
        my_adapt.initialize()
        my_adapt.run_iteration()

        # Rank in serial.
        sel_indices1, sel_gradients1, total_norm1, max_norm1 = my_adapt.rank_gradients(
            my_adapt.coefficients, my_adapt.indices, nworkers=1
        )

        # Rank in parallel.
        sel_indices2, sel_gradients2, total_norm2, max_norm2 = my_adapt.rank_gradients(
            my_adapt.coefficients, my_adapt.indices, nworkers=2
        )

        # Because some operators might have the same gradient, we cannot test based on
        # the indices.
        self.assertTrue(np.allclose(sel_gradients1, sel_gradients2))

if __name__ == '__main__':
    unittest.main()