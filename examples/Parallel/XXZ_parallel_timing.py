from time import perf_counter_ns

import numpy as np
import pandas as pd

from adaptvqe.pools import FullPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, TensorNetAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

max_mpo_bond = 100
max_mps_bond = 10

system_sizes = [4, 6, 8]
nworkers = [1, 2, 4, 6]
all_records = []

for l in system_sizes:
    for ncores in nworkers:
        j_xy = 1
        j_z = 1
        h = XXZHamiltonian(
            j_xy, j_z, l, diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=max_mps_bond
        )
        pool = FullPauliPool(n=l)
        my_adapt = TensorNetAdapt(
            pool=pool,
            custom_hamiltonian=h,
            verbose=False,
            threshold=10 ** -5,
            max_adapt_iter=5,
            max_opt_iter=10000,
            sel_criterion="gradient",
            recycle_hessian=False,
            rand_degenerate=True,
            max_mpo_bond=max_mpo_bond,
            max_mps_bond=max_mps_bond,
        )
        my_adapt.initialize()

        all_times = []
        for _ in range(6):
            start_time = perf_counter_ns()
            my_adapt.run_iteration()
            end_time = perf_counter_ns()
            elapsed_time = abs(end_time - start_time)
            all_times.append(elapsed_time)
        avg_time = np.average(all_times)
        all_records.append((l, ncores, avg_time))

df  = pd.DataFrame(all_records, columns=["l", "ncores", "avg_time"])
df.to_csv("xxz_parallel_times.csv")