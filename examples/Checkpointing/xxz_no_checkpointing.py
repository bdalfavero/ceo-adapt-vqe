"""Do what we would have done before - Run for some max number of iterations, save the data
to a file after the failure, then initialize again. Make sure that my edits have not broken anything."""

import os
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import FullPauliPool
from adaptvqe.utils import load_from_file, save_to_file

PREFIX = "xxz_checkpoint"

l = 4
j_xy = 1.
j_z = 2.
h = XXZHamiltonian(j_xy, j_z, l, diag_mode="quimb")
pool = FullPauliPool(n=l)

my_adapt = LinAlgAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=4,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
)
my_adapt.run()
resume_fname = my_adapt.data.file_name
save_to_file(my_adapt.data, resume_fname)

data = load_from_file(resume_fname)
my_adapt = LinAlgAdapt(
    pool=pool,
    # custom_hamiltonian=h,
    previous_data=data,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=20,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
    checkpoint_interval=2,
    checkpoint_prefix=PREFIX
)
my_adapt.run()