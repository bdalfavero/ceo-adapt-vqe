"""Test out the checkpointing capabilities of the code with an XXZ model. Run some number of iterations
with checkpointing, then resume from the latest one."""

import os
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import FullPauliPool
from adaptvqe.utils import load_from_file

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
    checkpoint_interval=2,
    checkpoint_prefix=PREFIX
)
my_adapt.run()

dir_files = os.listdir(".")
checkpoint_files = [f for f in dir_files if PREFIX in f]
assert len(checkpoint_files) != 0

data = load_from_file(checkpoint_files[0])
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