from time import perf_counter_ns
import torch
from adaptvqe.pools import FullPauliPool
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt

def to_backend(x, device_name="cuda"):
    return torch.tensor(x, dtype=torch.complex64, device=device_name)

max_mpo_bond = 100
dmrg_mps_bond = 15

l = 4
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l, diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=dmrg_mps_bond)
dmrg_energy = h.ground_energy
print(f"Got DMRG energy {dmrg_energy:4.5e}")
pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond)

my_adapt = TensorNetAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=5,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
    max_mpo_bond=100,
    max_mps_bond = 20
)
my_adapt.initialize()
start_time = perf_counter_ns()
my_adapt.run_iteration()
end_time = perf_counter_ns()
elapsed = abs(end_time - start_time)
print(f"No GPU: elapsed time {elapsed:5.4e} ns")

pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond, to_backend=lambda x: to_backend(x, device_name="mps"))
my_adapt = TensorNetAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=5,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
    max_mpo_bond=100,
    max_mps_bond = 20,
    to_backend=lambda x: to_backend(x, device_name="mps")
)
my_adapt.initialize()
start_time = perf_counter_ns()
my_adapt.run_iteration()
end_time = perf_counter_ns()
elapsed = abs(end_time - start_time)
print(f"GPU: elapsed time {elapsed:5.4e} ns")