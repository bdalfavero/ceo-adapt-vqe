from time import perf_counter_ns
import numpy as np
import pandas as pd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.pools import DVE_CEO, FullPauliPool, TiledPauliPool
from adaptvqe.hamiltonians import XXZHamiltonian

MAX_MPO_BOND = 200
NUM_ITER = 10

max_mpo_bond = 100
dmrg_mps_bond = 10
adapt_mps_bond = 5
l = 4

j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l, diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=dmrg_mps_bond)
dmrg_energy = h.ground_energy
print(f"Got DMRG energy {dmrg_energy:4.5e}")
pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond)

# Run 200 iterations of ADAPT-VQE for small problem instance, selecting randomly among degenerate gradients.
# Form a list of all unique operators ever selected for this small instance.
ixs = []
for _ in range(100):
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
    my_adapt.run()
    data = my_adapt.data
    for i in data.result.ansatz.indices:
        if i not in ixs:
            ixs.append(i)

print(f"Pool will be tiled from {len(ixs)} ops")
source_ops = [pool.operators[index].operator for index in ixs]

class ScalingResult:
    def __init__(self, N: int, chi: int, exact_energy: float, times: list[float], energies: list[float]):
        assert len(times) == len(energies)

        self._N = N
        self._chi = chi
        self._exact_energy = exact_energy
        self._times = np.array(times)
        self._energies = np.array(energies)

    def to_dataframe(self) -> pd.DataFrame:
        abs_errors = np.abs(self._energies - self._exact_energy)
        rel_errors = abs_errors / abs(self._exact_energy)

        df = pd.DataFrame(
            data={
                "iteration": np.array(range(self._times.size)),
                "time": self._times, "energy": self._energies,
                "abs_error": abs_errors, "rel_error": rel_errors
            }
        )
        df["N"] = self._N
        df["chi"] = self._chi
        # df.set_index("iteration", inplace=True)
        return df


def run_n_chi(N: int, chi: int, num_iter: int=NUM_ITER):
    new_l = N
    j_xy = 1
    j_z = 1
    h = XXZHamiltonian(
        j_xy, j_z, new_l,
        store_ref_vector=False,
        diag_mode="quimb", max_mpo_bond=MAX_MPO_BOND, max_mps_bond=dmrg_mps_bond
    )
    dmrg_energy = h.ground_energy
    print(f"Got DMRG energy {dmrg_energy:4.5e}")
    tiled_pool = TiledPauliPool(n=new_l, source_ops=source_ops)

    tn_adapt = TensorNetAdapt(
        pool=tiled_pool,
        custom_hamiltonian=h,
        verbose=True,
        threshold=10**-5,
        max_adapt_iter=30,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
        max_mpo_bond=max_mpo_bond,
        max_mps_bond=adapt_mps_bond
    )
    tn_adapt.initialize()

    adapt_energies = []
    adapt_times = []
    for _ in range(num_iter):
        start_time = perf_counter_ns()
        tn_adapt.run_iteration()
        end_time = perf_counter_ns()
        elapsed_time = end_time - start_time
        adapt_energies.append(tn_adapt.energy)
        adapt_times.append(elapsed_time)
    return ScalingResult(N, chi, dmrg_energy, adapt_times, adapt_energies)


if __name__ == "__main__":
    dfs = []
    for N in [8, 10, 20, 30]:
        for chi in [5, 7, 9]:
            print(f"N={N} chi={chi}")
            result = run_n_chi(N, chi, num_iter=20)
            df = result.to_dataframe()
            dfs.append(df)
            total_df = pd.concat(dfs)
            total_df.to_csv("xxz_results.csv", index=False)