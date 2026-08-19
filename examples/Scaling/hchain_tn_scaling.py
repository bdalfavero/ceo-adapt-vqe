from time import perf_counter_ns
import numpy as np
import pandas as pd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.pools import DVE_CEO

MAX_MPO_BOND = 200
NUM_ITER = 10

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
    r = 1.5
    N = 3
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 2  # odd number of electrons
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description='H3')
    mol = run_pyscf(mol, run_fci=True, run_ccsd=False)  # CCSD doesn't work here?
    exact_energy = mol.fci_energy
    pool = DVE_CEO(mol)

    chi = 10
    my_adapt = TensorNetAdapt(
        pool=pool,
        molecule=mol,
        max_adapt_iter=NUM_ITER + 1,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.1,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=chi,
        skip_converged_rename=True
    )
    my_adapt.initialize()

    energies = []
    times = []
    num_iter = 15
    for _ in range(num_iter):
        start_time = perf_counter_ns()
        my_adapt.run_iteration()
        end_time = perf_counter_ns()
        elapsed_time = float(abs(end_time - start_time))
        energies.append(my_adapt.energy)
        times.append(elapsed_time)
    return ScalingResult(N, chi, exact_energy, times, energies)

if __name__ == "__main__":
    N = 4
    chi = 10
    result = run_n_chi(N, chi)
    df = result.to_dataframe()
    df.to_csv("hchain_results.csv", index=False)