"""Compare the timing of exponentiation with LinAlgAdapt and TensorNetAdapt."""

from time import perf_counter_ns
from random import sample
import numpy as np
import pandas as pd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.pools import DVE_CEO, GSD, PairedGSD


MAX_MPO_BOND = 1000
MAX_DEPTH = 20


def test_adapt_timing(adapt: TensorNetAdapt | LinAlgAdapt, max_depth: int=5, samples: int = 10) -> pd.DataFrame:
    """Get timing to compute the state vs. the depth of the Ansatz."""

    avg_times = []
    std_times = []
    depths = list(range(1, max_depth))
    for depth in depths:
        times = []
        for _ in range(samples):
            indices = sample(range(len(pool.operators)), depth)
            coeffs = np.random.rand(len(indices))
            start_time = perf_counter_ns()
            _ = adapt.compute_state(indices=indices, coefficients=coeffs)
            end_time = perf_counter_ns()
            elapsed = abs(end_time - start_time)
            times.append(elapsed)
        avg_times.append(np.average(times))
        std_times.append(np.std(times))
    output_dict = {
        "depths": np.array(depths),
        "avg_times": np.array(avg_times),
        "std_times": np.array(std_times)
    }
    df = pd.DataFrame.from_dict(output_dict, orient='columns')
    df.set_index("depths", inplace=True)
    return df


if __name__ == "__main__":
    N = 4
    chi = 15
    num_iter = 10

    r = 1.5
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
    mol = run_pyscf(mol, run_fci=False, run_ccsd=True, run_scf=True)  # CCSD doesn't work here?

    pool = GSD(mol)

    la_adapt = LinAlgAdapt(
        pool=pool,
        molecule=mol,
        max_adapt_iter=num_iter + 1,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.1,
        skip_converged_rename=True,
        mpo_filename=f"hchain_mpo_N{N}_chi1000000.pkl"
    )
    la_adapt.initialize()

    dfs = []
    df_la = test_adapt_timing(la_adapt, max_depth=MAX_DEPTH)
    df_la["chi"] = 0
    df_la["method"] = "LinAlg"
    dfs.append(df_la)

    for chi in [15, 25, 30]:
        tn_adapt = TensorNetAdapt(
            pool=pool,
            molecule=mol,
            max_adapt_iter=num_iter + 1,
            recycle_hessian=True,
            tetris=True,
            verbose=True,
            threshold=0.1,
            max_mpo_bond=MAX_MPO_BOND,
            max_mps_bond=chi,
            skip_converged_rename=True,
        )
        tn_adapt.initialize()

        df_tn = test_adapt_timing(tn_adapt, max_depth=MAX_DEPTH)
        df_tn["chi"] = chi 
        df_tn["method"] = "TensorNet"
        dfs.append(df_tn)

    total_df = pd.concat(dfs)
    total_df.to_csv("exponential_timing.csv")
