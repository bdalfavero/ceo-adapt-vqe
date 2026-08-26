from time import perf_counter_ns
import pickle
import numpy as np
import pandas as pd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.pools import DVE_CEO, GSD, PairedGSD

MAX_MPO_BOND = 200

if __name__ == "__main__":
    N = 16
    chi = 5
    num_iter = 5

    r = 1.5
    geometry = [['H', [0, 0, i * r]] for i in range(N)]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    mol = MolecularData(geometry, basis, multiplicity, charge, description=f'H{N}')
    mol = run_pyscf(mol, run_fci=False, run_ccsd=True, run_scf=True)  # CCSD doesn't work here?
    hf_energy = mol.hf_energy
    exact_energy = mol.ccsd_energy
    print(f"hf_energy = {hf_energy}")
    print(f"exact_energy = {exact_energy}")

    start_time = perf_counter_ns()
    # pool = DVE_CEO(mol)
    pool = PairedGSD(mol)
    end_time = perf_counter_ns()
    elapsed = abs(end_time - start_time)
    print(f"Built pool in {elapsed:5.4e} ns.")

    my_adapt = TensorNetAdapt(
        pool=pool,
        molecule=mol,
        max_adapt_iter=num_iter + 1,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.1,
        max_mpo_bond=MAX_MPO_BOND,
        max_mps_bond=chi,
        skip_converged_rename=True
    )
    print("Initializing...")
    start_time = perf_counter_ns()
    my_adapt.initialize()
    end_time = perf_counter_ns()
    elapsed = abs(end_time - start_time)
    print(f"Initialized in {elapsed:5.4e} ns.")

    energies = []
    times = []
    for _ in range(num_iter):
        start_time = perf_counter_ns()
        my_adapt.run_iteration()
        end_time = perf_counter_ns()
        elapsed_time = float(abs(end_time - start_time))
        energies.append(my_adapt.energy)
        times.append(elapsed_time)
    
    output_dict = {
        "hf_energy": hf_energy,
        "exact_energy": exact_energy,
        "energies": energies,
        "times": times
    }

    with open("large_hchain_results.pkl", "wb") as f:
        pickle.dump(output_dict, f)