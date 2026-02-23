import json
from time import perf_counter_ns

from pyscf import ao2mo
from pyscf.tools.fcidump import from_scf, read

from adaptvqe.utils import hamiltonian_from_fcidump
from adaptvqe.hamiltonians import FermionicHamiltonian
from adaptvqe.pools import CEO, PairedDoubleCEO, OccupiedVirtualCEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, TensorNetAdapt

max_mpo_bond = 200
max_mps_bond = 30

lih_fname = "data/atp_0_be2_f4.fcidump"

print("Building Hamiltonians.")
h_fcidump, norb, nelec = hamiltonian_from_fcidump(lih_fname)
h = FermionicHamiltonian(
    h_fcidump, "atp", nelec, diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=max_mps_bond
)
nq = h.n
print(f"Hamiltonian has {nq} qubits.")

# Since we have alpha and beta e-'s, an n-qubit Hamiltonian as n/2 spatial orbitals.
assert h.n % 2 == 0
num_orbitals = h.n // 2

# TODO Get the exact energy from PySCF.
exact_energy = h.ground_energy
print(f"Got exact energy {exact_energy}.")

# Get one- and two-body integrals.
fci_read = read(lih_fname)
h1 = fci_read["H1"]
h2_packed = fci_read["H2"]
h2 = ao2mo.restore(1, h2_packed, num_orbitals)  # (norb,norb,norb,norb)
n_electrons = fci_read["NELEC"]
ecore = fci_read["ECORE"]
spin = 0 # TODO How would I know from the FCIDUMP alone?
num_elec_a = (n_electrons + spin) // 2
num_elec_b = (n_electrons - spin) // 2
nelec = (num_elec_a, num_elec_b)

start_time = perf_counter_ns()
ov_pool = OccupiedVirtualCEO(h, n_occ=nelec)
end_time = perf_counter_ns()
elapsed_time_ov_pool = abs(end_time - start_time)

print("Running.")
ov_adapt = TensorNetAdapt(
    pool=ov_pool,
    # molecule=mol,
    custom_hamiltonian=h,
    max_adapt_iter=1,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
    max_mpo_bond=max_mpo_bond,
    max_mps_bond=max_mps_bond
)

times = []
for i in range(10):
    start_time = perf_counter_ns()
    ov_adapt.run_iteration()
    ceo_energy = ov_adapt.energy
    end_time = perf_counter_ns()
    elapsed_time_ov_adapt = abs(end_time - start_time)
    times.append(elapsed_time_ov_adapt)

pool_size = len(ov_pool.operators)

fci_err = abs(h.ground_energy - ov_adapt.energy)

output_dict = {
    "errors": fci_err,
    "energies": {
        "exact": h.ground_energy,
        "ceo": ov_adapt.energy
    },
    "times": {
        "build": elapsed_time_ov_pool,
        "run": times
    }
}
with open("data/fciump_results.json", "w") as f:
    json.dump(output_dict, f)