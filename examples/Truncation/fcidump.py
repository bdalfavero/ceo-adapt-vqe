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

print("Building pools.")
start_time = perf_counter_ns()
ceo_pool = CEO(h)
end_time = perf_counter_ns()
elapsed_time_ceo_pool = abs(end_time - start_time)

start_time = perf_counter_ns()
ov_pool = OccupiedVirtualCEO(h, n_occ=nelec)
end_time = perf_counter_ns()
elapsed_time_ov_pool = abs(end_time - start_time)

# print("CEO pool:")
# for op in ceo_pool.operators:
#     print(op.operator, "\n")
# print("OV CEO pool:")
# for op in ov_pool.operators:
#     print(op.operator, "\n")

print("Running.")
start_time = perf_counter_ns()
ceo_adapt = TensorNetAdapt(
    pool=ceo_pool,
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

ceo_adapt.run()
ceo_energy = ceo_adapt.energy
end_time = perf_counter_ns()
elapsed_time_ceo_adapt = abs(end_time - start_time)

start_time = perf_counter_ns()
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

ov_adapt.run()
ov_energy = ov_adapt.energy
end_time = perf_counter_ns()
elapsed_time_ov_adapt = abs(end_time - start_time)

print("Sizes of pools:")
print(len(ceo_pool.operators), len(ov_pool.operators))

print("Times for regular CEO pool:")
print(f"Building pool: {elapsed_time_ceo_pool:5.4e}\nRun: {elapsed_time_ceo_adapt:5.4e}")
print("Times for paired CEO pool:")
print(f"Building pool: {elapsed_time_ov_pool:5.4e}\nRun: {elapsed_time_ov_adapt:5.4e}")

err = abs(ov_energy - ceo_energy)
print(f"Energy error: {err:5.4e}")
fci_err = abs(h.ground_energy - ov_energy)
print(f"Error wrt FCI: {fci_err:5.4e}")

output_dict = {
    "errors": {
        "ov_ceo": err,
        "ov_fci": fci_err
    },
    "energies": {
        "exact": h.ground_energy,
        "ceo": ceo_adapt.energy,
        "ov": ov_adapt.energy
    },
    "times": {
        "ceo": {
            "build": elapsed_time_ceo_pool,
            "run": elapsed_time_ceo_adapt
        },
        "ov": {
            "build": elapsed_time_ov_pool,
            "run": elapsed_time_ov_adapt
        }
    }
}
with open("data/fciump_results.json", "w") as f:
    json.dump(output_dict, f)