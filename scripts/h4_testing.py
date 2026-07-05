from adaptvqe.molecules import create_h4
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import PauliPool
from adaptvqe.circuits import count_gates
from adaptvqe.op_conv import get_qasm
from adaptvqe.utils import save_to_file

import numpy as np

import os

def acc_rz_counts(data, pool, fake_params=False):
    acc_counts = [0]
    ansatz_size = 0
    count = 0

    for iteration in data.evolution.its_data:
        indices = iteration.ansatz.indices
        coefficients = iteration.ansatz.coefficients
        new_indices = indices[ansatz_size:]
        new_coefficients = coefficients[ansatz_size:]
        ansatz_size += len(new_indices)

        if fake_params:
            new_coefficients = [np.random.rand() for _ in coefficients]

        new_circuit = pool.get_circuit(new_indices, new_coefficients)
        qasm_circuit = get_qasm(new_circuit)
        count += count_gates(qasm_circuit, "rz")
        acc_counts.append(count)

    return acc_counts


r = 3
molecule = create_h4(r)
pool = PauliPool(molecule)

adapt_rand = LinAlgAdapt(
    pool=pool,
    molecule=molecule,
    threshold=0.1,
    convergence_criterion="total_g_norm",
    sel_criterion="energy",
    candidates=5,
    rand_op_it_gap=1
)

adapt_rand.run()
data = adapt_rand.data

columns = os.get_terminal_size().columns
print("\n" + "─" * columns + "\n")

print("Saving to file...")
save_to_file(data, "h4_with_rand_data.pkl")
print("Done!")

qc = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
fig = qc.draw("mpl")

print("Saving Circuit...")
fig.savefig("h4_with_rand_circuit.png", bbox_inches="tight")
print("Done!")

adapt = LinAlgAdapt(
    pool=pool,
    molecule=molecule,
    threshold=0.1,
    convergence_criterion="total_g_norm",
    sel_criterion="energy",
    candidates=5
)

adapt.run()
data = adapt.data

columns = os.get_terminal_size().columns
print("\n" + "─" * columns + "\n")

print("Saving to file...")
save_to_file(data, "h4_data.pkl")
print("Done!")

qc = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
fig = qc.draw("mpl")

print("Saving Circuit...")
fig.savefig("h4_circuit.png", bbox_inches="tight")
print("Done!")