from adaptvqe.molecules import create_h3
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
molecule = create_h3(r)
pool = PauliPool(molecule)

adapt = LinAlgAdapt(
    pool=pool,
    molecule=molecule,
    threshold=0.01,
    convergence_criterion="total_g_norm",
    sel_criterion="energy",
    tetris=True,
    candidates=5
)

adapt.run()
data = adapt.data

columns = os.get_terminal_size().columns
print("\n" + "─" * columns + "\n")

# Anzats parameters
print("Ansatz Parameters:")
print("Ansatz Coefficients:", data.result.ansatz.coefficients)
print("Ansatz Indices:", data.result.ansatz.indices)
print("Ansatz Selection Gradients:", data.result.ansatz.sel_gradients)
print("==========")

nnz_gradients = data.evolution.nnz_gradients
print("Non Zero Gradients:", nnz_gradients)

nnz_g_ops = data.evolution.nnz_g_ops
print("Non Zero Gradient Operators:", nnz_g_ops)

num_cnots = data.acc_cnot_counts(pool)[-1]
print("Number of CNOT gates:", num_cnots)

num_rz = acc_rz_counts(adapt.data, pool)[-1]
print("Number of RZ gates:", num_rz)

print("Saving to file...")
save_to_file(data, "h3_data.pkl")
print("Done!")

qc = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
print("Final ansatz circuit:\n" + str(qc))

fig = qc.draw("mpl")

print("Saving Circuit...")
fig.savefig("h3_circuit.png", bbox_inches="tight")
print("Done!")
