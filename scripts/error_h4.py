from adaptvqe.molecules import create_h4
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import PauliPool
from adaptvqe.circuits import count_gates
from adaptvqe.op_conv import get_qasm
from adaptvqe.utils import save_to_file

import numpy as np

import matplotlib.pyplot as plt

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


def rand_vqe():
    r = 1
    molecule = create_h4(r)
    pool = PauliPool(molecule)

    adapt_rand = LinAlgAdapt(
        pool=pool,
        molecule=molecule,
        threshold=0.00001,
        convergence_criterion="total_g_norm",
        sel_criterion="energy",
        candidates=5,
        rand_op_it_gap=3
    )

    adapt_rand.run()
    data_rand = adapt_rand.data

    errors_rand = data_rand.evolution.errors
    x_rand = range(1, len(errors_rand) + 1)
    
    return x_rand, errors_rand

def vqe():
    r = 1
    molecule = create_h4(r)
    pool = PauliPool(molecule)
    
    adapt = LinAlgAdapt(
        pool=pool,
        molecule=molecule,
        threshold=0.00001,
        convergence_criterion="total_g_norm",
        sel_criterion="energy",
        candidates=5
    )

    adapt.run()
    data = adapt.data

    errors = data.evolution.errors

    x_data = range(1, len(errors) + 1)
    
    return x_data, errors

x, y = vqe()
plt.plot(x, y, label="Standard Data", color="red")

for _ in range(5):
    x, y = rand_vqe()
    plt.plot(x, y, label="Randomized Data", color="blue")
    

plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.title("Impact of Periodic Operator Randomization on ADAPT-VQE Error")
plt.legend()
plt.grid(True)

plt.savefig("error_evolution.png", bbox_inches="tight", dpi=300)
plt.show()