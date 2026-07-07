from adaptvqe.molecules import create_h4
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import PauliPool
from itertools import accumulate
import matplotlib.pyplot as plt


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
        add_random_when=3
    )

    adapt_rand.run()
    data_rand = adapt_rand.data

    errors_rand = data_rand.evolution.errors
    iter_rand = range(1, len(errors_rand) + 1)
    
    operators_rand = list(accumulate(len(sub) for sub in data_rand.evolution.nnz_g_ops))
    
    return iter_rand, errors_rand, operators_rand

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

    iteration = range(1, len(errors) + 1)
    
    operators = list(accumulate(len(sub) for sub in data.evolution.nnz_g_ops))
    
    return iteration, errors, operators

iteration, errors, operators = vqe()

data_rand = []

for _ in range(5):
    data_rand.append(rand_vqe())

plt.plot(iteration, errors, label="Standard Data", color="red")

for i, (iter_rand, errors_rand, operators_rand) in enumerate(data_rand):
    label = "Randomized Data" if i == 0 else None
    plt.plot(iter_rand, errors_rand, label=label, color="blue")

plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.title("Error over Iterations")
plt.legend()
plt.grid(True)

plt.savefig("error_evolution.png", bbox_inches="tight", dpi=300)

plt.close()

plt.plot(operators, errors, label="Standard Data", color="red")

for i, (iter_rand, errors_rand, operators_rand) in enumerate(data_rand):
    label = "Randomized Data" if i == 0 else None
    plt.plot(operators_rand, errors_rand, label=label, color="blue")
    
plt.xlabel("Num Operators")
plt.ylabel("Errors")
plt.title("Error Over Number of Operators per Iteration")
plt.legend()
plt.grid(True)

plt.savefig("error_evolution_operators.png", bbox_inches="tight", dpi=300)