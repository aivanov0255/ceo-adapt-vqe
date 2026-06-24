from scipy.sparse.linalg import expm, expm_multiply
from openfermion import get_sparse_operator
from qiskit.quantum_info import Operator, process_fidelity
import numpy as np

from adaptvqe.pools import NoZPauliPool, DVG_CEO, QE, GSD, ImplementationType
from adaptvqe.molecules import create_h4

# Define test case: molecule, ansatz size, coefficient list
r = 1.5
molecule = create_h4(r)
ansatz_size = 10
coefficients = np.random.random(ansatz_size)

# Decide which pools to test
pools = [NoZPauliPool(molecule), DVG_CEO(molecule), QE(molecule), GSD(molecule)]

print(f"Fidelity between unitaries generated from matrix multiplication and from circuit,"
      f" for random {ansatz_size} element ansatz with...")

for pool in pools:

    # Generate random ansatz with elements from current pool
    indices = np.random.randint(0, pool.size - 1, ansatz_size)

    pool.imp_type = ImplementationType.SPARSE
    goal_unitary = Operator(pool.get_unitary(coefficients, indices).todense())

    # Obtain unitary from circuit implementation to compare against target obtained via matrix algebra
    qc = pool.get_circuit(indices, coefficients)
    unitary = Operator(qc)

    pf = process_fidelity(unitary, goal_unitary)
    print(f"...{pool.name}: ",pf)
    assert np.abs(1-pf)<10**-8

print("Process fidelities all confirmed to be close to unity.")
