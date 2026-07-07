from adaptvqe.molecules import create_h4
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import PauliPool
from adaptvqe.utils import save_to_file


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
    add_random_when=1
)

adapt_rand.run()
data = adapt_rand.data

print("Saving to file...")
save_to_file(data, "h4_with_rand_data.pkl")
print("Done!")