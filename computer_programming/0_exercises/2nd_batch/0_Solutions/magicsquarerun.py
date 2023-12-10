import MagicSquare
import SimAnn
import numpy as np


msq = MagicSquare.MagicSquare(6, 150, seed=789)

best = SimAnn.simann(msq, mcmc_steps=5*10**4, seed=266,
                     beta0=1.0, beta1=2.0, anneal_steps=30)

np.set_printoptions(threshold=np.inf) # force numpy to print the whole table
print(f"final best configuration:\n{best}\ncost={best.cost()}")

if best.cost() == 0:
    print("FOUND A VALID ASSIGNMENT")
else:
    print("NOT SOLVED")
