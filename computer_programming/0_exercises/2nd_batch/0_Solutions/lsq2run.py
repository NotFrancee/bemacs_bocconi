import LatinSquare2
import SimAnn
import numpy as np

## this version can manage n==10, but n==15 for example is already problmatic
lsq = LatinSquare2.LatinSquare(10)

best = SimAnn.simann(lsq, mcmc_steps=10**4, seed=58473625,
                     beta0=1.0, anneal_steps=20)

np.set_printoptions(threshold=np.inf) # force numpy to print the whole table
print("final best configuration:\n", best, "\ncost=", best.cost())

if best.cost() == 0:
    print("FOUND A VALID ASSIGNMENT")
else:
    print("NOT SOLVED")
