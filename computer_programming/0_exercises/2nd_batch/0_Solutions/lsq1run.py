import LatinSquare1
import SimAnn
import numpy as np

## for this version n==10 is typically already too large for finding a solution
## quickly
lsq = LatinSquare1.LatinSquare(10, seed=678876)

best = SimAnn.simann(lsq, mcmc_steps=10**4, seed=58473625,
                     beta0=1.0, anneal_steps=20)

np.set_printoptions(threshold=np.inf) # force numpy to print the whole table
print("final best configuration:\n", best, "\ncost=", best.cost())

if best.cost() == 0:
    print("FOUND A VALID ASSIGNMENT")
else:
    print("NOT SOLVED")
