import MaxCut
import SimAnn
# import numpy as np

mxc = MaxCut.MaxCut(50, seed=678678)

best = SimAnn.simann(mxc, mcmc_steps=10**4, seed=58473625,
                     beta0=0.1, beta1=10.0, anneal_steps=20)

## Printing the solution is not very helpful here. A nicer way to display the
## result would be needed...

# np.set_printoptions(threshold=np.inf) # force numpy to print the whole table
# print("final best configuration:\n", best, "\ncost=", best.cost())
