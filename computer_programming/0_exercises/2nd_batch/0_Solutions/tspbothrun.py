import SimAnn
import TSP_both
import matplotlib.pyplot as plt

## We compare the two schemes for the moves, i.e. swap-cities vs cross-links.
## We ensure that we are trying to solve the same problem by using the
## same seed in both constructors.
## The seed of the simann function doesn't matter; however, in order to do a
## complete and fair comparison, we should try out different annealing schemes
## for both cases, and try a few seeds in each case, and do it for a few
## problems... It gets tedious, and obviously if you wanted to do things
## correctly you would automate that process as much as possible.

tspsc = TSP_both.TSP_SC(100, seed=456329)

bestsc = SimAnn.simann(tspsc, mcmc_steps = 5000, anneal_steps = 30,
                       beta0 = 1.0, beta1 = 50.0,
                       seed = 238723784)
bestsc.display()
plt.pause(2) # force pyplot to show us the result...

tspcl = TSP_both.TSP_CL(100, seed=456329)

bestcl = SimAnn.simann(tspcl, mcmc_steps = 5000, anneal_steps = 30,
                       beta0 = 1.0, beta1 = 50.0,
                       seed = 73762334)
bestcl.display()
plt.pause(2)

print("\nDONE. Results comparison:\n")
print("best result, swap-cities=", bestsc.cost(), "cross-links=", bestcl.cost())
