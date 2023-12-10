import SimAnn
import TSP_SC

## Generate a problem to solve.
tspsc = TSP_SC.TSP_SC(100, seed=456329)

## Optimize it.
best = SimAnn.simann(tspsc, mcmc_steps = 5000, anneal_steps = 30,
                     beta0 = 1.0, beta1 = 50.0,
                     seed = 238723784,
                     debug_delta_cost = False) # set to True to enable the check

best.display()
