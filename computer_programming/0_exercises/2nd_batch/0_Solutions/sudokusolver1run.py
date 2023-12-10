import Sudoku_Solver1
import SimAnn
import numpy as np

sdk = Sudoku_Solver1.Sudoku(9)

## This one is so easy that 2 annealing steps are enough
best = SimAnn.simann(sdk, mcmc_steps=10**4, seed=58473625, debug_delta_cost=True,
                     beta0=1.0, anneal_steps=2)

np.set_printoptions(threshold=np.inf) # force numpy to print the whole table
print("final best configuration:\n", best, "\ncost=", best.cost())

sdkpzz = Sudoku_Solver1.Sudoku(best, r=0.3)

sdkpzz.showpuzzle()

bestpzz = SimAnn.simann(sdkpzz, mcmc_steps=10**4, seed=783636464,
                        beta0=0.1, beta1=3.0, anneal_steps=30)

print("final best configuration:\n", bestpzz, "\ncost=", bestpzz.cost())

if bestpzz.cost() == 0:
    print("~SOLVED!~")
