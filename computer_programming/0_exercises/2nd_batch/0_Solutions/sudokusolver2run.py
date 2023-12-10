import Sudoku_Solver2
import SimAnn
import numpy as np

sdk = Sudoku_Solver2.Sudoku(9)

best = SimAnn.simann(sdk, mcmc_steps=10**4, seed=58473625,
                     beta0=1.0, anneal_steps=2)

np.set_printoptions(threshold=np.inf) # force numpy to print the whole table
print("final best configuration:\n", best, "\ncost=", best.cost())

pzz = best.gen_puzzle(0.3)

sdkpzz = Sudoku_Solver2.Sudoku(pzz)

sdkpzz.showpuzzle()

bestpzz = SimAnn.simann(sdkpzz, mcmc_steps=10**4, seed=783636464,
                        beta0=0.1, beta1=3.0, anneal_steps=30)

print("final best configuration:\n", bestpzz, "\ncost=", bestpzz.cost())

if bestpzz.cost() == 0:
    print("~SOLVED!~")
