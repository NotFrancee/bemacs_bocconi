import Sudoku
import SimAnn
import numpy as np

## n == 16 is solved pretty easily, larger n requires slower annlealing
sdk = Sudoku.Sudoku(16)

best = SimAnn.simann(sdk, mcmc_steps=10**4, seed=58473625,
                     beta0=1.0, anneal_steps=20)

np.set_printoptions(threshold=np.inf) # force numpy to print the whole table
print("final best configuration:\n", best, "\ncost=", best.cost())
