import Sudoku
import SimAnn as SA

## A hook for early exit in case a solution is found
## When this returns False, `simann` will break out of its
## loop, thus saving time.
def checksolved(sdk, best, beta, cost, accrate):
    ## Want to also debug your code? Just add this here:
    # SA.debug_hook(sdk, best, beta, cost, accrate)
    return cost != 0

sdk = Sudoku.Sudoku(9)

best = SA.simann(sdk, mcmc_steps=10**4, seed=58473625,
                 beta_list=SA.linearbeta(beta0=1.0, beta1=100.0, steps=10),
                 accept_hook=checksolved, mc_hook=checksolved)

print("final best configuration:\n", best, "\ncost=", best.cost())

pzz = best.gen_puzzle(0.3)
sdkpzz = Sudoku.Sudoku(pzz)

sdkpzz.showpuzzle()

bestpzz = SA.simann(sdkpzz, mcmc_steps=10**4, seed=783636464,
                    beta_list=SA.linearbeta(beta0=0.1, beta1=3.0, steps=30),
                    accept_hook=checksolved, mc_hook=checksolved)

print("final best configuration:\n", bestpzz, "\ncost=", bestpzz.cost())

if bestpzz.cost() == 0:
    print("~SOLVED!~")
