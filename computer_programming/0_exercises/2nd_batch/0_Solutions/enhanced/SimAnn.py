import numpy as np

### ENHANCED VERSION ###

## All the changes are in the `simann` function, and there are a couple
## of additional auxiliary functions.
## In `simann`, we have added:
##  1. A generic annealing protocol, passed through betalist. This substitutes
##     the `beta0`, `beta1` and `beta_steps` options.
##  2. A hook mechanism. One hook is executed every time that a move is
##     accepted. The other hook is executed at the end of every MCMC run at a
##     given beta. In both cases, if the hook returns False, the program
##     gets out of the loop.
##     The option `debug_delta_cost` was removed (it can be passed as a hook,
##     see below).
##
## Examples of how you can use the hooks:
##  1. Pass accept_hook=debug_hook to activate the debugging of the
##     compute_delta_cost method
##  2. Pass a hook that checks if cost==0, and if so returns False. So you can
##     exit as soon as you find a zero-cost configuration (e.g. in Sudoku and
##     similar problems), without waiting for the annealing to end.


## Stochastically determine whether to acccept a move according to the
## Metropolis rule (valid for symmetric proposals)
def accept(delta_c, beta):
    ## If the cost doesn't increase, we always accept
    if delta_c <= 0:
        return True
    ## If the cost increases and beta is infinite, we always reject
    if beta == np.inf:
        return False
    ## Otherwise the probability is going to be somwhere between 0 and 1
    p = np.exp(-beta * delta_c)
    ## Returns True with probability p
    return np.random.rand() < p

## A couple of standard annealing protocols
def linearT(T0=1.0, steps=11):
    Tlist = np.linspace(T0, 0.0, steps)
    beta_list = 1 / Tlist # uses broadcasting
    return beta_list

def linearbeta(beta0=1.0, beta1=10.0, steps=11):
    beta_list = np.zeros(steps)
    beta_list[:-1] = np.linspace(beta0, beta1, steps-1)
    beta_list[-1] = np.inf
    return beta_list

## An auxiliary function used for debugging (instead of the `debug_delta_cost` option).
## You can pass it as the `accept_hook` argument of `simann`.
## However, this will only check accepted moves, instead of
## checking `debug_delta_cost` each time.
def debug_hook(probl, best, beta, cost, accrate):
    # print("cost=", cost, "probl.cost()=", probl.cost())
    assert abs(cost - probl.cost()) < 1e-10

## The simulated annealing generic solver.
## Assumes that the proposals are symmetric.
## The `probl` object must implement these methods:
##    init_config()               # returns None [changes internal config]
##    cost()                      # returns a real number
##    propose_move()              # returns a (problem-dependent) move - must be symmetric!
##    compute_delta_cost(move)    # returns a real number
##    accept_move(move)           # returns None [changes internal config]
##    copy()                      # returns a new, independent opbject
## NOTE: The default beta0 and beta1 are arbitrary.
def simann(probl,
           anneal_steps = 10, mcmc_steps = 100,
           beta_list = linearbeta(),
           accept_hook = None, mc_hook = None,
           seed = None, debug_delta_cost = False):
    ## Optionally set up the random number generator state
    if seed is not None:
        np.random.seed(seed)

    # Set up the initial configuration, compute and print the initial cost
    probl.init_config()
    c = probl.cost()
    print(f"initial cost = {c}")

    ## Keep the best cost seen so far, and its associated configuration.
    best = probl.copy()
    best_c = c

    # Main loop of the annaling: Loop over the betas
    for beta in beta_list:
        ## At each beta, we want to record the acceptance rate, so we need a
        ## counter for the number of accepted moves
        accepted = 0
        # For each beta, perform a number of MCMC steps
        for t in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            ## Metropolis rule
            if accept(delta_c, beta):
                probl.accept_move(move)
                c += delta_c
                accepted += 1
                if c <= best_c:
                    best_c = c
                    best = probl.copy()
                if accept_hook is not None:
                    hret = accept_hook(probl, best, beta, c, accepted / (t+1))
                    if hret == False: # note: we need the explicit comparison here
                        break
        print(f"acc.rate={accepted/mcmc_steps} beta={beta} c={c} [best={best_c}]")
        if mc_hook is not None:
            hret = mc_hook(probl, best, beta, c, accepted / mcmc_steps)
            if hret == False:
                break

    ## Return the best instance
    print(f"final cost = {best_c}")
    return best
