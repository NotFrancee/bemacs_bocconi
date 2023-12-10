import numpy as np

## The greedy-random-search generic solver.
## The `probl` object must implement these methods:
##    init_config()               # returns None [changes internal config]
##    cost()                      # returns a real number
##    propose_move()              # returns a (problem-dependent) move
##    compute_delta_cost(move)    # returns a real number
##    accept_move(move)           # returns None [changes internal config]
##    copy()                      # returns a new, independent opbject
##    display()                   # returns None [optional, may do nothing]
def greedy(probl, repeats = 1, num_iters = 10, seed = None, debug_delta_cost = False):
    ## Optionally set up the random number generator state
    if seed is not None:
        np.random.seed(seed)
    ## Repeat the optimization from scratch a number of times.
    best_probl = None
    best_c = np.inf
    for step in range(repeats):
        probl.init_config()
        c = probl.cost()
        last_accepted_t = 0 # useful for inspection (do we have enough num_iters?)
        for t in range(num_iters):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            ## Optinal (expensive) check that `compute_delta_cost` works
            if debug_delta_cost:
                probl_copy = probl.copy()
                probl_copy.accept_move(move)
                assert abs(c + delta_c - probl_copy.cost()) < 1e-10
            if delta_c <= 0:
                probl.accept_move(move)
                c += delta_c
                last_accepted_t = t
        print(f"final cost of run {step} = {c} [obtained at t={last_accepted_t}]")
        if c < best_c:
            best_c = c
            best_probl = probl.copy()
    best_probl.display()
    print(f"best cost = {best_c}")
    return best_probl
