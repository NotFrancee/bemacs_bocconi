"""Testing during lectures
"""

import numpy as np

# generates samples from an array from 1 to 5 (excluded) specifying the probabilities
# # gen = np.random.choice(5, 10, p=[0.5, 0, 0.25, 0.25, 0])
# gen = np.random.choice(3, 10, p=[0.5, 0.25, 0.25])
# print(gen)

# first example from the lecture on the M.A. Algorithm


# we provide the probability distribution we want to reach
def mcmc_sampling(p, steps=1000):
    n = len(p)  # the number of states is the length of p

    # we initialize C
    # we can choose C arbitrarily
    #   (with a few catches, we have to avoid cycles, and make sure we can get everywhere)
    C = np.zeros((n, n))  # dtype is float by default

    # we fill up the possible jumps
    # we call i the starting point of the jump
    for i in range(n):
        # we take the column i of C and set the probabilities of jumping
        # we can jump to the previous and next node
        # we set 1 since in the initial config otherwise we'll have >1. we now normalize

        if i < n - 1:
            C[i + 1, i] = 1
        if i > 0:
            C[i - 1, i] = 1
            # we do that because otherwise at edges we would go out of index

    C = C / C.sum(axis=1)

    # compute A
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if C[i, j] == 0:
                continue

            A[i, j] = min(1, (C[j, i] * p[i]) / (C[i, j] * p[j]))

    # we start with a random state and move according to C and A
    s = np.random.randint(n)
    for _ in range(steps):
        # we now propose a new move. we can move to all the states where the p > 0
        new_s = np.random.choice(n, p=C[:, s])

        # decide whether or not to accept the move
        # we look at A and look at the column s at row new_s
        prob_acceptance = A[new_s, s]  # number between 0 and 1
        if np.random.random() < prob_acceptance:
            # accept the move
            s = new_s

        # run many iterations and see how many traj finished in one state, how many in another... etc
        # (visualize through an histogram)
        # is it true that the probabilities converge to the initial probabilities we specified as an input?
