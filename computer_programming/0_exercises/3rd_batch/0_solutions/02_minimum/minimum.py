## This is the solution to ex. 2

import numpy as np

## Recursive minimum, forward direction
def minimum_rec1(l):
    if len(l) == 0:
        return np.inf
    return min(l[0], minimum_rec1(l[1:]))

## Recursive minimum, backwards direction
## (more similar to dynamic programming)
def minimum_rec2(l):
    if len(l) == 0:
        return np.inf
    return min(minimum_rec2(l[:-1]), l[-1])

## Bottom-up dynamic programming minimum, basic version
## (does not keep track of the minimum location, like the recursive versions)
## This problem is so simple that we don't even need memoization.
def minimum_dp1(l):
    c = np.inf
    n = len(l)
    for i in range(n):
        new_c = l[i]
        if new_c < c:
            c = new_c
    return c

## Bottom-up dynamic programming minimum, with tracking
## (more similar to usual dyn.prog. algorithms)
## This problem is so simple that we don't even need memoization.
def minimum_dp2(l):
    c = np.inf
    k = -1
    n = len(l)
    for i in range(n):
        new_c = l[i]
        if new_c < c:
            c = new_c
            k = i
    return c, k

## Recursive minimum, backward direction, also returns the minimum location
## (mathematically identical to the bottom-up dynamic programming, but uses a
## lot more resources in terms of memory because it needs to "unwind" the whole
## list before it starts the chain of returns.)
## This problem is so simple that we don't even need memoization.
def minimum_rec3(l):
    n = len(l)
    if n == 0:
        return np.inf, -1
    m1, k1 = l[-1], n - 1
    m2, k2 = minimum_rec3(l[:-1])
    ## Notice that the use of `<` here instead of `<=` favors values with
    ## smaller indices in case of ex-aequo (just like the dyn.prog. version).
    if m1 < m2:
        return m1, k1
    else:
        return m2, k2

## Recursive minimum, forward direction, also returns the minimum location
## (slightly more complicated than the previous one since we need to offset
## the indices. Fortunately, Python allows optional arguments, otherwise we
## would have needed to use an auxiliary function with 2 arguments. Also note
## the use of `<=` instead of `<`.)
def minimum_rec4(l, offset = 0):
    n = len(l)
    if n == 0:
        return np.inf, -1
    m1, k1 = l[0], offset
    m2, k2 = minimum_rec4(l[1:], offset + 1)
    ## Notice that the use of `<=` here instead of `<` favors values with
    ## smaller indices in case of ex-aequo (just like the dyn.prog. version).
    if m1 <= m2:
        return m1, k1
    else:
        return m2, k2

## Test that all functions return the same values
def test(n = 100, seed = None):
    if seed is not None:
        np.random.seed(seed)
    l = np.random.rand(n)
    npm = np.min(l)
    print(npm)
    mr1 = minimum_rec1(l)
    print(mr1)
    mr2 = minimum_rec2(l)
    print(mr2)
    mdp1 = minimum_dp1(l)
    print(mdp1)
    mdp2 = minimum_dp2(l)
    print(mdp2)
    mr3 = minimum_rec3(l)
    print(mr3)
    mr4 = minimum_rec3(l)
    print(mr4)
    assert npm == mr1
    assert npm == mr2
    assert npm == mdp1
    assert npm == mdp2[0]
    assert npm == mr3[0]
    assert npm == mr4[0]
    assert mdp2 == mr3
    assert mdp2 == mr4
