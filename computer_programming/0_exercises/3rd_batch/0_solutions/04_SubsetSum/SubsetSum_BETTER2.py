# ALTERNATIVE (FURTHER IMPROVED) VERSION
#
# This is an alternative version of the same dynamic programming algorithm
# of "SubsetSum.py", but it performs the same computations using
# less memory [a single O(s) data structure instead of two O(s*n) ones]
# and slightly fewer operations. The "cost" is that it is probably a little bit
# harder to understand (and less similar to other dyn.progr. codes that we have
# seen).
# It is included for demonstration purposes.

import numpy as np

## Auxiliary function: returns the permutation of the indices that sorts a
## given vector `v`.
## For example, if `v = [5,2,8,1]`, returns `[3,1,0,2]` because the smallest
## element of `v` is at index 3, the next one is at index 1 etc.
def sortperm(v):
    return np.array(sorted(range(len(v)), key = lambda j: v[j]), dtype=int)

## The dynamic programming function that determines whether
## there exists a subset of the elements of `v` that sums up to `s`,
## and returns the subset's indices if there is (or None if there isn't).
def get_subset(v, s):
    ## Argument checks
    if not isinstance(v, np.ndarray) or v.ndim != 1:
        raise Exception("input v must be a 1-d array")
    if not isinstance(s, int) or s < 0:
        raise Exception("input s must be a non-negative integer")

    ## Check that v[j] > 0 for all j
    if not (v > 0).all(): # or: if np.any(v <= 0):
        raise Exception("all elements of v must be strictly positive")

    n = len(v)

    p = sortperm(v) # find the permutation of the indices that sorts `v`
                    # (we need to keep this for later)
    sv = v[p]       # this is basically the same as `sorted(v)`
                    # we work on this sorted version of `v` because it allows us
                    # to spare some computations (a lot of computations, possibly)

    ## The auxiliary structure that we need for the dyn.progr. algorithm.
    ## For each value of ss, with 0 <= ss <= s
    ##
    ##   w[ss] will tell us "what is the minimum j such that is there a subset
    ##                       of `sv[0:j]` that sums up to `ss`?"
    ##
    ##   if no valid j exists, we set it to n+1 by convention (a sentinel value)

    w = np.full(s+1, n+1) # array of length s+1 initialized to n+1 (sentinel value)
    # w = (n+1) * np.ones(s+1, dtype=int) # alternative version, equivalent

    w[0] = 0 # if the sum is 0, we can always choose the empty set

    cv = sv.cumsum() # cv[j] is the maximum sum that can ve achieved with `sv[0:j+1]`
                     # therefore if we are trying to achieve some value `ss` we don't
                     # need to bother checking those `j`s for which `cv[j] < ss`.
    j0 = 0           # `j0` will be the "minimum viable j", the first one for
                     # which `cv[j] >= ss`.
                     # Note that this obviously increases as `ss` increases.

    ## FORWARD PASS
    for ss in range(1, s+1):           # iterate over the values of the partial sums
        while j0 < n and cv[j0] < ss:  # update the minimum viable candidate for j
            j0 += 1
        for j in range(j0, n):         # iterate over the elements of sv
            rest = ss - sv[j]          # consider sv[j] and how much more we need to reach the sub-sum `ss`
            if rest < 0:               # since `sv` is sorted, when the elements get too big...
                break                  # ...we can just skip the remaining ones
            if w[rest] <= j:           # if `rest` can be achieved with `j` elements or fewer...
                w[ss] = j+1            # ...then `ss` can be achieved with `j+1` elements or more...
                break                  # ...and we need to look no further

    ## if w[s] is set to the sentinel value, then we're done and we just return None
    if w[s] == n+1:
        return None

    ## BACKWARD PASS

    ## If a valid subset exists, we trace it back by reconstructing
    ## the indices of the original vector that we need to take
    inds = []
    ss = s              # start from the end
    while w[ss] != 0:   # keep going until we get an empty set
        j = w[ss]-1     # get the element's index
        ss -= sv[j]      # reduce the sum by that element's value
        inds.append(j)  # store the index

    ## debug code: we can make sure that we have made no mistakes
    assert ss == 0
    assert sv[inds].sum() == s

    ## recover the indices of the original (unsorted) vector, by going through the
    ## permutation array `p` that we had saved at the beginning.
    ## (the call to `sorted` is not strictly necessary, it just makes the result nicer)
    inds = sorted(p[inds])
    assert v[inds].sum() == s

    ## finally we return the indices
    return inds


## Test functions
## The first one just tests the arguments checks
## The rest all use an auxiliary test function called test_get_subset
## They just test some progressively more complex situations

def test0():
    ok = True
    try:
        get_subset(np.array([3, 2, 0, 4]), 5)
    except:
        ok = False
    if ok:
        raise Exception("expected an error for a vector with non-positive values")
    ok = True
    try:
        get_subset(np.array([3, 2, 1, 4]), -5)
    except:
        ok = False
    if ok:
        raise Exception("expected an error for non-positive sum")
    print("test0 passed")

def test1():
    v = np.array([])
    # With an empty vector, the only possible subset is empty and the only valid sum is 0
    test_get_subset(v, 0, True)  # should return []
    test_get_subset(v, 1, False)
    test_get_subset(v, 2, False)
    print("test1 passed")

def test2():
    v = np.array([1,2,3])
    ## all sums from 0 to 6 are possible
    for s in range(7):
        test_get_subset(v, s, True)
    ## anything beyond 6 is impossible
    for s in range(7,10):
        test_get_subset(v, s, False)
    print("test2 passed")

def test3():
    v = np.array([3,7,2,10])
    # For each value of s, test_s[s] tells us whether the problem is solvable
    # (the comments report s and the possible solutions
    test_s = [True,  # 0     []
              False, # 1
              True,  # 2     [2]
              True,  # 3     [0]
              False, # 4
              True,  # 5     [0,2]
              False, # 6
              True,  # 7     [1]
              False, # 8
              True,  # 9     [1,2]
              True,  # 10    [0,1] or [3]
              False, # 11
              True,  # 12    [0,1,2] or [2,3]
              True,  # 13    [0,3]
              False, # 14
              True,  # 15    [0,2,3]
              False, # 16
              True,  # 17    [1,3]
              False, # 18
              True,  # 19    [1,2,3]
              True,  # 20    [0,1,3]
              False, # 21
              True,  # 22    [0,1,2,3]
              False] # 23
    for (s, ok) in zip(range(len(test_s)), test_s):
        test_get_subset(v, s, ok)
    print("test3 passed")


## An auxiliary function that tests that get_subset works as expected
def test_get_subset(v, s, expected):
    call_worked = False
    valid_inds = False
    try:
        inds = get_subset(v, s)
        call_worked = True
        ok = inds is not None
        assert ok == expected
        if ok:
            s1 = v[inds].sum()
            valid_inds = True
            assert s1 == s
        else:
            assert inds == None
    except:
        if not call_worked:
            raise Exception(f"call to get_subset failed with arguments v={v} s={s}: the call produced an error")
        outstr = f"wrong result for get_subset({v}, {s}): "
        if ok != expected:
            if expected:
                assert inds is None
                outstr += f"the call returned None but a solution exists"
            else:
                assert inds is not None
                outstr += f"the call returned {inds} but None was expected"
        elif ok and not valid_inds:
            outstr += f"the call returned {inds} which appear to be invalid indices for vector v={v}"
        elif ok and v[inds].sum() != s:
            outstr += f"the call returned {inds} but the sum of the elements of v at indices {inds} does not give {s}"
        else:
            outstr += f"I CAN'T DETERMINE THE ERROR, THIS IS A BUG IN THE TESTS"
        raise Exception(outstr)

    return True


## Uncomment these to run the tests
print("running tests...")
test0()
test1()
test2()
test3()
