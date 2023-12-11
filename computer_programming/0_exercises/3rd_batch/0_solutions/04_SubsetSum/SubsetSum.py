import numpy as np

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

    ## The auxiliary structures that we need for the dyn.progr. algorithm.
    ## For each value of ss and j (with 0 <= ss <= s, 0 <= j <= n):
    ##
    ##   m[ss,j] will tell us "is there a subset of `v[0:j]` that sums up to `ss`?"
    ##
    ##   if m[ss,j] is True, then whence[ss,j] will tell us what is the index of the
    ##      last element of that subset of `v[0:j]` (special case: if ss==0 it gives -1)
    ##
    ##   if m[ss,j] is False, then whence[ss,j] doesn't matter, and should be -1.

    m = np.zeros((s+1, n+1), dtype=bool)     # initialized to False
    whence = -np.ones((s+1, n+1), dtype=int) # initialized to -1 (sentinel value)

    ## Base case
    m[0, :] = True  # if the sum is 0, we can always choose the empty set

    ## FORWARD PASS
    for ss in range(1, s+1):        # iterate over the values of the partial sums
        for j in range(1, n+1):     # iterate over the size of the sub-vectors v[0:j]
            rest = ss - v[j-1]      # consider the last element of the sub-vector and
                                    #   compute how much we need to reach the sub-sum `ss`
            if rest >= 0 and m[rest, j-1]:  # if we found a match...
                m[ss, j:] = True            # ...all sub-vectors v[0:j1] with j1 >= j are good
                whence[ss, j:] = j-1        # ...and the last element of the subset is j-1
                break                       # ...and we need to look no further

    ## The bottom-right corner of the mask gives us the answer to our problem:
    ## if it is False, we're done and we just return None
    if not m[s, n]:
        return None

    ## BACKWARD PASS

    ## If the answer is True, we trace back our subset by reconstructing
    ## the indices of the original vector that we need to take
    inds = []
    ss, j = s, n               # start from the bottom-right
    while whence[ss, j] != -1: # keep going until we hit the sentiel value
        j = whence[ss, j]      # get the previous element's index
        ss -= v[j]             # reduce the sum by that element's value
        inds.append(j)         # store the index value

    inds.reverse()             # reverse the indices so that they are sorted
                               # (not strictly necessary, but nicer)

    ## debug code: we can make sure that we have made no mistakes
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
