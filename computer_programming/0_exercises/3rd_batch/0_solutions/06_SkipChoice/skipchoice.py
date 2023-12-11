import numpy as np

## This is the generalized solution that uses the `gap` variable.
## If you just substitute `1` instead of `gap` within the code
## it reduces to the original code described in the text
def skipchoice(w, gap = 1):
    ## Argument checks
    if not (isinstance(w, np.ndarray) and w.ndim == 1):
        raise Exception("w must be an array")
    if not (isinstance(gap, int) and gap >= 0):
        raise Exception("gap must be a non-negative int")

    n = len(w)

    ## Conrner case: empty input
    if n == 0:
        return 0.0, []

    if np.any(w <= 0):
        raise Exception("w entries must be positive")

    ## Allocation of the auxiliary dynamic programming structures
    c = np.zeros(n)
    whence = np.zeros(n, dtype=int)

    step = 1 + gap # make the code slightly simpler

    ## Initialization (recursion base case)
    c[0] = w[0]
    whence[0] = step # we always start with the "pick" case

    ## FORWARD PASS
    for i in range(1,n):
        xs = c[i-1]                                     # "skip" case
        xp = w[i] + (c[i-step] if i-step >= 0 else 0.0) # "pick" case
        if xp >= xs:
            c[i] = xp
            whence[i] = step
        else:
            c[i] = xs
            whence[i] = 1

    ## The optimal value
    opts = c[-1]

    ## BACKWARD PASS
    ## Reconstruct the optimal solution
    inds = []
    i = n-1
    while i >= 0:
        d = whence[i]
        if d == step: # "pick" case
            inds.append(i)
        i -= d
    inds.reverse()

    return opts, inds


def checksolution(w, s, inds, gap = 1):
    ## check that `inds` is consistent with `w` and `s`
    ## NOTE: in order to be safe against floating-point approximation, one
    ##       could use approximate equality here. We can get away with
    ##       strict equality since we expect that the summation will always
    ##       be carried out in the same order, making the result predictable
    ##       all the way to the last bit.
    assert np.sum(w[inds]) == s

    ## check that `inds` satisfies the constraints
    ## (two possible versions; the second one is more complicated because np.min
    ##  cannot be called on empty arrays or lists)
    assert np.all(np.diff(inds) > gap)
    assert len(inds) <= 1 or np.min(np.diff(inds)) > gap

    return True


## A few test cases. Each one is a tuple in which the
## first element is the input array w, and the other two
## are the return values of skipchoice: the optimal value s
## and the list of indices corresponding to that value.

wtest0 = (
    np.array([]),
    0, [])

wtest1 = (
    np.array([1]),
    1, [0])

wtest2 = (
    np.array([2, 1]),
    2, [0])

wtest3 = (
    np.array([1, 2]),
    2, [1])

wtest4 = (
    np.array([1, 2, 1, 3]),
    5, [1, 3])

wtest5 = (
    np.array([14, 3, 27, 4, 5, 15, 1]),
    56, [0, 2, 5])

wtest6 = (
    np.array([14, 3, 27, 4, 5, 15, 11]),
    57, [0, 2, 4, 6])

wtest7 = (
    np.array([3, 15, 17, 23, 11, 3, 4, 5, 17, 23, 34, 17, 18, 14, 12, 15]),
    126, [1, 3, 6, 8, 10, 12, 15])

alltests = [wtest0, wtest1, wtest2, wtest3, wtest4, wtest5, wtest6, wtest7]

## A few WRONG test cases. These have the same structure
## as the previous ones, but they are incorrect and thus
## should be caught by `checksolution`

errtest0 = (
    np.array([]),
    1, [])

errtest1 = (
    np.array([1]),
    2, [0])

errtest2 = (
    np.array([2, 1]),
    2, [1])

errtest3 = (
    np.array([1, 2]),
    2, [0])

errtest4 = (
    np.array([10, 3, 20, 4]),
    37, [0, 2])

errtest5 = (
    np.array([10, 3, 20, 4]),
    33, [0, 1, 2])

errtest6 = (
    np.array([10, 3, 20, 4]),
    30, [2, 0])

errtest7 = (
    np.array([10, 3, 20, 4]),
    30, [1, 1])

allerrtests = [errtest0, errtest1, errtest2, errtest3, errtest4, errtest5, errtest6, errtest7]

## A couple more test cases for different gaps.
## This time there is an extra item in the tuple,
## the gap.

wtestgap1 = (
    np.array([14, 3, 27, 4, 5, 15, 11]), 2,
    42, [2, 5])

wtestgap2 = (
    np.array([14, 3, 27, 4, 5, 15, 11]), 0,
    79, [0, 1, 2, 3, 4, 5, 6])

wtestgap3 = (
    np.array([1, 30, 4, 4, 5, 50, 1]), 2,
    80, [1, 5])

wtestgap4 = (
    np.array([1, 30, 4, 4, 5, 50, 1]), 1000,
    50, [5])

alltestgaps = [wtestgap1, wtestgap2, wtestgap3, wtestgap4]

## A few more WRONG test cases, this time with gaps.

errgaptest0 = (
    np.array([]), 0,
    1, [])

errgaptest1 = (
    np.array([1]), 2,
    2, [0])

errgaptest2 = (
    np.array([2, 1]), 3,
    2, [1])

errgaptest3 = (
    np.array([1, 2]), 3,
    2, [0])

errgaptest4 = (
    np.array([10, 3, 20, 4]), 2,
    30, [0, 2])

errgaptest5 = (
    np.array([10, 3, 20, 4]), 3,
    14, [0, 3])

allerrgaptests = [errgaptest0, errgaptest1, errgaptest2, errgaptest3, errgaptest4, errgaptest5]

## Test functions. They are called at the end

def test0():
    for wt in [alltests[0]] + alltests[2:]:
        w, strue, indstrue = wt
        s, inds = skipchoice(w)
        assert s == strue
    print("test0 passed")

def test1():
    for wt in alltests:
        w, strue, indstrue = wt
        s, inds = skipchoice(w)
        assert s == strue
    print("test1 passed")

def test2():
    for wt in alltests:
        w, strue, indstrue = wt
        s, inds = skipchoice(w)
        assert s == strue
        assert inds == indstrue
    print("test2 passed")

def test3():
    for wt in alltests:
        w, strue, indstrue = wt
        checksolution(w, strue, indstrue)
    for errt in allerrtests:
        w, strue, indstrue = errt
        failedonerror = False
        try:
            checksolution(w, strue, indstrue)
        except:
            failedonerror = True
        assert failedonerror
    print("test3 passed")

def test4a():
    for wt in alltestgaps:
        w, gap, strue, indstrue = wt
        s, inds = skipchoice(w, gap)
        assert s == strue
        assert inds == indstrue
    print("test4a passed")

def test4b():
    for wt in alltestgaps:
        w, gap, strue, indstrue = wt
        checksolution(w, strue, indstrue, gap)
    for errt in allerrgaptests:
        w, gap, strue, indstrue = errt
        failedonerror = False
        try:
            checksolution(w, strue, indstrue, gap)
        except:
            failedonerror = True
        assert failedonerror
    print("test4b passed")


test0()
test1()
test2()
test3()
test4a()   # tests skipchoice
test4b()   # tests checksolution
