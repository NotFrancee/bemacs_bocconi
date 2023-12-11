import numpy as np

def findpath(w):
    ## Argument checks
    if not (isinstance(w, np.ndarray) and w.ndim == 2 and w.size > 0): ## last check solves task 1
        raise Exception("input w must be a non-empty 2-d array")
    if np.any(w < 0): # sol. of task 1
        raise Exception("all entries of w must be non-negative")

    n, m = w.shape

    ## Allocation of the auxiliary dynamic programming structures
    c = np.zeros((n, m))                 # the costs matrix
    whence = np.zeros((n, m), dtype=int) # the back-tracking matrix

    ## Initialization (recursion base cases)
    c[:,0] = np.cumsum(w[:,0])
    c[0,:] = np.cumsum(w[0,:])

    whence[:,0] = 1    # up
    whence[0,:] = -1   # left
    whence[0,0] = -100 # sentinel value (doesn't really matter)

    ## FORWARD PASS
    for i in range(1, n):
        for j in range(1, m):
            xl = c[i, j-1]
            xu = c[i-1, j]
            if xl <= xu: # using <= gives precedence to left moves
                whence[i, j] = -1
                c[i, j] = xl + w[i, j]
            else:
                whence[i, j] = 1
                c[i, j] = xu + w[i, j]

    ## The optimal cost
    optc = c[-1, -1]

    ## BACKWARD PASS

    ## Reconstruct the optimal solution
    path = []            # will contain -1/1 (meaning left/up)
    i, j = n-1, m-1      # current position (bottom-right corner)

    while (i,j) != (0,0):        # keep going until we reach the top-left corner
        wh = whence[i, j]        # read the pointer to the previous cell
        path.append(wh)          # store the pointer in the path
        if wh == -1:             # left case
            j -= 1               #   move left -> decrease column
        else:                    # up case
            i -= 1               #   more up -> decrease row

    ## The list was reconstructed backwards, reverse it
    path.reverse()

    ## Debug code: check the length of the path
    assert len(path) == n + m - 2
    ## Debug code: check that the path has exactly m-1 minus-ones and exctly n-1 ones
    assert sum(np.array(path) == 1) == n - 1
    assert sum(np.array(path) == -1) == m - 1
    ## Debug code: check the total cost along the path
    i, j = 0, 0
    pcost = w[0, 0]
    for wh in path:
        if wh == 1:
            i += 1
        else: # wh == -1
            j += 1
        pcost += w[i, j]
    assert (i,j) == (n-1,m-1)
    assert abs(optc - pcost) < 1e-12

    ## We're done
    return optc, path


## Test functions. They are called at the end of the file.

def test1():
    ok = True
    try:
        findpath(np.array([1.0, 2.0]))
    except:
        ok = False
    if ok:
        raise Exception("expected an error for a vector")

    ok = True
    try:
        findpath(np.array([[]]))
    except:
        ok = False
    if ok:
        raise Exception("expected an error for an empty array")

    ok = True
    try:
        findpath(np.array([[1.0, 2.0],[-1.0, 0.0]]))
    except:
        ok = False
    if ok:
        raise Exception("expected an error for an array with negative elements")
    print("test1 passed")

testw1 = (np.array([
    [1.0, 2.0, 4.0, 1.0],
    [0.0, 1.0, 1.0, 2.0],
    [0.0, 7.0, 4.0, 2.0]
    ]), 7.0, [1, -1, -1, -1, 1])

testw2 = (np.array([
    [1.0, 2.0, 4.0, 1.0],
    [3.0, 1.0, 1.0, 4.0],
    [0.0, 7.0, 3.0, 2.0]
    ]), 10.0, [-1, 1, -1, 1, -1])

testw3 = (np.array([
    [1.0, 3.0, 0.0],
    [2.0, 1.0, 7.0],
    [4.0, 1.0, 3.0],
    [1.0, 4.0, 2.0]
    ]), 10.0, [1, -1, 1, -1, 1])

testw4 = (np.array([
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 1.0, 2.0, 3.0],
    [0.0, 1.0, 2.0, 0.0],
    [0.0, 0.0, 0.5, 3.0],
    [2.0, 1.0, 1.0, 0.0],
    ]), 4.5, [1, 1, 1, -1, -1, 1, -1])

testw5 = (np.ones((4,3)), 6.0, [1, 1, 1, -1, -1])

alltests = [testw1, testw2, testw3, testw4, testw5]

def test2():
    for tst in alltests:
        tw = tst[0]
        optc, path = findpath(tw[:1,:1])
        assert optc == tw[0,0]
        for i in range(1, tw.shape[0] + 1):
            optc, path = findpath(tw[:1,:i])
            assert optc == sum(tw[0,:i])
        for j in range(1, tw.shape[1] + 1):
            optc, path = findpath(tw[:j,:1])
            assert optc == sum(tw[:j,0])
    print("test2 passed")

def test3():
    for tst in alltests:
        tw = tst[0]
        optc, path = findpath(tw)
        if abs(optc - tst[1]) > 1e-12:
            raise Exception(f"wrong cost found for w={tw}, expected {tst[1]}, obtained {optc}")
    print("test3 passed")

def test4():
    for tst in alltests[:-1]:
        tw = tst[0]
        optc, path = findpath(tw)
        if path != tst[2]:
            raise Exception(f"wrong path found for w={tw}, expected {tst[2]}, obtained {path}")
    print("test4 passed")

def test5():
    for tst in alltests:
        tw = tst[0]
        optc, path = findpath(tw)
        if path != tst[2]:
            raise Exception(f"wrong path found for w={tw}, expected {tst[2]}, obtained {path}")
    print("test5 passed")

## Uncomment these to run the tests
print("running tests...")
test1()
test2()
test3()
test4()
test5()
