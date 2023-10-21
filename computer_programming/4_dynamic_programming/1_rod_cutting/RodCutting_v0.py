import numpy as np

## Some test cases

p_test10 = np.array([1, 5, 8, 9, 10, 17, 17, 20, 24, 30])
p_test4 = p_test10[:4]
p_test7 = p_test10[:7]

opt_r_test10 = 30.0
opt_r_test4 = 10.0
opt_r_test7 = 18.0

opt_cuts_test10 = [10]
opt_cuts_test4 = [2, 2]
opt_cuts_test7 = [1, 6]

## test functions


## this one tests only the revenue
def test_opt_r(f, p, opt_r):
    r = f(p)
    assert abs(r - opt_r) < 1e-12
    print("test_opt_r ok")


## this one tests both the revenue and the cuts
def test_opt_rcuts(f, p, opt_r, opt_cuts):
    r, cuts = f(p)
    assert abs(r - opt_r) < 1e-12
    assert sorted(cuts) == sorted(opt_cuts)
    print("test_opt_rcuts ok")


## recursive version, only revenue
# p carries info on the prices of piece by length and also the maximum length we are using
def opt_cut_rec1(p):
    if not (isinstance(p, np.ndarray) and p.ndim == 1):
        raise Exception("p must be a 1-d array")

    # length of p represents the length of the rod
    j = len(p)

    if j == 0:
        return 0.0

    j = len(p)

    count[0] += 1

    best_r = -np.infty
    for i in range(1, j + 1):
        r = p[i - 1] + opt_cut_rec1(p[: j - i])  # revenue for this cut

        if r > best_r:
            best_r = r

    return best_r


# you have to do this because if you iniitalize a int, the function will do a += 1
# when doing += 1 it reassings the variable. now a is a new variable within the scope of a function
# that is shadowing the global a variabel. so the value of the var in the global scope is not chaning
# it's changing only within the scope of the funciton and will reinitialized on every recursion step
count = [0]
## uncomment these when the function is written
# test_opt_r(opt_cut_rec1, p_test4, opt_r_test4)
# print(f"time of recursive calls {count}")
# count = [0]
# test_opt_r(opt_cut_rec1, p_test7, opt_r_test7)
# print(f"time of recursive calls {count}")
# count = [0]
# test_opt_r(opt_cut_rec1, p_test10, opt_r_test10)
# print(f"time of recursive calls {count}")
# count = [0]


## recursive version with memoization, only revenue
def opt_cut_rec1_memo(p, memo=None):
    if not (isinstance(p, np.ndarray) and p.ndim == 1):
        raise Exception("p must be a 1-d array")

    if memo is None:
        # if is none it means it's the first call
        memo = dict()

    # length of p represents the length of the rod
    j = len(p)

    if j == 0:
        return 0.0

    j = len(p)

    count[0] += 1

    best_r = -np.infty
    for i in range(1, j + 1):
        # check whether the subproblem j-i has already been solved
        if j - i in memo:
            cost_ji = memo[j - i]
        else:
            cost_ji = opt_cut_rec1_memo(p[: j - i], memo)
            memo[j - i] = cost_ji

        r = p[i - 1] + cost_ji  # revenue for this cut

        if r > best_r:
            best_r = r

    return best_r


## uncomment these when the function is written
test_opt_r(opt_cut_rec1_memo, p_test4, opt_r_test4)
test_opt_r(opt_cut_rec1_memo, p_test7, opt_r_test7)
test_opt_r(opt_cut_rec1_memo, p_test10, opt_r_test10)


## recursive version with memoization, both revenue and cuts
# why are we not initializing memo as dict()? the empty ditioanry is created when the function is initiazlied by the interpreter
# the ref is stored and if you try to reuse the function with different prices the dictionary will contain calculations
# from previous runs
# you should think carefully about giving a default value which is a mutable object, avoid mutable objects as default args
def opt_cut_rec2_memo(p, memo=None):
    if not (isinstance(p, np.ndarray) and p.ndim == 1):
        raise Exception("p must be a 1-d array")

    if memo is None:
        # if is none it means it's the first call
        memo = dict()

    # length of p represents the length of the rod
    j = len(p)

    if j == 0:
        return 0.0, []

    j = len(p)

    best_r = -np.infty
    best_cuts = None

    for i in range(1, j + 1):
        # check whether the subproblem j-i has already been solved
        if j - i in memo:
            cost_ji, cuts_ji = memo[j - i]
        else:
            cost_ji, cuts_ji = opt_cut_rec2_memo(p[: j - i], memo)
            memo[j - i] = cost_ji, cuts_ji

        r = p[i - 1] + cost_ji  # revenue for this cut

        if r > best_r:
            best_r = r
            best_cuts = [i] + cuts_ji
    return best_r, best_cuts


## uncomment these when the function is written
# test_opt_rcuts(opt_cut_rec2_memo, p_test4, opt_r_test4, opt_cuts_test4)
# test_opt_rcuts(opt_cut_rec2_memo, p_test7, opt_r_test7, opt_cuts_test7)
# test_opt_rcuts(opt_cut_rec2_memo, p_test10, opt_r_test10, opt_cuts_test10)


## bottom-up dynamic programming version
## at first only write the revenue computation (forward pass)
## then also return cuts (backward pass)
def opt_cut(p):
    if not (isinstance(p, np.ndarray) and p.ndim == 1):
        raise Exception("p must be a 1-d array")

    n = len(p)
    r = np.zeros(n + 1)
    whence = np.zeros(n + 1, dtype=int)  # array same shape as r, but contains ints
    whence[0] = -1
    # FORWARD PASS
    # Loop over the subproblems
    for j in range(1, n + 1):
        best_r = -np.infty
        best_i = None

        for i in range(1, j + 1):
            r_i = p[i - 1] + r[j - i]  # revenue for this cut

            if r_i > best_r:
                best_r = r_i
                best_i = i
        r[j] = best_r
        whence[j] = best_i

    # BACKWARD PASS
    length = n
    cuts = []
    while length > 0:
        cut = whence[length]
        length -= cut
        cuts.append(cut)

    return best_r, cuts


## uncomment these when the FIRST VERSION of the function is written
## (when the final version is written, comment it again, or adjust it...)
# test_opt_r(opt_cut, p_test4, opt_r_test4)
# test_opt_r(opt_cut, p_test7, opt_r_test7)
# test_opt_r(opt_cut, p_test10, opt_r_test10)

## uncomment these when the function is written
test_opt_rcuts(opt_cut, p_test4, opt_r_test4, opt_cuts_test4)
test_opt_rcuts(opt_cut, p_test7, opt_r_test7, opt_cuts_test7)
test_opt_rcuts(opt_cut, p_test10, opt_r_test10, opt_cuts_test10)
