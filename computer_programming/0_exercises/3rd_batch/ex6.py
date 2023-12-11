from typing import Optional
import numpy as np

tests = {
    "w0": np.array([2, 3, 4]),
    "i0": np.array([0, 2]),
    "s0": 6,
    "w1": np.array([2, 9, 4]),
    "i1": np.array([1]),
    "s1": 9,
    "w2": np.array([14, 3, 27, 4, 5, 15, 1]),
    "i2": np.array([0, 2, 5]),
    "s2": 56,
    "w3": np.array([14, 3, 27, 4, 5, 15, 11]),
    "i3": np.array([0, 2, 4, 6]),
    "s3": 57,
    "w4": np.array([]),
    "i4": np.array([]),
    "s4": 0,
    "w5": np.array([4]),
    "i5": np.array([0]),
    "s5": 4,
}


def optimal_choice(w: np.ndarray):
    if not isinstance(w, np.ndarray) or np.any(w <= 0):
        raise Exception("w must be array with real positive numbers")

    n = len(w)

    # edge case
    if n == 0:
        return 0.0, []

    # cost structure
    c = np.zeros(n)
    c[0] = w[0]

    # whence
    whence = np.zeros(n, dtype=int)
    whence[0] = 2

    # fwd pass
    for i in range(1, n):
        ci1 = c[i - 1]
        ci2 = w[i]

        # edge case for i == 1
        if i != 1:
            ci2 += c[i - 2]

        # recursive relation
        if ci1 >= ci2:
            # we don't pick i, we pick c[i-1]
            c[i] = ci1
            whence[i] = 1

        else:
            # we pick i and i - 2
            c[i] = ci2
            whence[i] = 2

    opt_cost = c[n - 1]

    # backward pass
    i = n - 1
    path = []

    while i >= 0:
        d = whence[i]

        if d == 2:
            path.append(i)

        i -= d

    path.reverse()

    return opt_cost, path


def checksolution(w, s, inds: list, gap: int = 1):
    # check sum
    assert np.sum(w[inds]) - s < 1e-5

    # check sorted
    assert np.array_equal(inds, np.sort(inds))

    # check step
    assert np.all(np.diff(inds) > gap)


def test(i: Optional[int] = None):
    if isinstance(i, int):
        w = tests[f"w{i}"]
        s = tests[f"s{i}"]

        opt_cost, path = optimal_choice(w)

        checksolution(w, s, path)

    else:
        for i in range(6):
            w = tests[f"w{i}"]
            s = tests[f"s{i}"]
            opt_cost, path = optimal_choice(w)

            checksolution(w, s, path)

    print("passed test without gap!")


test()


def optimal_choice_step(w: np.ndarray, gap: int = 1):
    if not isinstance(w, np.ndarray) or np.any(w <= 0):
        raise Exception("w must be array with real positive numbers")

    if not gap >= 0:
        raise Exception("step must be >= 0")

    n = len(w)
    step = gap + 1

    # edge case
    if n == 0:
        return 0.0, []

    # cost structure
    c = np.zeros(n)
    c[0] = w[0]

    # whence
    whence = np.zeros(n, dtype=int)
    whence[0] = step

    # fwd pass
    for i in range(1, n):
        ci1 = c[i - 1]
        ci2 = w[i]

        # edge case for i == 1
        if i >= step:
            ci2 += c[i - step]

        # recursive relation
        if ci1 >= ci2:
            # we don't pick i, we pick c[i-1] and skip
            c[i] = ci1
            whence[i] = 1

        else:
            # we pick i and i - 2
            c[i] = ci2
            whence[i] = step

    opt_cost = c[n - 1]

    # backward pass
    i = n - 1
    path = []

    while i >= 0:
        d = whence[i]

        if d == step:
            path.append(i)

        i -= d

    path.reverse()

    return opt_cost, path


def test_gap():
    w = np.array([14, 3, 27, 4, 5, 15, 1])
    s2 = 42
    inds2 = [2, 5]

    s0 = w.sum()
    inds0 = list(range(len(w)))

    gap = 2
    _, path2 = optimal_choice_step(w, gap)
    checksolution(w, s2, inds2, gap)

    gap = 0
    _, path2 = optimal_choice_step(w, gap)
    checksolution(w, s0, inds0, gap)

    print("passed test with gap!")


test_gap()
