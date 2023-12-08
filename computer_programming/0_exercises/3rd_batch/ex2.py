import numpy as np


def minimum_rec1(l: list):  # noqa
    # recursive approach
    if len(l) == 0:
        return np.inf

    if len(l) == 1:
        return l[0]

    return np.min([l[0], minimum_rec1(l[1:])])


def test(tlist: list):
    assert minimum_rec1(tlist) == np.min(tlist)
    print(f"OK: {tlist}")


test([1, 2, 3, 4, 5])
test([1, 2, 3, 4, 5, 0])
test([1, 2, 3, 4, 5, -1])
test([1, 2, 3, 4, 5, -1, -2])
test([7, 2, 5, 3, 1, 8, 9, 4, 6])
