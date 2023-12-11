import numpy as np


def ex1(n: int):
    li = np.arange(n, dtype=int)

    h1, h2 = li[: n // 2], li[n // 2 :]  # noqa

    st = np.hstack((h1, h2))

    st[-1] = 0

    print(st, li)


# ex1(10)


def ex7(n: int):
    li = np.arange(n, dtype=int)
    print(li)

    sl = li[n - 1 - (n - 1) % 2 : 0 : -2]
    print(sl)


# ex7(10)


def ex8(n):
    if not n % 2 == 1 and n >= 1:
        raise Exception("n must be odd and >= 1")

    li = np.arange(n)

    inds = [0, n // 2, n - 1]
    print(li[inds])


ex8(11)
