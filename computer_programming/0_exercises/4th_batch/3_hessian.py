from typing import Callable
import numpy as np


def g(x):
    return x[0] ** 2 + 4 * x[1] ** 2 - 2 * x[0] * x[1] + 3 * x[0] - x[1] - 1


def grad_g(x):
    dg_0 = 2 * x[0] - 2 * x[1] + 3
    dg_1 = 8 * x[1] - 2 * x[0] - 1
    return np.array([dg_0, dg_1])


def grad2_g(x):
    return np.array([[2, -2], [-2, 8]])


def grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    delta: float = 1e-5,
):
    n = len(x)

    g = np.zeros(n)
    for i in range(n):
        x_old = x[i]

        x[i] += delta
        rx = f(x)

        x[i] -= 2 * delta
        lx = f(x)

        x[i] = x_old
        g[i] = (rx - lx) / (2 * delta)

    return g


def grad2(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    delta: float = 1e-5,
):
    n = len(x)

    g = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            xi_old = x[i]
            xj_old = x[j]

            if i == j:
                x[i] = xi_old + 2 * delta
            else:
                x[i] = xi_old + delta
                x[j] = xj_old + delta

            fpp = f(x)

            if i == j:
                x[i] = xi_old
            else:
                x[i] = xi_old + delta
                x[j] = xj_old - delta

            fpm = f(x)

            if i == j:
                x[i] = xi_old
            else:
                x[i] = xi_old - delta
                x[j] = xj_old + delta

            fmp = f(x)

            if i == j:
                x[i] = xi_old - 2 * delta
            else:
                x[i] = xi_old - delta
                x[j] = xj_old - delta

            fmm = f(x)

            x[i], x[j] = xi_old, xj_old

            g[i, j] = (fpp - fpm - fmp + fmm) / (4 * delta**2)

    return g


x0 = np.array([0.1234, 0.5678])
print(grad2(g, x0))
print(grad2_g(x0))
