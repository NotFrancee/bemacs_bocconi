import numpy as np
from scipy.special import binom


class G:
    def lin(self, x: float):
        return x**4 + 4 * x**3 + x**2 - 10 * x + 1

    def grad1(self, x: float):
        return 4 * x**3 + 12 * x**2 + 2 * x - 10

    def grad2(self, x: float):
        return 12 * x**2 + 24 * x + 2

    def grad3(self, x: float):
        return 24 * x + 24

    def grad4(self, x: float):
        return 24

    def test_err(self, grad, n: int, x: float):
        real = 0
        match n:
            case 2:
                real = self.grad2(x)
            case 3:
                real = self.grad3(x)
            case 4:
                real = self.grad4(x)

        print(np.arange(1, 10))
        deltas = np.power(0.1, np.arange(1, 10))

        err = lambda dd: abs(real - grad(n, self.lin, x, dd))  # noqa

        errors = err(deltas)
        print(errors)


def grad(f, x, delta=1e-5):
    return (f(x + delta) - f(x - delta)) / (2 * delta)


def gradn(n, f, x, delta):
    if n == 0:
        return f(x)

    else:
        return (
            gradn(n - 1, f, x + delta, delta)
            - gradn(
                n - 1,
                f,
                x - delta,
                delta,
            )
        ) / (2 * delta)


def gradn_impr(n, f, x, delta):
    r = np.arange(n + 1)
    delta_coeffs = np.arange(n, -n - 1, -2)
    coeffs = binom(n, r)
    signs = (-1) ** r

    return np.sum(signs * coeffs * f(x + delta_coeffs * delta)) / (
        2**n * delta**n
    )  # noqa


g = G()
n = 3
x0 = 0.1234
delta = 1e-5

# r = gradn(n, g.lin, x0, delta)
# rimpr = gradn_impr(n, g.lin, x0, delta)

# print(r)
# print(rimpr)
# print(g.grad3(0.1234))
g.test_err(gradn_impr, n, x0)
