import numpy as np
import matplotlib.pyplot as plt

power = 3


def f(z: int):
    return z**power - 1


def df(z: int):
    return power * z ** (power - 1)


roots = [np.exp(1j * 2 * np.pi / power * k) for k in range(power)]


# newton's algorithm: you initialize a grid of points,
# so that you don't end up in a single note
def newton_update(z, alpha=1.0):
    return z - alpha * f(z) / df(z)


def fixed_points(n: int, T: int = 10, lim: float = 2.0):
    z_grid = np.zeros((n, n), dtype=complex)
    z_grid.real = np.linspace(-lim, lim, n) + np.zeros(n).reshape(n, 1)
    z_grid.imag = np.linspace(-lim, lim, n).reshape(n, 1) + np.zeros(n)

    for t in range(T):
        z_grid[:] = newton_update(z_grid, 2.1)

    dist = [np.abs(z_grid - roots[k]) for k in range(power)]
    closest = np.argmin(dist, axis=0)

    plt.clf()
    plt.imshow(closest, extent=(-lim, lim, lim, -lim))
    plt.show()
    return closest


fixed_points(1000, T=10, lim=2)
