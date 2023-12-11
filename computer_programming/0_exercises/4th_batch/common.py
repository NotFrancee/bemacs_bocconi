from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt


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


class GDResults:
    def __init__(
        self,
        x: np.ndarray,
        fval: float,
        iters: int,
        converged: bool,
        xs: np.ndarray,
    ):
        self.x = x
        self.fval = fval
        self.iters = iters
        self.converged = converged
        self.xs = xs

    def __repr__(self):
        msg = [
            f"Has converged: {self.converged}",
            f"Final x: {self.x}, f(x) = {self.fval}",
            f"Iters: {self.iters}",
        ]
        return "GD RESULTS SUMMARY" + "\n\t" + "\n\t".join(msg)


def norm(x: np.ndarray) -> float:
    return np.sqrt(np.sum(x**2))


def plot_trajectory(
    k: Callable[[np.ndarray], float],
    trajectories: list[np.ndarray],
    labels: list[str],
):
    plt.close("all")  # close the previous figures
    # fig = plt.figure()  # create a new one
    fig, ax = plt.subplots()

    x0 = np.linspace(-3.0, 3.0, 1000)  # plotting in [-3, 3]x[-3,3]
    x1 = x0
    x0, x1 = np.meshgrid(x0, x1)
    # this produces two grids, one with the x0 coordinates
    # and one with the x1 coordinates

    z = k(
        np.stack((x0, x1))
    )  # this computes a function (in this case g) over the stacked grids

    # do a contour plot
    ax.contour(x0, x1, z, 50, cmap="RdGy")

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], "-", label=labels[i])

    ax.legend()

    plt.show()


def grad_desc(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    grad_f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 0.01,
    max_epochs: int = 100,
    epsilon: float = 1e-8,
    keep_intermediate: bool = False,
    verbosity: int = 0,
):
    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)  # noqa

    x = x0.copy()

    xs = [x0] if keep_intermediate else None

    converged = False
    iters = 0

    for t in range(max_epochs):
        iters += 1
        # adaptation of alpha
        p = grad_f(x)  # this is a vector
        assert len(p) == len(x)
        x -= alpha * p

        if keep_intermediate and xs is not None:
            xs.append(x.copy())

        # verbose prints
        if verbosity > 1:
            print(f"iteration {t}/{max_epochs}")
            print(
                f"\tx: {x}",
                f"f(x): {f(x)}",
                f"gradf: {p}",
                f"norm(gradf): {norm(p):.3f}",
                sep="\n\t",
            )

        # stopping criteria
        if norm(p) < epsilon:
            converged = True
            break
    xs = np.array(xs)

    results = GDResults(x, f(x), iters, converged, xs)

    if verbosity > 1:
        print(f"converged: {converged}")
    if verbosity > 0:
        print(results)

    return results
