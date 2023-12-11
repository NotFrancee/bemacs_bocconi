from typing import Callable, Optional
import numpy as np
from common import grad, g, norm


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


x0 = np.array([0.1234, 0.5678])

res = grad_desc(g, x0, max_epochs=100, keep_intermediate=True, verbosity=2)
print(res.xs)
