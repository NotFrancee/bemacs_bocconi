from typing import Callable, Optional
import numpy as np
from common import g, grad, norm, GDResults, plot_trajectory


def grad_desc(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    grad_f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 0.01,
    beta: float = 0.5,
    max_epochs: int = 100,
    epsilon: float = 1e-8,
    keep_intermediate: bool = False,
    verbosity: int = 0,
    nesterov: bool = False,
):
    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)  # noqa

    x = x0.copy()

    v = np.zeros(len(x)) if nesterov else None  # initialize v for nesterov
    p = None

    xs = [x0] if keep_intermediate else None

    converged = False
    iters = 0

    for t in range(max_epochs):
        iters += 1

        # nesterov momentum
        if nesterov:
            if v is None:
                raise Exception("v is none")

            v = v * beta - alpha * grad_f(x + beta * v)
            assert len(v) == len(x)
            x += v

        else:
            p = grad(f, x)
            x -= alpha * p

        if keep_intermediate and xs is not None:
            xs.append(x.copy())

        # stopping criteria (for nesterov and normal)
        if nesterov and v is not None:
            norm_val = norm(v)
        elif p is not None:
            norm_val = norm(p)
        else:
            raise Exception("an error occured")

        # verbose prints
        if verbosity > 1:
            print(f"iteration {t}/{max_epochs}")
            print(
                f"\tx: {x}",
                f"f(x): {f(x)}",
                f"gradf: {p if p is not None else v}",
                f"norm(gradf): {norm_val:.3f}",
                sep="\n\t",
            )

        if norm_val < epsilon:
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
ALPHA = 0.01
MAX_EPOCHS = 10_000

trajs = []

betas = np.linspace(0, 0.8, 5)
for beta in betas:
    results_nesterov = grad_desc(
        g,
        x0,
        None,
        ALPHA,
        beta,
        MAX_EPOCHS,
        keep_intermediate=True,
        nesterov=True,
    )

    trajs.append(results_nesterov.xs)

results_normal = grad_desc(
    g,
    x0,
    None,
    ALPHA,
    MAX_EPOCHS,
    keep_intermediate=True,
    nesterov=False,
)
trajs.append(results_normal.xs)

print(trajs)

labels = [f"beta: {beta:.2f}" for beta in betas] + ["no nesterov"]

plot_trajectory(g, trajs, labels)
