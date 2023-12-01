import numpy as np


def grad(f, x: np.ndarray, delta=1e-6):
    n = len(x)
    g = np.zeros(n)

    # repeat for all components of the input
    for i in range(n):
        # compute partial derivative

        # shift xi by delta
        x[i] += delta

        # eval f
        fp = f(x)

        # shift xi by -delta
        x[i] -= 2 * delta

        fm = f(x)

        x[i] += delta  # restore xi

        g[i] = (fp - fm) / (2 * delta)

    return g  # the vector of partial derivatives


def norm(x):
    return np.sqrt(np.sum(x**2))


# grad f now retunrs a vector of all the part derivatives
def grad_desc(
    f,
    x0: np.ndarray,
    grad_f=None,
    alpha: float = 0.01,
    max_epochs: int = 100,
    epsilon: float = 1e-8,
):
    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)

    x = x0.copy()
    xs = [x0]
    converged = False
    for t in range(max_epochs):
        # adaptation of alpha
        p = -grad_f(x)  # this is a vector now

        x[:] = x + alpha * p
        xs.append(x.copy())
        # stopping criteria

        if norm(p) < epsilon:
            converged = True
            break

    xs = np.array(xs)
    return x, xs, converged


# l = [np.random.randint(5, size=5), np.random.randint(5, size=5)]
# al = np.array(l)
# print(al)
