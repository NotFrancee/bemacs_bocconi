import numpy as np


def grad(f, x, delta=1e-6):
    n = len(x)
    g = np.zeros(n)
    # repeat for all components of the input
    for i in range(n):
        # compute the partial derivative
        # shift x[i] by delta
        x_old = x[i]
        x[i] = x_old + delta
        # evaluate f
        fp = f(x)
        # shift x[i] by -delta
        x[i] = x_old - delta
        # evaluate f
        fm = f(x)

        # restore initial x[i]
        x[i] = x_old

        g[i] = (fp - fm) / (2 * delta)

    return g  # the vector of partial derivatives


def norm(x):
    return np.sqrt(np.sum(x**2))
    # return np.sqrt(np.dot(x, x))
    # return np.sqrt(x @ x)


def grad_desc(f, x0, grad_f=None, alpha=0.01, max_epochs=100, epsilon=1e-8):
    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)  # noqa

    x = x0.copy()
    xs = [x0]
    converged = False
    for t in range(max_epochs):
        # adaptation of alpha
        p = -grad_f(x)  # this is a vector
        assert len(p) == len(x)
        x = x + alpha * p
        xs.append(x)
        # stopping criteria
        if norm(p) < epsilon:
            converged = True
            break
    xs = np.array(xs)

    return x, xs, converged
