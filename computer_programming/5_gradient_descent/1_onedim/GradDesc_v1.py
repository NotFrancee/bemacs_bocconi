import numpy as np
import matplotlib.pyplot as plt


def g(x):
    return x**4 + 3 * x**3 - x**2 - 10 * x + 1


def grad_g(x):
    return 4 * x**3 + 9 * x**2 - 2 * x


def h(x):
    return np.log(1 + np.exp(x))


def k(x):
    return h(g(x))


def grad(f, x, delta=1e-5):
    # return (f(x+delta) - f(x)) / delta
    return (f(x + delta) - f(x - delta)) / (2 * delta)


xs = np.linspace(-4, 4, 10000)
ys = g(xs)

plt.clf()
plt.plot(xs, ys, c="blue")


def grad_desc(f, x0, grad_f=None, alpha=0.01, max_epochs=100, epsilon=1e-8):
    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)  # noqa

    x = x0
    xs = [x0]
    converged = False
    for t in range(max_epochs):
        # adaptation of alpha
        p = -grad_f(x)
        x = x + alpha * p
        xs.append(x)
        # stopping criteria
        if abs(p) < epsilon:
            converged = True
            break
    xs = np.array(xs)

    return x, xs, converged
