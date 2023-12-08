import numpy as np
import matplotlib.pyplot as plt


def g(x):
    return x**4 + 3 * x**3 - x**2 + 1


def grad_g(x):
    return 4 * x**3 + 9 * x**2 - 2 * x


def grad_f(f, x, delta=1e-5):
    return (f(x + delta) - f(x - delta)) / (2 * delta)


xs = np.linspace(-5, 5, 1000)
ys = g(xs)

plt.clf()
plt.plot(xs, ys, c="blue")


def grad_desc(f, grad_f, x0, alpha=0.01, max_epochs=100):
    x = x0
    xs = np.zeros(max_epochs + 1)
    xs[0] = x0
    for t in range(max_epochs):
        # adaptation of alpha

        p = -grad_f(x)
        x = x + alpha * p
        xs[t + 1] = x
        # stopping criteria

    return x, xs
