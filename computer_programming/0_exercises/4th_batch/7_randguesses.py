import numpy as np
import matplotlib.pyplot as plt
from common import grad_desc, GDResults


def s(x, a, b, c):
    return a + b * np.sin(c * x)


a_true, b_true, c_true = 0.5, 1.2, 3.5
param_true = np.array([a_true, b_true, c_true])
x_grid = np.linspace(-2, 2, 1000)


def gen_data(n_points):
    x = np.linspace(-2, 2, n_points)

    return x, s(
        x,
        a_true,
        b_true,
        c_true,
    ) + 0.2 * np.random.standard_normal(n_points)


def plot(x_data, y_data):
    plt.plot(x_grid, s(x_grid, a_true, b_true, c_true))

    plt.plot(x_data, y_data, "o", c="black")


n_points = 100
x_data, y_data = gen_data(n_points)

plot(x_data, y_data)

a_range = (-50, 50)
b_range = (1, 100)
c_range = (-10, 50)

trials = 100


def MSE(params, x_data, y_data):
    return np.mean((y_data - s(x_data, params[0], params[1], params[2])) ** 2)


for _ in range(trials):
    a_guess = np.random.uniform(a_range[0], a_range[1])
    b_guess = np.random.uniform(b_range[0], b_range[1])
    c_guess = np.random.uniform(c_range[0], c_range[1])

    guess = np.array([a_guess, b_guess, c_guess])

    results = grad_desc(
        lambda p: MSE(p, x_data, y_data),
        guess,
        max_epochs=10_000,
    )

    if results.converged:
        print(results)


# a_guess, b_guess, c_guess = 0.1, 2.0, 1.2
# param_guess = np.array([a_guess, b_guess, c_guess])

# y_guess = s(x_grid, a_guess, b_guess, c_guess)
# plt.plot(x_grid, y_guess, c="red")

plt.show()
