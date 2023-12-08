import numpy as np
import matplotlib.pyplot as plt

# This script provides an example of using a non-linear minimization method
# to perform a least-squares fit


# Suppose that we have some non-linear function that depends on a real input x
# and on some parameters a, b, c
def s(x, a, b, c):
    return a + b * np.sin(c * x)


# Now for the sake of example we generate some data. The general idea of this
# test is:
#   1) we start with some known values of the parameters a, b, c (we call
#      these the "true" values)
#   2) we pick few values of x and get the corresponding function output,
#      using the true values of a, b, c
#   3) we add some "noise" (random jitter) to the output values
#   4) this gives us some data points (an array of x and a corresponding array
#      of y)
#   5) now we want to see if we can recover the original values of a, b, c by
#      just knowing the data points x,y & the general shape of the function s.
#      At least, we wish to get something close to the true values.
#   6) we use a "least squares" cost function to estimate how far we are from
#      the true values (more on this below)
#   7) we thus minimize the cost function using either our grad_desc algorithm
#      or scpiy.optimize.minimize
#   8) this procedure will give us some "optimal" values of a, b, c. We have a
#      look at how close we got to actually guessing the true values.
#
# This mimics the situation in which we have some data points and a model that
# we want to fit, and that model has some unknown parameters that we wish to
# infer from the data.
# In a realistic setting, this can be used to predict the result over inputs
# that we have never observed. Or, if the parameters have an interpretation
# (e.g. the "rate of production" of something), we can use the data to measure
# their value. Or both.
# NOTE: this method is general but determining whether it's appropriate and
# how to actually interpret the results depends on many things (how good is
# the model,
# how good is the cost function for your particular situation, etc.).
# You need a careful statistical analysis before you can make sense of the
# outcome, which is out of scope of this course.

a_true, b_true, c_true = 0.5, 1.2, 3.5
param_true = np.array([a_true, b_true, c_true])

x_grid = np.linspace(-2, 2, 1000)
y_grid = s(x_grid, a_true, b_true, c_true)

plt.clf()
# plt.plot(x_grid, y_grid, c="blue")

n_points = 100
x_data = np.linspace(-2, 2, n_points)
y_data = s(x_data, a_true, b_true, c_true) + 0.2 * np.random.randn(n_points)
plt.plot(x_data, y_data, "o", c="black")

a_guess, b_guess, c_guess = 0.1, 2.0, 1.2
param_guess = np.array([a_guess, b_guess, c_guess])

y_guess = s(x_grid, a_guess, b_guess, c_guess)
plt.plot(x_grid, y_guess, c="red")


def discrepancy(a, b, c, x_data, y_data):
    n = len(y_data)  # noqa
    # mse = 0.
    # for i in range(n):
    #     y_pred = s(x_data[i], a, b, c)
    #     mse += (y_data[i] - y_pred)**2
    # return mse/n
    y_pred = s(x_data, a, b, c)
    return np.mean((y_data - y_pred) ** 2)


print(f"MSE guess: {discrepancy(a_guess, b_guess, c_guess, x_data, y_data)}")
print(f"MSE true: {discrepancy(a_true, b_true, c_true, x_data, y_data)}")


def MSE(param, x_data, y_data):
    return discrepancy(param[0], param[1], param[2], x_data, y_data)


from scipy.optimize import minimize  # noqa
from GradDescND import grad_desc  # noqa

param_gd, param_gd_traj, converged = grad_desc(
    lambda p: MSE(p, x_data, y_data), param_guess, alpha=0.1, max_epochs=10000
)
print(f"MSE true: {MSE(param_gd, x_data, y_data)}")
y_gd = s(x_grid, param_gd[0], param_gd[1], param_gd[2])
plt.plot(x_grid, y_gd, c="green")

res = minimize(lambda p: MSE(p, x_data, y_data), param_guess)
param_min = res.x
y_min = s(x_grid, param_min[0], param_min[1], param_min[2])
plt.plot(x_grid, y_min, c="orange")

plt.show()
