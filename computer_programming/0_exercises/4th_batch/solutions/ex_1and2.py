# flake8: noqa

import numpy as np
import matplotlib.pyplot as plt

# Some function definitions, and their derivatives
# up to the 4th grade, used for testing.


def g(x):
    return x**4 + 4 * x**3 + x**2 - 10 * x + 1


def grad_g(x):
    return 4 * x**3 + 12 * x**2 + 2 * x - 10


def grad2_g(x):
    return 12 * x**2 + 24 * x + 2


def grad3_g(x):
    return 24 * x + 24


def grad4_g(x):
    return 24.0


def h(x):
    return np.sin(x)


def grad_h(x):
    return np.cos(x)


def grad2_h(x):
    return -np.sin(x)


def grad3_h(x):
    return -np.cos(x)


def grad4_h(x):
    return np.sin(x)


# First-order derivative with finite differences.
# This is what we had after the lecture


def grad(f, x, delta=1e-5):
    return (f(x + delta) - f(x - delta)) / (2 * delta)


# This is the start of exercise 1.
# Second-order derivatives


# this is ex. 1 point 1
def grad2_naive1(grad_f, x, delta=1e-5):
    return grad(grad_f, x, delta)


# this is ex. 1 point 2
def grad2_naive2(f, x, delta=1e-5):
    gf = lambda x: grad(f, x, delta)
    return grad(gf, x, delta)


# this is ex. 1 point 3
# NOTE: We use 2*delta here (and 4*delta**2 in the denominator)
#       just so that the result coincides with that of grad2_naive2.
#       (If you expand what that function does, on paper, you will
#       see that the resulting expression is the one written here)
def grad2(f, x, delta=1e-5):
    return (f(x + 2 * delta) - 2 * f(x) + f(x - 2 * delta)) / (4 * delta**2)


# A test to verify that all 3 versions do the same thing, and that the
# result is the same as the analytical expression (up to numerical
# inaccuracies)
def test_grad2(f, grad_f, grad2_f, x, delta=1e-5):
    print("testing second derviatives")
    an = grad2_f(x)  # analytical
    naive1 = grad2_naive1(grad_f, x, delta)  # numerical 1
    naive2 = grad2_naive2(f, x, delta)  # numerical 2
    better = grad2(f, x, delta)  # numerical 3
    print(f"analytical={an} naive1={naive1} naive2={naive2} better={better}")
    print("naive1-analytical=", abs(naive1 - an))
    print("naive2-analytical=", abs(naive2 - an))
    print("better-analytical=", abs(better - an))
    print()


test_grad2(g, grad_g, grad2_g, 0.1234)
test_grad2(h, grad_h, grad2_h, 0.1234)


# This is the start of exercise 2.
# Third-order derivatives. Basically the same story as the second-order one


def grad3_naive1(grad2_f, x, delta=1e-4):
    return grad(grad2_f, x, delta)


def grad3_naive2(f, x, delta=1e-4):
    gf = lambda x: grad(f, x, delta)
    gf2 = lambda x: grad(gf, x, delta)
    return grad(gf2, x, delta)


# this one is the simplified finite-differences expression, the analog of `grad2`
def grad3(f, x, delta=1e-3):
    return (
        f(x + 3 * delta) - 3 * f(x + delta) + 3 * f(x - delta) - f(x - 3 * delta)
    ) / (8 * delta**3)


# always test your code...
def test_grad3(f, grad2_f, grad3_f, x, delta=1e-3):
    print("testing third derviatives")
    an = grad3_f(x)
    naive1 = grad3_naive1(grad2_f, x, delta)
    naive2 = grad3_naive2(f, x, delta)
    better = grad3(f, x, delta)
    print(f"analytical={an} naive1={naive1} naive2={naive2} better={better}")
    print("naive1-analytical=", abs(naive1 - an))
    print("naive2-analytical=", abs(naive2 - an))
    print("better-analytical=", abs(better - an))
    print()


test_grad3(g, grad2_g, grad3_g, 0.1234)
test_grad3(h, grad2_h, grad3_h, 0.1234)

# Fourth-order derivatives - at this point you should see the pattern
# emerging (hint: it's actually clearer if you write it on paper, rather than
# in the code)


def grad4(f, x, delta=1e-3):
    return (
        f(x + 4 * delta)
        - 4 * f(x + 2 * delta)
        + 6 * f(x)
        - 4 * f(x - 2 * delta)
        + f(x - 4 * delta)
    ) / (16 * delta**4)


# this is exercise 2 point 2: n-th order derivatives using recursion
# This evaluates the function 2**n times
def gradn(n, f, x, delta=1e-3):
    if n == 0:
        return f(x)
    return grad(lambda x: gradn(n - 1, f, x, delta), x, delta)


from scipy.special import binom


# This is exercise 2 point 3: n-th order derivatives without using recursion
# This evaluates the function only n+1 times
def gradn_better(n, f, x, delta=1e-3):
    # evaluation points: x shifted by -n*delta, -(n-2)*delta, ..., (n-2)*delta, n*delta
    ev_points = x + np.linspace(-n, n, n + 1) * delta
    # compute f at the evaluation points
    fs = f(ev_points)
    # binomial coefficients
    coeffs = binom(n, np.arange(n + 1))
    # make the signs alternating - but the sign of the first one depends
    # on the parity of `n`, which is the reason for the second term
    coeffs *= (-1) ** np.arange(n + 1) * (-1) ** n
    # take the dot product and divide
    return np.dot(fs, coeffs) / (2 * delta) ** n


# always test your code...
def test_gradn(f, grad_f, grad2_f, grad3_f, grad4_f, x, delta=1e-2):
    print(f"testing n-th derviatives with delta={delta}")
    an = [f(x), grad_f(x), grad2_f(x), grad3_f(x), grad4_f(x)]
    for n in range(5):
        naive = gradn(n, f, x, delta)
        better = gradn_better(n, f, x, delta)
        print(f"order={n} analytical={an[n]} naive={naive} better={better}")
        print("        naive-analytical=", abs(naive - an[n]))
        print("        better-analytical=", abs(better - an[n]))
    for n in range(5, 8):
        naive = gradn(n, f, x, delta)
        better = gradn_better(n, f, x, delta)
        print(f"order={n} naive={naive} better={better}")
        print("        naive-better=", abs(naive - better))
    print()


# NOTE: smaller deltas work well with low-order derivatives, then for
#       higher-order derivatives one needs larger deltas.
test_gradn(g, grad_g, grad2_g, grad3_g, grad4_g, 0.1234, delta=1e-2)
test_gradn(h, grad_h, grad2_h, grad3_h, grad4_h, 0.1234, delta=1e-2)


# This is some code to analyze the effect of delta, as suggested in exercises 1 and 2:
# the code is general (notice that the `grad` argument represents the function that
# computes the numerical derivatives, but they can be derivatives of whatever order,
# so we can pass `grad` but also `grad2` or `gradn` etc.)


# This function  plots the error of finite-differences vs analytical gradients as a function of delta,
# in log-log scale, for a given input.
# For example, for the first derivative, you can try
#    plot_err_vs_delta(g, grad_g, grad, 0.3453)
# For the second derivative:
#    plot_err_vs_delta(g, grad2_g, grad2, 0.3453)
# etc.
#
def plot_err_vs_delta(f, grad_f, grad, x, label=None):
    # log-range of the values to explore (from 1e-10 to 1e10)
    log_deltas = np.arange(-10.0, 10.0, 0.1)
    deltas = 10 ** (log_deltas)
    # a function that computes the finite-differeces version of the gradient,
    # but here it's seen as a function of `delta` while `f` and `x` are fixed
    finite_diff_grad = lambda delta: grad(f, x, delta)
    # again write a function of delta, this time returning the approximation
    # error made for that delta (while `x` is considered fixed)
    error = lambda delta: np.abs(grad_f(x) - finite_diff_grad(delta))
    # now we can just exploit broadcasting to call the `error` function on
    # the whole range of deltas that we're interested in
    all_errors = error(deltas)
    # when plotting, it's nicer to visualize the log of the error
    log_errors = np.log10(all_errors)
    # plot the result
    plt.xlabel("log10(delta)")
    plt.ylabel("log10(error)")
    plt.plot(log_deltas, log_errors, label=label)
    plt.legend()


# This plots several comparisons at once (for a given function), see its usage below
def plot_err_vs_delta_many(f, gradflst, gradlst, x):
    plt.clf()
    for grad_f, grad in zip(gradflst, gradlst):
        plot_err_vs_delta(f, grad_f, grad, x, label=grad_f.__name__)


# plot_err_vs_delta_many(
#     h, [grad_h, grad2_h, grad3_h, grad4_h], [grad, grad2, grad3, grad4], 0.3453
# )

plot_err_vs_delta(g, grad2_g, grad2, 0.3453)
plt.show()
