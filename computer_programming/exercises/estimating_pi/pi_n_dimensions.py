# create a function that does this in as many dimensions as you want
# 1. redo in 3D, 4D, ...; (look up the formula on wikipedia)
# study the scaling of the variance of the estimate with the number of points

# volume of a unit n-ball:
# V(n) = pi^(n/2) / gamma(n/2 + 1)
# we are only going to select one sector of the n dimensional cube, so we have to divide by 2^n
# thus pi_est = (2^(n) * gamma(n/2 + 1) * ninside / npoints)^(2/n)


# check that it converges to better and better estimates of pi (on average) and that std goes down as 1/sqrt(n)

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def estimate_pi(npoints, ndims):
    # create n x ndim random numbers
    # columns are the various values, rows are the dimensions (axis)
    points = np.random.random((npoints, ndims))

    # check the points that fall inside
    dist = np.sum(points**2, axis=1)
    ninside = np.sum(dist <= 1)

    # estimate pi
    pi_est = np.exp(
        (2 / ndims) * (np.log(gamma(ndims / 2 + 1)) + np.log(ninside / npoints))
        + 2 * np.log(2)
    )

    return pi_est


def display(x_values, means, stds, x_labels):
    tup = plt.subplots(1, 2)
    fig = tup[0]
    ax: list[Axes] = tup[1]

    for i in range(len(x_values)):
        ax[i].plot(x_values[i], means[i], stds[i])
        if i == 1:
            ax[i].set_xscale("log")
        ax[i].set_xlabel(x_labels[i])

    plt.show()


# study the behavior of this function as n and dims are (independenlty) increased.
# estimate the variance from repeated trials
def compute_stats(trials, dims, points):
    n_dims_est = len(dims)
    dim_estimates_means = np.zeros(n_dims_est)
    dim_estimates_std = np.zeros(n_dims_est)

    n_points_est = len(points)
    points_est_means = np.zeros(n_points_est)
    points_est_std = np.zeros(n_points_est)

    # estimates = np.zeros(trials)

    for j, dim in enumerate(dims):
        estimates = np.zeros(trials)

        for i in range(trials):
            pi_est = estimate_pi(1000, dim)
            estimates[i] = pi_est

        dim_estimates_means[j] = estimates.mean()
        dim_estimates_std[j] = estimates.std()

    for j, n_points in enumerate(points):
        estimates = np.zeros(trials)

        for i in range(trials):
            pi_est = estimate_pi(n_points, 2)
            estimates[i] = pi_est

        points_est_means[j] = estimates.mean()
        points_est_std[j] = estimates.std()

    print("dims", dim_estimates_means, dim_estimates_std, sep="\n")
    print("npoints", points_est_means, points_est_std, sep="\n")

    x_values = [dims, points]
    means_values = [dim_estimates_means, points_est_means]
    std_values = [dim_estimates_std, points_est_std]
    x_labels = ["N Dims", "N points"]

    display(x_values, means_values, std_values, x_labels)


dims = np.arange(2, 11)
npoints = np.linspace(100, 100000, 100, dtype=int)
compute_stats(50, dims, npoints)
