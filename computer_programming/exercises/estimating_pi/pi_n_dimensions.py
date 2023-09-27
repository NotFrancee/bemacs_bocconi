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


class Trials:
    def __init__(self, trials, dims, points) -> None:
        self.trials = trials
        self.dims = dims
        self.points = points

        self.dim_est_means = np.zeros(len(dims))
        self.dim_est_std = np.zeros(len(dims))

        self.points_est_means = np.zeros(len(points))
        self.points_est_std = np.zeros(len(points))

    def display(self):
        x_values = [self.dims, self.points]
        y_row1 = [self.dim_est_means, self.points_est_means]
        y_row2 = [self.dim_est_std, self.points_est_std]
        x_labels = ["N Dims", "N points"]

        fig, (row1, row2) = plt.subplots(2, 2, sharex="col")

        for i, ax in enumerate(row1):
            ax: Axes = ax
            ax.plot(x_values[i], y_row1[i], color="orange")
            ax.hlines(np.pi, min(x_values[i]), max(x_values[i]), colors=["black"])
            ax.set_xlabel(x_labels[i])
            if i == 0:
                ax.set_ylabel("Mean")

        for i, ax in enumerate(row2):
            ax.plot(x_values[i], y_row2[i])
            if i == 1:
                ax.set_xscale("log")
            ax.set_xlabel(x_labels[i])
            if i == 0:
                ax.set_ylabel("Stdev")

            # plot 1/n^2
            xs = np.linspace(max(x_values[i]), 100)
            ys = 1 / (np.sqrt(xs))
            plt.plot(xs, ys, color="gray")

        plt.tight_layout()
        plt.show()

    def run(self):
        estimates = np.zeros(self.trials)

        for j, ndim in enumerate(self.dims):
            for i in range(self.trials):
                pi_est = estimate_pi(50000, ndim)
                estimates[i] = pi_est

            self.dim_est_means[j] = estimates.mean()
            self.dim_est_std[j] = estimates.std()

        for j, n_points in enumerate(self.points):
            for i in range(self.trials):
                pi_est = estimate_pi(n_points, 2)
                estimates[i] = pi_est

            self.points_est_means[j] = estimates.mean()
            self.points_est_std[j] = estimates.std()


dims = np.arange(2, 11)
npoints = np.linspace(10, 100000, 100, dtype=int)

trials = Trials(50, dims, npoints)
trials.run()
trials.display()
