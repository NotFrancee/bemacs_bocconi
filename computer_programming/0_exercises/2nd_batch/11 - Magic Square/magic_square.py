# basic ideas
# cost : number of items that do not satisfy the constraint
# propose move:

# start by imposing no constraints

from matplotlib.axes import Axes
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class MagicSquare:
    def __init__(self, n: int, s: int) -> None:
        self.n, self.s = n, s

        if not (isinstance(n, int) and isinstance(s, int)):
            raise Exception("n and s must be integers")

        if not (s > n**2 + 1):
            raise Exception("s must be greater than n ** 2 + 1")

        self.config = np.zeros((n, n), dtype=int)
        self.init_config()

    def __repr__(self) -> str:
        return str(self.config)

    def init_config(self):
        # initialize with random ints between 0 and s
        # constraints for the numbers
        # (1) all positive integers => n between 1 and s
        # (2) all distinct integers
        #   => size of the matrix is n ** 2 so you need n^2 ints
        #   => between 1 and s there are s - 1 ints
        #   => s-1 > n**2 => s > n**2 + 1

        n, s = self.n, self.s

        # +1 to remove 0
        sample = np.random.choice(s - 1, n**2, replace=False) + 1
        self.config[:] = sample.reshape(n, n)

    def cost(self):
        # calculating cost:
        # (1) how much the sums are away from the tgt sum
        # (2) all entries must be different
        #   (one unit of cost per duplicate number)

        config, n, s = self.config, self.n, self.s
        # sums is [cols, rows, diags]
        sums = np.zeros(2 * n + 2, dtype=int)

        sums[:n] = config.sum(axis=0)
        sums[n : n * 2] = config.sum(axis=1)
        sums[n * 2 :] = config.trace(), np.rot90(config).trace()
        cost = np.sum(np.abs(sums - s))

        # calculating the cost of duplicate entries
        unique = np.unique(config)
        n_dups = n**2 - len(unique)
        cost += n_dups

        # store the distances to better choose move proposals?
        return cost

    def propose_move(self):
        # select cell at random and choose a new number at random
        n, s, config = self.n, self.s, self.config

        i, j = np.random.choice(n, 2)

        unique = np.unique(self.config)
        selection = np.setdiff1d(np.arange(1, s), unique)
        new_value = np.random.randint(1, s)

        return i, j, new_value

    def accept_move(self, move):
        i, j, val = move
        self.config[i, j] = val

    def compute_delta_cost(self, move):
        n, s, config = self.n, self.s, self.config
        i, j, new_val = move

        old_val = config[i, j]

        if i == j:
            old_sum = config.trace()
            new_sum = old_sum - old_val + new_val
            # compute diag cost
            # catch: you also have to catch the case where it is in the opposite diagonal

            pass
        if n % 2 == 1 and i == (n - 1) / 2:
            pass

        # compute row and col cost

        # old method
        old_cost = self.cost()

        new_probl = self.copy()
        new_probl.accept_move(move)
        new_cost = new_probl.cost()

        delta_cost = new_cost - old_cost
        return delta_cost

    def copy(self):
        return deepcopy(self)

    def display(self):
        print(self.config)
