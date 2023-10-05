import numpy as np
from copy import deepcopy

"""Goal: each row and col must have all the numbers between 0 and n-1
Like sudoku but without the sub-squares

The cost function is the number of squares that violate the constraint.
The min cost is 0 and that means we reached our goal
"""

import time


def is_row_valid(arr, n):
    unique = np.unique(arr)
    return len(unique) == n


class LatinSquare:
    def __init__(self, n: int) -> None:
        if not isinstance(n, int) and n > 0:
            raise Exception("n must be an integer > 0")

        self.n = n
        self.config = np.zeros((n, n))

        self.init_config()

    def __repr__(self) -> str:
        return str(self.config)

    def display(self):
        pass

    def copy(self):
        return deepcopy(self)

    def init_config(self):
        n = self.n
        for i in range(n):
            for j in range(n):
                self.config[i, j] = np.random.randint(n)

    def cost(self):
        n, config = self.n, self.config
        cost = 0

        for i in range(n):
            row = config[i, :]
            col = config[:, i]

            if not is_row_valid(row, n):
                cost += 1

            if not is_row_valid(col, n):
                cost += 1

        return cost

    def propose_move(self):
        # first method: pick a random cell and change its value
        # make sure it actually changes

        n = self.n

        row = np.random.randint(n)
        col = np.random.randint(n)

        prev = self.config[row, col]

        while True:
            new = np.random.randint(n)
            if new != prev:
                break

        move = (row, col, new)
        return move

    def compute_delta_cost(self, move):
        n = self.n

        row, col, new_value = move

        old_row = self.config[row]
        old_col = self.config[:, col]

        new_row = old_row.copy()
        new_col = old_col.copy()
        new_row[col] = new_value
        new_col[row] = new_value

        # if old and row are different, change the delta cost
        old_cost = sum([not is_row_valid(arr, n) for arr in [old_col, old_row]])
        new_cost = sum([not is_row_valid(arr, n) for arr in [new_col, new_row]])

        delta_cost = new_cost - old_cost

        return delta_cost

    def accept_move(self, move):
        row, col, new_value = move

        self.config[row, col] = new_value


t = LatinSquare(5)

for i in range(100):
    move = t.propose_move()

    t.compute_delta_cost(move)
