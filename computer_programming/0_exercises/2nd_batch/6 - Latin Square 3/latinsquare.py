from copy import deepcopy
import numpy as np


# update row cost
def row_cost(arr, n):
    """Computes cost of a row/column.
    Returns 1 if the elements are not all unique, 0 otherwise"""

    unique = np.unique(arr)
    return n - len(unique)


class LatinSquare:
    """Latin square class"""

    def __init__(self, n: int) -> None:
        if not isinstance(n, int) and n > 0:
            raise Exception("n must be an integer > 0")

        self.n = n
        self.config = np.arange(n, dtype=int).reshape((-1, 1)) * np.ones(n, dtype=int)

    def __repr__(self) -> str:
        return str(self.config)

    def display(self):
        """Placeholder"""
        pass

    def copy(self):
        """Deepcopies the object"""
        return deepcopy(self)

    def init_config(self):
        """Inits the config with columns all ranges"""
        n = self.n
        self.config[:] = np.arange(n, dtype=int).reshape((-1, 1)) * np.ones(
            n, dtype=int
        )

    def cost(self):
        """Computes the cost of the current configuration"""
        n, config = self.n, self.config
        cost = 0

        for i in range(n):
            row = config[i, :]
            col = config[:, i]

            cost += row_cost(row, n)
            cost += row_cost(col, n)

        return cost

    def propose_move(self):
        """pick a random col and swap two entries at random"""
        n = self.n

        col_index = np.random.randint(n)
        i1, i2 = np.random.choice(n, size=2, replace=False)

        move = (col_index, i1, i2)
        return move

    def compute_delta_cost(self, move, debug=False):
        """Compute the delta cost between the two moves"""
        n = self.n

        col, i1, i2 = move

        old_row1 = self.config[i1]
        old_row2 = self.config[i2]

        new_row1 = old_row1.copy()
        new_row2 = old_row2.copy()

        new_row1[col], new_row2[col] = new_row2[col], new_row1[col]

        old_cost = row_cost(old_row1, n) + row_cost(old_row2, n)
        new_cost = row_cost(new_row1, n) + row_cost(new_row2, n)

        delta_cost = new_cost - old_cost

        if debug:
            old_cost = self.cost()
            new_probl = self.copy()
            new_probl.accept_move(move)
            new_cost = new_probl.cost()

            old_delta = new_cost - old_cost
            assert old_delta == delta_cost

        return delta_cost

    def accept_move(self, move):
        """Accepts the move by switching the two items in the configuration"""
        col_i, i1, i2 = move

        config = self.config

        config[i1, col_i], config[i2, col_i] = (
            config[i2, col_i],
            config[i1, col_i],
        )


def test():
    """Testing"""
    t = LatinSquare(5)

    for _ in range(100):
        move = t.propose_move()

        t.compute_delta_cost(move)


test()
