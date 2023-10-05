from copy import deepcopy
import numpy as np


# update row cost
def row_cost(arr, n):
    """Computes cost of a row/column.
    Returns 1 if the elements are not all unique, 0 otherwise"""

    unique = np.unique(arr)
    return n - len(unique)


class Sudoku:
    """Sudoku gen class"""

    def __init__(self, n: int) -> None:
        if not isinstance(n, int) and n > 0:
            raise Exception("n must be an integer > 0")

        if not isinstance(int(np.sqrt(n)), int):
            raise Exception("n must be a perfect square")

        self.n = n
        self.sn = int(np.sqrt(n))
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

    def get_subsquare_index(self, i, j):
        sn = self.sn
        si, sj = i // sn, j // sn

        subsquare_index = (si, sj)
        # print(f"{(i, j)} => {subsquare_index}")
        return subsquare_index

    def get_subsquare_view(self, si, sj, config=None):
        """Gets the subsquare given the indices si and sj.
        i and j will be both between 0 and sn - 1"""

        n = self.n
        subsquare_size = self.sn

        if config is None:
            config = self.config

        start_i = subsquare_size * si
        end_i = start_i + subsquare_size
        start_j = subsquare_size * sj
        end_j = start_j + subsquare_size

        if end_i == n:
            end_i = None
        if end_j == n:
            end_j = None

        subsquare = config[start_i:end_i, start_j:end_j]
        return subsquare

    def cost(self):
        """Computes the cost of the current configuration"""
        n, config = self.n, self.config
        cost = 0

        for i in range(n):
            row = config[i, :]
            col = config[:, i]

            cost += row_cost(row, n)
            cost += row_cost(col, n)

        # now calculate the additional cost for each subsquare
        for i in range(self.sn):
            for j in range(self.sn):
                subsquare = self.get_subsquare_view(i, j)
                cost += row_cost(subsquare, n)

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
        n, config = self.n, self.config
        col, i1, i2 = move

        # create a copy of the config and switch the elements
        new_config = config.copy()
        new_config[i1, col], new_config[i2, col] = (
            new_config[i2, col],
            new_config[i1, col],
        )

        # access the affected rows and their new versions
        old_row1, old_row2 = config[i1], config[i2]
        new_row1, new_row2 = new_config[i1], new_config[i2]

        old_cost = row_cost(old_row1, n) + row_cost(old_row2, n)
        new_cost = row_cost(new_row1, n) + row_cost(new_row2, n)

        # now we have to consider the subsquares as well and we have 2 cases
        # 1) the two indices are in the same subsquare
        # => in that case the cost is the same
        # 2) the two indices are in different subsquares
        # => the cost changes

        # calculate the indices of the subsquares
        subsquare1_i = self.get_subsquare_index(i1, col)
        subsquare2_i = self.get_subsquare_index(i2, col)

        # we only consider case 2
        if subsquare1_i == subsquare2_i:
            pass
        else:
            old_subsquare1 = self.get_subsquare_view(*subsquare1_i)
            old_subsquare2 = self.get_subsquare_view(*subsquare2_i)
            new_subsquare1 = self.get_subsquare_view(*subsquare1_i, config=new_config)
            new_subsquare2 = self.get_subsquare_view(*subsquare2_i, config=new_config)

            old_cost += row_cost(old_subsquare1, n) + row_cost(old_subsquare2, n)
            new_cost += row_cost(new_subsquare1, n) + row_cost(new_subsquare2, n)

        delta_cost = new_cost - old_cost

        if debug:
            old_cost = self.cost()
            new_probl = self.copy()
            new_probl.accept_move(move)
            new_cost = new_probl.cost()

            old_delta = new_cost - old_cost
            print(old_delta, delta_cost)
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
    t = Sudoku(9)
    print(t)
    t.cost()

    for _ in range(100):
        move = t.propose_move()

        t.compute_delta_cost(move)


test()
