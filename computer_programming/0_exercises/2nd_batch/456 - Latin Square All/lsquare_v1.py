from copy import deepcopy
import numpy as np


def check(arr: np.ndarray, n: int):
    return len(np.unique(arr)) == n


class LatinSquare:
    def __init__(self, n: int):
        self.n = n
        self.config = np.zeros((n, n), dtype=int)
        self.init_config()

    def init_config(self):
        n = self.n
        self.config[:] = np.random.randint(n, size=(n, n))

    def cost(self):
        cost = 0
        n, config = self.n, self.config
        for i in range(n):
            if not check(config[i, :], n):
                cost += 1

            if not check(config[:, i], n):
                cost += 1

        return cost

    def propose_move(self):
        n, config = self.n, self.config
        row, col = np.random.randint(n, size=2)

        choices = np.arange(n)
        new = np.random.choice(choices[choices != config[row, col]])

        return row, col, new

    def compute_delta_cost(
        self, move: tuple[int, int, int], debug_delta_cost: bool = False
    ):
        row, col, new = move
        n, config = self.n, self.config

        old_r = config[:, col].copy()
        old_c = config[row, :].copy()

        checks = np.zeros((2, 2), dtype=int)
        checks[0] = [check(old_r, n), check(old_c, n)]

        old_r[row] = new
        old_c[col] = new
        checks[1] = [check(old_r, n), check(old_c, n)]

        delta = np.sum(checks[0] - checks[1])

        return delta

    def accept_move(self, move: tuple[int, int, int]):
        row, col, new = move

        self.config[row, col] = new

    def __repr__(self) -> str:
        return str(self.config)

    def display(self):
        pass

    def copy(self):
        return deepcopy(self)


def test():
    ex = LatinSquare(3)

    print(ex)
    print(ex.cost())

    move = ex.propose_move()

    print(move)
    ex.compute_delta_cost(move)

    ex.accept_move(move)
    print(ex)


test()
