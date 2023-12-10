from copy import deepcopy
import numpy as np


def check(arr: np.ndarray, n: int):
    return len(np.unique(arr)) == n


class LatinSquare:
    def __init__(self, n: int):
        self.n = n
        self.config = np.zeros((n, n), dtype=int)
        self.init_config()

    # new initialization
    def init_config(self):
        n = self.n

        self.config[:] = np.arange(n).reshape(-1, 1) * np.ones(n)
        assert self.cost() == n

    def cost(self):
        cost = 0
        n, config = self.n, self.config
        for i in range(n):
            if not check(config[i, :], n):
                cost += 1

            if not check(config[:, i], n):
                cost += 1

        return cost

    # new move proposal
    def propose_move(self):
        n = self.n
        col = np.random.randint(n)
        rowi1, rowi2 = np.random.choice(np.arange(n), replace=False, size=2)

        return col, rowi1, rowi2

    # new deltacost
    def compute_delta_cost(
        self, move: tuple[int, int, int], debug_delta_cost: bool = False
    ):
        col, rowi1, rowi2 = move

        n, config = self.n, self.config

        row1 = config[rowi1, :].copy()
        row2 = config[rowi2, :].copy()

        delta_cost = check(row1, n) + check(row2, n)

        row1[col], row2[col] = row2[col], row1[col]

        delta_cost -= check(row1, n) + check(row2, n)

        if debug_delta_cost:
            c = self.copy()
            old_c = c.cost()
            c.accept_move(move)
            delta_dbg = c.cost() - old_c
            assert delta_cost == delta_dbg

        return delta_cost

    def accept_move(self, move: tuple[int, int, int]):
        col, rowi1, rowi2 = move
        config = self.config

        config[rowi1, col], config[rowi2, col] = (
            config[rowi2, col],
            config[rowi1, col],
        )

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
