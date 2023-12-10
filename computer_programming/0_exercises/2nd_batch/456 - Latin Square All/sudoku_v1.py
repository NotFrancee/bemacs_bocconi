from copy import deepcopy
import numpy as np


def check(arr: np.ndarray, n: int):
    return n - len(np.unique(arr))


class Sudoku:
    def __init__(self, n: int):
        if not int(np.sqrt(n)) == np.sqrt(n):
            raise Exception("n must be perfect square")

        self.n = n
        self.sn = np.sqrt(n).astype(int)
        self.config = np.zeros((n, n), dtype=int)
        self.init_config()

    # new initialization
    def init_config(self):
        n = self.n

        self.config[:] = np.arange(n).reshape(-1, 1) * np.ones(n)

    def subsquare2view(self, si: int, sj: int):
        sn, config = self.sn, self.config

        if si >= sn or sj >= sn:
            raise ValueError("index out of range")

        return config[sn * si : sn * (si + 1), sn * sj : sn * (sj + 1)]  # noqa

    def index2subsquare(self, i: int, j: int):
        sn = self.sn

        return np.floor_divide([i, j], sn)

    def cost(self):
        cost = 0
        n, sn, config = self.n, self.sn, self.config
        for i in range(n):
            cost += check(config[i, :], n) + check(config[:, i], n)

        for si in range(sn):
            for sj in range(sn):
                cost += check(self.subsquare2view(si, sj), n)
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

        n, sn, config = self.n, self.sn, self.config

        row1 = config[rowi1, :].copy()
        row2 = config[rowi2, :].copy()

        cost_old = check(row1, n) + check(row2, n)

        # inverting the rows and cols
        row1[col], row2[col] = row2[col], row1[col]

        cost_new = check(row1, n) + check(row2, n)

        # adding subsquares
        si1, sj1 = self.index2subsquare(rowi1, col)
        si2, sj2 = self.index2subsquare(rowi2, col)
        # if exchanging in the same subsquare delta cost is 0
        if not (si1 == si2 and sj1 == sj2):
            sub1 = self.subsquare2view(si1, sj1).copy()
            sub2 = self.subsquare2view(si2, sj2).copy()

            cost_old += check(sub1, n) + check(sub2, n)

            sub1[rowi1 % sn, col % sn], sub2[rowi2 % sn, col % sn] = (
                sub2[rowi2 % sn, col % sn],
                sub1[rowi1 % sn, col % sn],
            )

            cost_new += check(sub1, n) + check(sub2, n)

        delta_cost = cost_new - cost_old

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
    ex = Sudoku(4)

    print(ex)
    print(ex.cost())

    move = ex.propose_move()

    print(move)
    ex.compute_delta_cost(move)

    ex.accept_move(move)
    print(ex)


test()
