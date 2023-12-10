from copy import deepcopy
from typing import Optional
import numpy as np


def check(arr: np.ndarray, n: int):
    return n - len(np.unique(arr))


class Sudoku:
    def __init__(
        self,
        n: Optional[int] = None,
        sudoku_inst=None,
        r: Optional[float] = None,
    ):
        self.mask = None

        if isinstance(sudoku_inst, Sudoku):
            print("you have provided a pre-initialized sudoku")
            if r is None or not 0 < r < 1:
                raise ValueError("r must be between 0 and 1")

            n = sudoku_inst.n
            self.n = n

            n_entries = round(r * n * n)

            # create mask for fixed entries
            mask = np.zeros((n, n), dtype=bool)
            mask.ravel()[:n_entries] = True
            mask[:] = np.random.permutation(mask)
            self.mask = mask

            self.config = np.zeros((n, n), dtype=int)
            self.init_config(sudoku_inst.config)

        if isinstance(n, int):
            print("initializing a new sudoku")
            if not int(np.sqrt(n)) == np.sqrt(n):
                raise Exception("n must be perfect square")

            self.n = n
            self.config = np.zeros((n, n), dtype=int)
            self.init_config()

        self.sn = np.sqrt(self.n).astype(int)

    # new initialization
    def init_config(self, preinitialized_config: Optional[np.ndarray] = None):
        n, config = self.n, self.config
        mask = self.mask

        if preinitialized_config is None:
            config[:] = np.arange(n).reshape(-1, 1) * np.ones(n)
        else:
            if mask is None:
                raise Exception("mask is none")

            # shuffle the entries
            config[mask] = preinitialized_config[mask]

            for col in range(n):
                col_mask = mask[:, col]
                permuted = np.random.permutation(
                    preinitialized_config[:, col][~col_mask],
                )
                config[:, col][~col_mask] = permuted

            assert np.array_equal(config[mask], preinitialized_config[mask])

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
        n, mask = self.n, self.mask
        col = np.random.randint(n)

        while True:
            rowi1, rowi2 = np.random.choice(
                np.arange(n),
                replace=False,
                size=2,
            )

            # if the items are fixed do not change them
            if mask is not None:
                if mask[rowi1, col] or mask[rowi2, col]:
                    continue

            break

        return col, rowi1, rowi2

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
