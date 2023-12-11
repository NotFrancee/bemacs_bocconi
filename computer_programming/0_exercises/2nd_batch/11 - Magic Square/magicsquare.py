from copy import deepcopy
import numpy as np


class MagicSquare:
    def __init__(self, n: int, s: int) -> None:
        if n < 3:
            raise Exception("Size must be > 2.")

        M = int(n * (n**2 + 1) / 2)
        print(f"M is {M}")

        self.n = n
        self.s = s

        self.config = np.zeros((n, n), dtype=int)

    def init_config(self):
        n, s = self.n, self.s
        self.config[:] = np.random.randint(1, s, size=(n, n))

    def cost(self):
        n, s, config = self.n, self.s, self.config

        rows_cost = s - config.sum(axis=0)
        cols_cost = s - config.sum(axis=1)

        diag1_cost = s - config.trace()
        diag2_cost = s - config.T.trace()

        unique_cost = n * n - len(np.unique(config))

        return np.sum(
            np.absolute(
                [
                    *rows_cost,
                    *cols_cost,
                    diag1_cost,
                    diag2_cost,
                    unique_cost,
                ]
            )
        )

    def propose_move(self):
        n, s = self.n, self.s

        i, j = np.random.randint(n, size=2)

        while True:
            val = np.random.randint(1, s)

            if self.config[i, j] != val:
                break

        return i, j, val

    def compute_delta_cost(self, move: tuple[int, int, int]):
        n, s, config = self.n, self.s, self.config
        i, j, val = move

        # old method
        p = self.copy()
        p.accept_move(move)
        delta_old = p.cost() - self.cost()

        # new method
        delta_val = val - config[i, j]

        delta_row = abs(s - (config[i, :].sum() + delta_val)) - abs(
            s - config[i, :].sum()
        )
        delta_col = abs(s - (config[:, j].sum() + delta_val)) - abs(
            s - config[:, j].sum()
        )

        delta = delta_col + delta_row

        if i == j:
            delta_diag1 = abs(s - (config.trace() + delta_val)) - abs(
                s - config.trace()
            )
            delta += delta_diag1
        if i + j == n - 1:
            delta_diag2 = abs(s - (config.T.trace() + delta_val)) - abs(
                s - config.T.trace()
            )
            delta += delta_diag2

        # add if unique

        print(move)
        print(delta, delta_old)
        assert delta == delta_old

        return delta

    def accept_move(self, move: tuple[int, int, int]):
        i, j, val = move

        self.config[i, j] = val

    def copy(self):
        return deepcopy(self)

    def display(self):
        return

    def __repr__(self) -> str:
        return str(self.config)
