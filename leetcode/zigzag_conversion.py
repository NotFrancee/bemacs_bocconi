import numpy as np
import time


def solve(s: str, nrows: int):
    t0 = time.time()
    res = ""

    if nrows == 1:
        return s

    for i in range(nrows):
        j = 0
        jump = i
        move_down = True

        while True:
            try:
                j += jump
                res += s[j]

                # calculate the next jump
                jump = 0
                while jump == 0:
                    if move_down:
                        jump = 2 * (nrows - i - 1)
                    else:
                        jump += 2 * i
                    move_down = not move_down
            except:
                break
    t1 = time.time()
    return res, t1 - t0


test1 = "PAYPALISHIRING"
nrows1 = 5

# test = "A"
# nrows = 1
# print(solve(test, nrows))


class Solver:
    def get_cols(self):
        s, nr = self.s, self.nr
        l = len(s)

        el_per_block = 2 * (nr - 1)
        cols_per_block = nr - 1

        total_cols = cols_per_block * (l // el_per_block)
        # print(
        #     f"total cols excluding remainder are {total_cols}. There are {cols_per_block} cols per block and each block has {el_per_block} els "
        # )

        remainder = l % el_per_block
        total_cols += max(remainder - nr + 1, 1)

        return total_cols

    def solve(self, s: str, nr: int):
        t0 = time.time()
        self.s, self.nr = np.array(list(s)), nr

        nc = self.get_cols()

        matr = np.zeros((nr, nc), dtype=int)
        matr[:, 0] = np.arange(nr)

        # jump up is jump down reversed
        row_indices = matr[:, 0]
        jump_up = 2 * (nr - row_indices - 1)
        jump_down = 2 * row_indices
        # print("jump")
        # print(jump_up, jump_down)
        # move_down = True

        for j in range(1, nc):
            # if col odd apply a func, otherwise the other
            if j % 2 == 0:
                jump = jump_down
            else:
                jump = jump_up

            # print(f"about to make jump at col {j}\n", jump, matr[:, j - 1])
            matr[:, j] = matr[:, j - 1] + jump

        print(f"matr before deleting {matr}")
        # matr[[0, -1], 1::2] = -1
        matr = matr[(matr >= 0) & (matr < len(s))].flatten()
        print(f"matr before pulling unique {matr}")
        unique_idx = np.unique(matr, return_index=True)
        print(unique_idx)
        # matr = matr[unique_idx]
        print(f"matr after {matr}")

        res = "".join(self.s[matr])
        print(res)
        t1 = time.time()
        return res, t1 - t0


def trials(n, s, nr):
    meas = np.zeros((2, n))
    solver = Solver()

    for i in range(n):
        _, timefast = solver.solve(s, nr)
        _, timeslow = solve(s, nr)

        meas[0, i] = timeslow
        meas[1, i] = timefast

    avg = meas.mean(axis=1)
    stdev = meas.std(axis=1)

    print(
        f"on average fast is {(avg[1] / avg[0] - 1) * 100:.2f}% faster than slow, with stdev {np.mean(stdev)}"
    )

    return meas.mean(axis=1), meas.std(axis=1)


s = "ABC"
solver = Solver()
solver.solve(s, 2)

a = np.array([1, 2, 14, 14, 14, 2, 3, 3, 4, 4, 5, 5, 8, 8, 8, 10])

print(np.unique(a, return_index=False, return_counts=False, return_inverse=True))
