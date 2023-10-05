"""Runs the greedy algorithm on TSP
"""

from sudoku import Sudoku
from greedy import greedy


def run():
    N = 9
    N_REPEATS = 2
    N_ITERS = 10_000

    probl = Sudoku(N)
    greedy(probl, N_REPEATS, N_ITERS)


run()
