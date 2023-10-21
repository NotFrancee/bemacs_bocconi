"""Runs the greedy algorithm on TSP
"""

from magic_square import MagicSquare
from greedy import greedy


def run():
    N = 4
    SUM = 25
    N_REPEATS = 5
    N_ITERS = 10_000

    probl = MagicSquare(N, SUM)
    greedy(probl, N_REPEATS, N_ITERS)


run()
