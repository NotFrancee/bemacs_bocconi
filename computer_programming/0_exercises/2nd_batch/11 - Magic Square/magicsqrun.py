"""Runs the greedy algorithm on TSP
"""

from magicsquare import MagicSquare
from greedy import greedy


def run():
    N = 4
    SUM = 34
    N_REPEATS = 20
    N_ITERS = 10_000

    probl = MagicSquare(N, SUM)
    greedy(probl, N_REPEATS, N_ITERS)


run()
