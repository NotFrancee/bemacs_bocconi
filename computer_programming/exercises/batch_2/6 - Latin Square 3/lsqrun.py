"""Runs the greedy algorithm on TSP
"""

from latinsquare import LatinSquare
from greedy import greedy


def run():
    N = 15
    N_REPEATS = 10
    N_ITERS = 10_000

    probl = LatinSquare(N)
    greedy(probl, N_REPEATS, N_ITERS)


run()
