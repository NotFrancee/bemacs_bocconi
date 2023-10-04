"""Runs the greedy algorithm on TSP
"""

from latinsquare import LatinSquare
from greedy import greedy

N = 5
N_REPEATS = 2
N_ITERS = 10_000

probl = LatinSquare(N)
greedy(probl, N_REPEATS, N_ITERS)
