"""Runs the greedy algorithm on TSP
"""

from TSP import TSP
from Greedy import greedy

CITIES = 100
N_REPEATS = 20
N_ITERS = 10_000

probl = TSP(CITIES)
greedy(probl, N_REPEATS, N_ITERS)
