"""Runs the greedy algorithm on TSP
"""

from time import time
from sudoku_v1 import Sudoku as S1
from greedy import greedy

N = 4
N_REPEATS = 2
N_ITERS = 20_000

s1 = S1(N)

t0 = time()
greedy(s1, N_REPEATS, N_ITERS, debug=True)

t1 = time()

print("---Time Summary---")
print(f"V1: {t1-t0:.3f}")
