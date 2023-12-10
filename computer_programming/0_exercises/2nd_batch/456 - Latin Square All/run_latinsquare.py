"""Runs the greedy algorithm on TSP
"""

from time import time
from lsquare_v1 import LatinSquare as LS1
from lsquare_v2 import LatinSquare as LS2
from lsquare_v3 import LatinSquare as LS3
from greedy import greedy

N = 15
N_REPEATS = 2
N_ITERS = 20_000

problv1 = LS1(N)
problv2 = LS2(N)
problv3 = LS3(N)

t0 = time()
greedy(problv1, N_REPEATS, N_ITERS, debug=False)

t1 = time()
greedy(problv2, N_REPEATS, N_ITERS, debug=False)

t2 = time()
greedy(problv3, N_REPEATS, N_ITERS, debug=False)
t3 = time()

print("---Time Summary---")
print(f"V1: {t1-t0:.3f}")
print(f"V2: {t2-t1:.3f}")
print(f"V3: {t3-t2:.3f}")
print(f"V2 is {(t2-t1)/(t1-t0):.3%} the time of V1")
print(f"V3 is {(t3-t2)/(t2-t1):.3%} the time of V2")
