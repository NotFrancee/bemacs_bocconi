"""Runs the greedy algorithm on the Sudoku
"""

from time import time
from sudoku_v2 import Sudoku
from greedy import greedy

N = 9
N_REPEATS = 10
N_ITERS = 10_000

s = Sudoku(N)

t0 = time()
greedy(s, N_REPEATS, N_ITERS, debug=True)
t1 = time()

print(s)
input("continue to solver")

new = Sudoku(sudoku_inst=s, r=0.1)

t2 = time()
greedy(new, N_REPEATS, N_ITERS, debug=True)
t3 = time()

print("---Time Summary---")
print(f"V1: {t1-t0:.3f}")
print(f"Solver: {t3-t2:.3f}")
