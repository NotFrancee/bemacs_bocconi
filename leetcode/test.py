import numpy as np
import time
import numpy.random as rnd

# def estimate_pi(n, dims=2):
#     assert(dims >= 2)
#     ps = rnd.rand(dims,n)
#     ninside = ((ps**2).sum(axis=0) <= 1).sum()
#     return ninside / n * 4
# def test(): 
#     t0 = time.time()
#     pi = estimate_pi(100_000_000)
#     t1 = time.time()
#     print(f'{t1-t0}')

# test()

def listsplit(l: list[int]):
    sum_left = 0
    sum_right = 0
    diff = 0
    idx = len(l)-1


    for i in l: 
        sum_right += i
    diff = abs(sum_left - sum_right)

    for i in range(len(l)):
        item = l[i] 
        sum_left += item
        sum_right -= item

        new_diff = abs(sum_left - sum_right)
        if new_diff < diff: 
            diff = new_diff
            idx = i

    return idx + 1

assert listsplit([2, 4, -1, 5, -2]) == 3
assert listsplit([-12, 14, 1, -5, -2]) == 4
assert listsplit([1, 2, 3, 4, 5, 6]) == 4
