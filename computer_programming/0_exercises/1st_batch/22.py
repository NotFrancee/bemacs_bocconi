import numpy as np


def gen_arr(n: int):
    arr = [np.arange(i, i + 5) for i in range(n)]
    return arr


li = gen_arr(5)
print(li)

print(np.array(li))

print(np.array(li).flatten())
