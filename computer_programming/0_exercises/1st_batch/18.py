import numpy as np


def genarr(n: int):
    arr = np.linspace(0, 2 * np.pi, n)
    arr2 = np.linspace(0, 2 * np.pi, n, endpoint=False)

    return arr, arr2


arr, arr2 = genarr(10)
print(arr, arr2)
sin1 = np.sin(arr)
sin2 = np.sin(arr2)
