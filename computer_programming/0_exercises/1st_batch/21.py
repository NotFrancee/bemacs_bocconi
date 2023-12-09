import numpy as np


a = np.arange(20).reshape(4, 5)
print(a)

n, m = a.shape
print(np.arange(n).reshape(-1, 1) + np.arange(m))
a[(np.arange(n).reshape(-1, 1) + np.arange(m)) % 2 == 0] = 0
print(a)
