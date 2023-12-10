import numpy as np

## Some examples of 1-D functions, i.e. functions R->R.
## for use with grad_desc1d

def g(x):
    return x**4 + 4 * x**3 + x**2 - 10 * x + 1

def grad_g(x):
    return 4 * x**3 + 12 * x**2 + 2 * x - 10

def h(x):
    return np.log(1 + np.exp(x))

def k(x):
    return h(g(x))
