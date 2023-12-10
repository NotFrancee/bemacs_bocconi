import numpy as np
import matplotlib.pyplot as plt

## Import some symbols from the other modules
## directly into the script
from function_examples_1d import k
from GradDesc import grad_desc1d

## Plot the function k in the range [-5,3].
## We take 1000 samples: if you zoom in enough you'll notice it.
## Note that `k(xv)` uses broadcasting! This is possible because
## of how the function `k` was written. It might not be possible
## in general (for example if inside the function you have a
## conditional like `if x < 3` this will not work and you'd need
## to use a comprehension, or a map).
xv = np.linspace(-5, 3, 1000)
yv = k(xv)

plt.clf()
plt.plot(xv, yv)

## Optimize k.
## Play around with the parameters and the initial value to see
## their effect on the convergence and on the time required.
## This uses automatic differentiation.
x0 = 0.0
xf, xs, converged = grad_desc1d(k, x0,
                                alpha = 1e-2,
                                max_t = 10000)
print(f"converged = {converged}")
plt.plot(xs, k(xs), '-+')
