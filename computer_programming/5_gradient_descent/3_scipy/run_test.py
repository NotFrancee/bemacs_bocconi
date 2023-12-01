from operator import call
import numpy as np
from function_examples import g, grad_g, k, grad_k

# ~~~~~~~~~~~~~~~~~~~~~ ##

# Some commands to visualize a function R^2 -> R,
# like the functions g and k that we're trying to optimize

import matplotlib.pyplot as plt

plt.close("all")  # close the previous figures
fig = plt.figure()  # create a new one
x0 = np.linspace(-3.0, 3.0, 1000)  # we'll be plotting in the interval [-3,3]x[-3,3]
x1 = x0
x0, x1 = np.meshgrid(
    x0, x1
)  # this produces two grids, one with the x0 coordinates and one with the x1 coordinates
z = k(
    np.stack((x0, x1))
)  # this computes a function (in this case g) over the stacked grids


# # do a contour plot
plt.contour(x0, x1, z, 50, cmap="RdGy")
# alterntative to contour plot: a surface plot
# (in 3-D, that you can rotate etc)
# to use it, comment the plt.contour line and uncomment the follwoing ones
# (note that we need some additional modules for this)

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# ax = fig.gca(projection="3d")  # prepare the figure to hold a 3-d plot
# ax.plot_surface(x0, x1, z, alpha=0.5, cmap=cm.coolwarm)  # 3-d plot
# plt.show()
# ~~~~~~~~~~~~~~~~~~~~~ ##

# Optimization, gradient descent version (TODO)
from scipy import optimize


def callback(xk):
    print(xk)
    return False


x0 = np.array([0.1, 0.23])
res = optimize.minimize(g, x0, callback=callback)

print(res)
