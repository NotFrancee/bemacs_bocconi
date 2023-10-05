import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from scipy import special

# create a function that does this in as many dimensions as you want
# 1. redo in 3D, 4D, ...; (look up the formula on wikipedia)
# study the scaling of the variance of the estimate with the number of points

# volume of a unit n-ball:
# V(n) = pi^(n/2) / gamma(n/2 + 1)
# we are only going to select one sector of the n dimensional cube, so we have to divide by 2^n
# thus pi_est = (2^(n) * gamma(n/2 + 1) * ninside / npoints)^(2/n)

def estimate_pi(n, dims=2):
    assert(dims >= 2)
    ps = rnd.rand(dims,n)
    ninside = ((ps**2).sum(axis=0) <= 1).sum()
    return np.exp((2/dims) * (np.log(special.gamma(dims/2 + 1)) + np.log(ninside / n)) + 2*np.log(2))

# study the behavior of this function as n and dims are (independenlty) increased.
# estimate the variance from repeated trials    


nseeds = 10
ndim = 2
npoints = np.logspace(2,8,10, dtype=int)
nmeans = []
nstds = []
for n in npoints:
    res = np.array([estimate_pi(n, dims=ndim) for i in range(nseeds)])
    rmean, rstd = res.mean(), res.std()
    nmeans.append(rmean) # check that it converges to better and better estimates of pi (on average)
    nstds.append(rstd) #check that this goes down as 1/sqrt(n)
    

plt.clf()  
plt.subplot(1,2,1)
plt.errorbar(ndims,nmeans,yerr=nstds)
plt.axhline(y=np.pi, linestyle='--')
plt.title("npoints")


nseeds = 10
n = 10000
ndims = np.linspace(2,10,10, dtype=int)
means = []
stds = []
for ndim in ndims:
    res = np.array([estimate_pi(n, dims=ndim) for i in range(nseeds)])
    rmean, rstd = res.mean(), res.std()
    means.append(rmean)
    stds.append(rstd) # it gets worse!!! why? think about it
  
plt.subplot(1,2,2)
plt.errorbar(ndims,means,yerr=stds)
plt.axhline(y=np.pi, linestyle='--')
plt.title("ndims")
plt.suptitle("Scaling analysis")




