import numpy as np
import matplotlib.pyplot as plt


def mc_asset(S0, r, sigma, T, Nsteps, Nrep):
    SPATH = np.zeros((Nrep, 1 + Nsteps))
    SPATH[:, 0] = S0
    dt = T / Nsteps
    nudt = (r - 0.5 * sigma**2) * dt
    sidt = sigma * np.sqrt(dt)

    for i in range(0, Nrep):
        for j in range(0, Nsteps):
            SPATH[i, j + 1] = SPATH[i, j] * np.exp(nudt + sidt * np.random.normal())
    return SPATH


S0 = 100
K = 110
CallOrPut = "call"
r = 0.03
sigma = 0.25
T = 0.5
Nsteps = 1000
Nrep = 100
SPATH = mc_asset(S0, r, sigma, T, Nsteps, Nrep)

plt.figure(figsize=(10, 8))
for i in range(len(SPATH)):
    plt.plot(SPATH[i])
plt.xlabel("Numbers of steps")
plt.ylabel("Stock price")
plt.title("Monte Carlo Simulation for Stock Price")
plt.show()
