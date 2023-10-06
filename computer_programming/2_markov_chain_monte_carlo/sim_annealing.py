import numpy as np


def boltzmann(energies, beta):
    # from the energies we compute the probabilities

    # we use the formula (+ broadcasting)
    w = np.exp(-beta * energies)

    # we normalize
    return w / np.sum(w)


# investigating the role of beta
# beta = 0 => uniform distribution
beta0 = 0
beta1 = 0.1
beta2 = 1.0
energies = np.array([25, 13, 2, 4, 9, 11])
p = boltzmann(energies, beta2)
print(p)
