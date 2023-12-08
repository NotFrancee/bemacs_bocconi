"""Runs the greedy algorithm on TSP
"""

from TSP import TSP
from simulated_annealing import simann

CITIES = 100
BETA0 = 3
BETA1 = 30
ANNEALING_STEPS = 40
MCMC_STEPS = 10000  # you need at least 100

probl = TSP(CITIES, seed=123456)
simann(probl, BETA0, BETA1, ANNEALING_STEPS, MCMC_STEPS)

"""
We start with
BETA0 = 0.1
BETA1 = 3.0

and see the accepted frequency, whichis 95 at the beginning

try again with
BETA0 = 3
BETA1 = 30

now in the first step the freq is 50% which is ok with our rule
we now look at how the probability is coming down.
we expect the freqs to go down exponentially.
the temperatures should be good but still it is not optimized.
what is the problem? the mcmc_steps. so you either add more steps at each temperature or increase the number of temps

we now try with
ANNEALING_STEPS = 40
MCMC_STEPS = 10000

we can try better. if the parameters are good, the final cost should not depend on the initial configuration.
now we have cost = 8.24, let's run the algo some more times.
8.6
8.68
8.4
8.2

the cost changes so we should improve the number of steps. the spread should be low between different iterations
(you have to try with the same initial config though).

if we try now we get very little spread between the different tries
"""
