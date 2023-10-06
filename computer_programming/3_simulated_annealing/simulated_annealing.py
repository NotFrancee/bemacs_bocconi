"""Class for the greedy algorithm"""
import numpy as np


def accept_with_prob(delta_cost: float, beta: float):
    """Return true with the correct probability, false otherwise"""
    # see slides
    # if delta_cost <= 0 we are always accepting the move
    if delta_cost <= 0:
        return True

    # now delta_cost is surely >= 0

    # if beta is infty we are in the greedy case
    # the delta_cost is <= 0 we are never accepting the move
    if beta == np.infty:
        return False

    # delta_cost>0 => the exponential will always be < 1 => discard the min
    # prob = min(1, np.exp(-beta * delta_cost))
    prob = np.exp(-beta * delta_cost)

    return np.random.rand() < prob  # it is already a boolean value


# remove repeats since the algo is just running once
# add betas and annealing_steps
# change n_iters to mcmc_steps
def simann(probl, beta0=0.1, beta1=10.0, annealing_steps=10, mcmc_steps=100, seed=None):
    """see code from bb"""

    if seed is not None:
        np.random.seed(seed)

    # initialize the config just once at the top
    probl.init_config()
    cx = probl.cost()
    best_cost = cx
    best_probl = probl
    print(f"initial cost is {cx:.5f}, starting route is {probl.route}")

    # define the arrays of betas
    # we put -1 since we want the final item to be infinity
    betas = np.zeros(annealing_steps)
    betas[:-1] = np.linspace(beta0, beta1, annealing_steps - 1)
    betas[-1] = np.infty

    # loop to slowly decrease the temperatures
    for beta in betas:
        # we do not init every time
        # probl.init_config()

        # store the number of accepted moves to tweak params later
        accepted_moves = 0

        for _ in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)

            # now if delta_c<=0, we have to make this probabilistically
            # we run the code with p given by the distribution we've seen before
            # if delta_c <= 0:
            # we are now accepting a move with a probabilistic rule
            if accept_with_prob(delta_c, beta):
                probl.accept_move(move)
                cx += delta_c
                accepted_moves += 1

            if cx < best_cost:
                best_cost = cx
                best_probl = probl.copy()

        # we are at the end of a given beta
        print(
            f"beta={beta}, current_cost={cx}, best_cost={best_cost}, acc_freq={accepted_moves/mcmc_steps*100:.2f}%"
        )

    best_probl.display()
    print(f"Best cost: {best_cost}")

    return best_cost, best_probl
