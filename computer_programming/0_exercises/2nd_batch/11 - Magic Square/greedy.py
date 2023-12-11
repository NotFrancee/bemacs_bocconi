import numpy as np
from magicsquare import MagicSquare


def greedy(probl: MagicSquare, repeats=1, num_iters=100, seed=None):
    """Greedy algorithm
    * probl: the problem to solve
    * repeates: how many times you repeat the algorithm
    * num_iters: how many iterations per run
    * seed: for consistent results
    """

    best_config = None
    best_cost = np.infty
    best_probl = probl

    for _ in range(repeats):
        if seed is not None:
            np.random.seed(seed)

        # store the configuration
        # inside the object and not in the greedy function
        probl.init_config()
        cx = probl.cost()

        for t in range(num_iters):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)

            if delta_c <= 0:
                # accepted!
                probl.accept_move(move)
                cx += delta_c

                # print the new cost
                # print(f"\tmove accepted, c = {cx}, t = {t}")

        # stopping criteria -> max number of iterations reached
        # print(f"final cost: {cx}")

        if cx < best_cost:
            best_cost = cx
            best_probl = probl.copy()  # use the method defined in the object

    best_probl.display()
    print(f"Best cost: {best_cost}")
    print(best_probl)

    return best_cost, best_config
