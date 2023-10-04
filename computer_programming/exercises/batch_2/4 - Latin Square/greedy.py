import numpy as np


def greedy(probl, repeats=1, num_iters=100, seed=None):
    """Greedy algorithm
    * probl: the problem to solve
    * repeates: how many times you repeat the algorithm
    * num_iters: how many iterations per run
    * seed: for consistent results
    """

    best_config = None
    best_cost = np.infty

    for _ in range(repeats):
        if seed is not None:
            np.random.seed(seed)

        # store the configuration inside the object and not in the greedy function
        probl.init_config()
        cx = probl.cost()

        for t in range(num_iters):
            # we make propose move a method of the problem so that you can specify it in the problem object
            # y = probl.propose_move(x)
            move = probl.propose_move()

            # now we pass move so that we optimize the calc of the delta cost
            delta_c = probl.compute_delta_cost(move)

            if delta_c <= 0:
                # accepted!
                probl.accept_move(move)
                # cx = cy
                cx += delta_c

                # print the new cost
                print(f"\tmove accepted, c = {cx}, t = {t}")

        # stopping criteria -> max number of iterations reached
        print(f"final cost: {cx}")

        if cx < best_cost:
            best_cost = cx
            # copying the object straight away is not correct though!
            # this assigns the reference to the object, and when later on the
            #   probl gets initialized again, the best_probl will point to the new
            #   problem
            # but by just doing copy,there are arrays in the object that are references themselves,
            #   so those references will still point to the arrays from the iniitla obj (which will chang )

            # hence there are different types of copy, shallow (depth = 1, leads to the problem described above)
            #   we are going to use deep copy which solves the problem we've described
            best_probl = probl.copy()  # use the method defined in the object

    best_probl.display()
    print(f"Best cost: {best_cost}")
    print(best_probl)

    return best_cost, best_config
