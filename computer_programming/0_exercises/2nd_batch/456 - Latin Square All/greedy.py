import numpy as np
from typing import Union
from lsquare_v1 import LatinSquare as LS1
from lsquare_v2 import LatinSquare as LS2
from lsquare_v3 import LatinSquare as LS3
from sudoku_v1 import Sudoku as S1
from sudoku_v2 import Sudoku as S2


def greedy(
    probl: Union[LS1, LS2, LS3, S1, S2],
    repeats=1,
    num_iters=100,
    seed=None,
    debug: bool = False,
):
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

        # store the configuration inside the object
        # and not in the greedy function
        probl.init_config()
        cx = probl.cost()

        for t in range(num_iters):
            # we make propose move a method of
            # the problem so that you can specify it in the problem object
            # y = probl.propose_move(x)
            move = probl.propose_move()

            # now we pass move so that we optimize
            # the calc of the delta cost
            delta_c = probl.compute_delta_cost(
                move, debug_delta_cost=True if debug else False
            )

            if delta_c <= 0:
                # accepted!
                probl.accept_move(move)
                # cx = cy
                cx += delta_c

                # print the new cost
                print(f"\tmove accepted, c = {cx}, t = {t}")

            if cx == 0:
                break

        # stopping criteria -> max number of iterations reached
        print(f"final cost: {cx}")

        if cx < best_cost:
            best_cost = cx
            best_probl = probl.copy()  # use the method defined in the object

    best_probl.display()
    print(f"Best cost: {best_cost}")
    print(best_probl)

    return best_cost, best_config
