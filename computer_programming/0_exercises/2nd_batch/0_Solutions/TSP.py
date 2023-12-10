import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

class TSP:
    def __init__(self, n, seed = None):
        if not (isinstance(n, int) and n >= 4):
            raise Exception("n must be an int greater than 3")
        self.n = n

        ## Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)

        ## Random coordinates in [0,1)x[0,1)
        x = np.random.rand(n)
        y = np.random.rand(n)
        self.x, self.y = x, y

        ## Pre-compute the distances.
        xT = x.reshape((n,1))
        yT = y.reshape((n,1))
        dist = np.sqrt((x - xT)**2 + (y - yT)**2)
        self.dist = dist

        ## Allocate the memory for the route, then initialize it
        self.route = np.zeros(n, dtype=int)
        self.init_config()

    ## Initialize (or reset) the current configuration
    def init_config(self):
        n = self.n
        ## Remember the `[:]` for in-place assignment!
        self.route[:] = np.random.permutation(n)


    ## Plot the cities and the current configuration
    def display(self):
        x, y = self.x, self.y
        route = self.route

        plt.clf()
        ## Cities locations
        plt.plot(x, y, 'o')
        ## Route
        plt.plot(x[route], y[route], '-', c='orange')
        xcomeback = [x[route[-1]], x[route[0]]]
        ycomeback = [y[route[-1]], y[route[0]]]
        plt.plot(xcomeback, ycomeback, '-', c='orange')

        ## Pause to actually see something
        plt.pause(0.00001)

    ## Cost of the current configuration, computed from scratch
    def cost(self):
        n, route, dist = self.n, self.route, self.dist
        c = 0.0
        for e in range(n):
            city1 = route[e]
            city2 = route[(e+1) % n]
            c += dist[city1, city2]
        return c

    ## Propose a valid random move. Returns two edge indices to cross.
    def propose_move(self):
        n = self.n
        while True:
            e1 = np.random.randint(n)
            e2 = np.random.randint(n)
            if e1 > e2:
                e1, e2 = e2, e1
            if e1 != e2 and e1 + 1 != e2 and not (e1 == 0 and e2 == n-1):
                break
        move = (e1, e2)
        return move

    ## Modify the current configuration, accepting the proposed move
    def accept_move(self, move):
        ## Accepting a move is much easier (and computationally cheaper) in
        ## this case, compared to the swap paths version
        e1, e2 = move
        route = self.route
        route[e1+1:e2+1] = route[e2:e1:-1]

    ## Compute the extra cost of the move (new-old, negative means convenient)
    def compute_delta_cost(self, move):
        e1, e2 = move
        n, route, dist = self.n, self.route, self.dist
        ## Cities involved in the first (old) edge
        city11, city12 = route[e1], route[(e1+1) % n]
        ## Cities involved in the second (old) edge
        city21, city22 = route[e2], route[(e2+1) % n]
        ## Costs of the old edges
        d1_old = dist[city11, city12]
        d2_old = dist[city21, city22]
        c_old = d1_old + d2_old
        ## Costs of the new edges
        d1_new = dist[city11, city21]
        d2_new = dist[city12, city22]
        c_new = d1_new + d2_new
        ## Cost difference
        delta_c = c_new - c_old
        return delta_c

    ## Make an entirely independent duplicate of the current object.
    def copy(self):
        return deepcopy(self)
