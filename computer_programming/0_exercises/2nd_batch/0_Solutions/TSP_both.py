import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

# TSP

## This file defines both the cross-links and the swap-cities versions of TSP.
## It avoids duplicating code by using inheritance.
## This means that for example you can change the `display` method for both
## classes at once.

## Base class: nothing related to Simulated Annealing is here.
## It's basically the class we created during the lecture, except:
##   1. It doesn't have the methods which deal with the moves
##   2. There is a small change in the copy() method
class TSP_Base:
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

    ## Make an entirely independent duplicate of the current object.
    def copy(self):
        return deepcopy(self)


## Cross-links version: the one we've seen in class
class TSP_CL(TSP_Base):
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


## Swap-cities version (i.e. let's make it worse...)
class TSP_SC(TSP_Base):
    ## Propose a valid random move. We want to propose the "swap two random
    ## cities" type of move. We will encode a move with just two indices, in
    ## a tuple, representing two positions *in the route*.
    def propose_move(self):
        n = self.n
        while True:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i > j:
                i, j = j, i
            ## In this version we only need to avoid that the two are the same. We
            ## don't need to check if they are neighbors: that is a perfectly valid
            ## move in this case.
            if i != j:
                break
        move = (i, j)
        return move

    ## Modify the current configuration, accepting the proposed move
    def accept_move(self, move):
        ## Accepting a move is much easier (and computationally cheaper) in
        ## this case, compared to the cross-links version
        i, j = move
        route = self.route

        ## Swap the cities at the given posistions in the route
        route[[i,j]] = route[[j,i]]

        ## Alternative version:
        # route[i], route[j] = route[j], route[i]

    ## Compute the extra cost of the move (new-old, negative means convenient)
    def compute_delta_cost(self, move):
        ## Consider that with our move only a few links of the route change.
        ## The number depends on whether the two cities are neighbors though.
        ## If they are not, then we are changing 4 links with another 4 links.
        ## If they are, we are changing 2 links with another 2.

        i, j = move
        route, dist = self.route, self.dist
        n = self.n

        ## First let's do the non-neighbors case (NOTICE THE `(0,n-1)` CASE!!)
        if j - i > 1 and (i,j) != (0,n-1):
            ## Remove 4 old links, add 4 new links:
            ## old_cost = (i-1 -> i) + (i -> i+1) + (j-1 -> j) + (j -> j+1)
            ## new_cost = (i-1 -> j) + (j -> i+1) + (j-1 -> i) + (i -> j+1)
            city_ip = route[(i-1) % self.n]
            city_i  = route[i]
            city_in = route[(i+1) % self.n]

            city_jp = route[(j-1) % self.n]
            city_j  = route[j]
            city_jn = route[(j+1) % self.n]

            di_old = dist[city_ip, city_i] + dist[city_i, city_in]
            dj_old = dist[city_jp, city_j] + dist[city_j, city_jn]

            di_new = dist[city_jp, city_i] + dist[city_i, city_jn]
            dj_new = dist[city_ip, city_j] + dist[city_j, city_in]

            ## The backslashes are used to break up one line of code into two
            c_old = di_old + dj_old
            c_new = di_new + dj_new

        ## Then there is the neighbors case j == i + 1
        else:
            ## We want `i` to be the antecedent of `j` so we must take care
            ## of the (0,n-1) case, again...
            if (i,j) == (0,n-1):
                i, j = n-1, 0

            ## Remove 2 old links, add 2 new links (the i->j link becomes j->i but
            ## we don't care about that because our problem is symmetric):
            ## old_cost = (i-1 -> i) + (j -> j+1)
            ## new_cost = (i-1 -> j) + (i -> j+1)
            city_ip = route[(i-1) % self.n]
            city_i  = route[i]
            city_j  = route[j]
            city_jn = route[(j+1) % self.n]

            c_old = dist[city_ip, city_i] + dist[city_j, city_jn]

            c_new = dist[city_ip, city_j] + dist[city_i, city_jn]

        ## Cost difference
        delta_c = c_new - c_old
        return delta_c
