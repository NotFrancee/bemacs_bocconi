from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class TSP:
    """Class that handles the Travelling Sales Man Problem"""

    def __init__(self, n, seed=None):
        """Inits the TSP Problem

        * n: number of cities in the problem
        """

        # check if n is an int greater than 4
        if not (isinstance(n, int) and n >= 4):
            raise Exception("n needs to be an int larger or equal than 4")

        if seed is not None:
            np.random.seed(seed)

        # store n for later use
        self.n = n

        # create the coordinates of the cities
        x = np.random.rand(n)
        y = np.random.rand(n)

        # stores them in the object
        self.x, self.y = x, y

        self.route = np.zeros(n, dtype=int)
        self.init_config()

        # USING BROADCASTING
        # the first difference will create a table with all the differences
        distance = np.sqrt((x - x.reshape(-1, 1)) ** 2 + (y - y.reshape(-1, 1)) ** 2)
        self.distance = distance

    # instead of having it as a function make it a method of the Cities object
    def dist(self, city1, city2):
        """Computes the distance between two cities"""
        x, y = self.x, self.y

        d = np.sqrt((x[city2] - x[city1]) ** 2 + (y[city2] - y[city1]) ** 2)

        return d

    def init_config(self):
        """Create a first random hamiltonian path. Returns the indices of the cities."""

        # gets the number of nodes (the number of cities)
        n = self.n

        # create a random permutation of indices of the cities
        # np.random.permutation gives a permutation of numbers up to n
        # assigning like this changes the pointer
        # self.route = np.random.permutation(n)

        # this reassigns the elements in the array instead
        #   of having to find new space in memory for it
        self.route[:] = np.random.permutation(n)

        # we do this here and resassign so that when we repeat init_config
        #   we use always the same sapce of memory
        # we call __init__ once but init_config many times,
        #   so we are going to always target the same point in memory

    def display(self):
        """Displays a route and the dots of the cities using matplotlib

        Uses .pause() method to show how the plot evolves for each iteration
        """

        # clear the plot always
        plt.clf()

        x, y, route = self.x, self.y, self.route

        # plot the routes
        plt.plot(x[route], y[route], "-", color="orange")

        # plot the last edge
        comeback = [route[-1], route[0]]
        plt.plot(x[comeback], y[comeback], color="orange")

        # plot the nodes (after to put them on the top)
        plt.plot(x, y, "o", color="black")

        plt.show()
        # plt.pause because otherwise it renders all the plots at the end.
        # this way we can see the route evolve
        plt.pause(0.01)

    def cost(self):
        """Calculates all the distances and sums them to get to the final cost"""

        dist = 0.0
        route = self.route

        for i in range(self.n):
            city1 = route[i]
            city2 = route[(i + 1) % self.n]

            dist += self.distance[city1, city2]

        return dist

    # we split the propose_move in two functions
    def propose_move(self):
        n = self.n

        # select the two edges randomly
        while True:
            e1 = np.random.randint(n)
            e2 = np.random.randint(n)

            if e1 > e2:
                e1, e2 = e2, e1

            # we also want to avoid to choose the first and last index, otherwise it's just going to invert the route

            # we want to avoid e1 = e2 and that the two edges are already aadjacent (otherwise nothing will change)
            if e2 > e1 + 1 and not (e1 == 0 and e2 == n - 1):
                break

        move = (e1, e2)
        return move

    def accept_move(self, move):
        """Accepts the move, reversing the order of the vertices between the
        ones specified in move

        Args:
            move (tuple): (e1, e2)
        """
        e1, e2 = move
        route = self.route
        # we revert the order of the indices in that segment of the configuration
        route[e1 + 1 : e2 + 1] = route[e2:e1:-1]  # reason about the indices choice

    def compute_delta_cost(self, move):
        """We compute the delta cost between the two routes without having to recalculate the whole cost

        NEW: we now want to compute just the delta cost, not the whole cost
        old cost = cost of all edges + cost(e1) + cost(e2)
        new cost = cost of all edges + cost(E1) + cost(E2) where E1, E2 are the new edges with the vertices switched
        """

        # temporarily: we duplicate the whole problem and accept the move in the new problem
        # old_c = self.cost()

        # new_tsp = self.copy()  # create a new problem
        # new_tsp.accept_move(move)  # accept the move in the new problem
        # new_c = new_tsp.cost()  # compute the cost for the new problem

        # delta_c_slow = new_c - old_c

        route = self.route
        e1, e2 = move
        city11, city12 = route[e1], route[e1 + 1]  # cause e1 < e2
        city21, city22 = route[e2], route[(e2 + 1) % self.n]

        # instead of calculating the distances every time, we want to store them and access them from a matrix
        # c_e1 = self.dist(city11, city12)
        # c_e2 = self.dist(city21, city22)
        # c_E1 = self.dist(city11, city21)
        # c_E2 = self.dist(city12, city22)

        c_e1 = self.distance[city11, city12]
        c_e2 = self.distance[city21, city22]
        c_E1 = self.distance[city11, city21]
        c_E2 = self.distance[city12, city22]

        old_c = c_e1 + c_e2
        new_c = c_E1 + c_E2

        delta_c = new_c - old_c

        # assert(abs(delta_c_slow - delta_c) < 1e-10) # check that the two lead to the same result.
        # since floating point operations lead to slightly different results in computers make just sure that the distance is small

        return delta_c

    def copy(self):
        # we could optimize it more, i.e. the coordinates of the cities will not change so
        #   we just need to copy the reference to the route
        return deepcopy(self)
