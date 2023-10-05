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

        route = self.route
        e1, e2 = move
        city11, city12 = route[e1], route[e1 + 1]  # cause e1 < e2
        city21, city22 = route[e2], route[(e2 + 1) % self.n]

        c_e1 = self.distance[city11, city12]
        c_e2 = self.distance[city21, city22]
        c_E1 = self.distance[city11, city21]
        c_E2 = self.distance[city12, city22]

        old_c = c_e1 + c_e2
        new_c = c_E1 + c_E2

        delta_c = new_c - old_c

        return delta_c

    def copy(self):
        return deepcopy(self)


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

        # probl.display()

        print(f"initial cost is {cx:.5f}, starting route is {probl.route}")

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

    return best_cost, best_config


"""
So now to produce a move you just need to choose two indices at random between  0  (included) and  n
(excluded). (Make sure they’re different.) Call the indices  1  and  j . The move would swap the two cities at
positions  i  and  j  in the route.

Then, you need to compute the cost of such move. Start out by doing it in the most trivial way, as in the debug
code which is in  greedy : get the old cost with the  cost  function; copy the problem into a new one, swap the
cities on the copy and get the new cost, and return the cost difference. This is inefficient but it’s easy to do and it
works.

Then compute the cost more efficiently: only a small part of the route is affected. Note that this is more difficult
than in the previous cross-links case, mainly because there are more terms involved. You also need to consider
different possibilities, depending on the chosen cities. Analyze the problem with small examples (8 cities or so),
consider several situations, etc. Make sure that your new code gives you the same result as the trivial one, except
at most for floating-point approximations. You can also use the  debug_delta_cost  option in the solver. Note that
it is very easy to get it wrong. It’s likely that you will get it right a few times and then find a case when it’s wrong. If
so, try to understand why and fix it. Keep going until you have fixed all the cases, then you can abandon the trivial
code.

Experiment with this new move scheme. You can try to compare the two, city-swap vs cross-links, and see which
is better. Try different problem sizes and see how things change as you increase the size
"""


class TSP_SC(TSP):
    def propose_move(self):
        # select two indices at random to swap
        while True:
            c1 = np.random.randint(self.n)
            c2 = np.random.randint(self.n)

            if c2 < c1:
                c1, c2 = c2, c1

            if c1 != c2:
                break

        move = (c1, c2)
        return move

    def compute_delta_cost(self, move):
        # efficient cost
        i_c1, i_c2 = move  # indices of the city

        c1, c2 = self.route[[i_c1, i_c2]]

        c1_prev = self.route[(i_c1 - 1) % self.n]
        c1_next = self.route[(i_c1 + 1) % self.n]

        c2_prev = self.route[(i_c2 - 1) % self.n]
        c2_next = self.route[(i_c2 + 1) % self.n]

        print(f"current route: {self.route}")
        print(
            f"new method:\n\told cost: {c1_prev}-{c1}-{c1_next} + {c2_prev}-{c2}-{c2_next}"
        )
        print(f"\tnew cost: {c1_prev}-{c2}-{c1_next} + {c2_prev}-{c1}-{c2_next}")

        old_cost = (
            self.distance[c1_prev, c1]
            + self.distance[c1, c1_next]
            + self.distance[c2_prev, c2]
            + self.distance[c2, c2_next]
        )

        print(f"indices: c1 {i_c1}, c2 {i_c2}")
        if i_c2 == i_c1 + 1:
            # then also c_2 prev and c1 next get swapped
            print("\tedge case detected")
            c1_next = self.route[i_c1]
            c2_prev = self.route[i_c2]

        if i_c2 == self.n - 1 and i_c1 == 0:
            c1_prev = self.route[i_c1]
            c2_next = self.route[i_c2]

        print(f"\tnew cost: {c1_prev}-{c2}-{c1_next} + {c2_prev}-{c1}-{c2_next}")

        new_cost = (
            self.distance[c1_prev, c2]
            + self.distance[c2, c1_next]
            + self.distance[c2_prev, c1]
            + self.distance[c1, c2_next]
        )

        delta_cost = new_cost - old_cost

        # more efficient method
        if i_c2 == i_c1 + 1:  # or the difference is 1 to capture the ends
            edges = self.route[i_c1 - 1 : i_c2 + 2]

            c1_prev = self.route[i_c1 - 1]
            c1 = self.route[i_c1]
            c2 = self.route[i_c2]
            c2_next = self.route[(i_c2 + 1) % self.n]

        # if there are points in between consider them in the calculation
        else:
            # all of the above, plus
            c1_next = self.route[i_c1 + 1]
            c2_prev = self.route[i_c2 - 1]

            # if c2 prev is c1, then the indices are consecutive

        return delta_cost

    def accept_move(self, move):
        c1, c2 = move

        self.route[[c1, c2]] = self.route[[c2, c1]]


probl = TSP_SC(8)
greedy(probl, 1, 1000)
