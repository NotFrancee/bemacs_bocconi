# TSP solving algorithm
import numpy as np
import matplotlib.pyplot as plt


class TSP:
    def __init__(self, n_cities: int) -> None:
        if n_cities < 1:
            raise Exception("Cities must be more than 1")

        # initialize city coordinates
        self.n_cities = n_cities
        x = np.random.rand(n_cities)
        y = np.random.rand(n_cities)

        # initialize the distances
        self.dist = np.sqrt(
            (x - x.reshape(n_cities, 1)) ** 2 + (y - y.reshape(n_cities, 1)) ** 2
        )

        self.x, self.y = x, y
        self.init_config()

    def init_config(self):
        # choose the initial configuration
        route = np.random.permutation(self.n_cities)
        self.route = route

    def propose_move(self):
        # propose the move by switching the edges
        n_cities = self.n_cities

        # select two vertices at random
        e1 = np.random.choice(n_cities)
        while True:
            e2 = np.random.choice(n_cities)

            if e2 < e1:
                e1, e2 = e2, e1

            if e2 > e1 + 1:
                break

        return e1, e2

    def compute_delta_cost(self, move):
        e1, e2 = move
        route = self.route

        c1 = route[e1]
        c2 = route[e1 + 1]

        c3 = route[e2]
        c4 = route[(e2 + 1) % self.n_cities]

        prev_cost = self.dist[c1, c2] + self.dist[c3, c4]
        new_cost = self.dist[c1, c3] + self.dist[c2, c4]

        return new_cost - prev_cost

    def accept_move(self, move):
        # invert the order between e1 and e2
        e1, e2 = move
        self.route[e1 + 1 : e2 + 1] = self.route[e2:e1:-1]

    def display(self):
        plt.clf()
        route = self.route
        plt.scatter(self.x, self.y)

        plt.plot(self.x[route], self.y[route], color="orange")

        start = route[0]
        end = route[-1]
        plt.plot(self.x[[end, start]], self.y[[end, start]], color="orange")

        plt.pause(0.01)


def greedy(probl: TSP, n_iters: int, n_attempts: int):
    probl.display()

    best_probl = None

    for _ in range(n_iters):
        for _ in range(n_attempts):
            move = probl.propose_move()
            delta_cost = probl.compute_delta_cost(move)

            if delta_cost < 0:
                probl.accept_move(move)
                print(f"Accepted move {move}, new cost is WIP")

                probl.display()


t = TSP(20)
greedy(t, 1000)
