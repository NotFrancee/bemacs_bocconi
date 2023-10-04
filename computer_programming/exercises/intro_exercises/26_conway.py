"""
Implement Conway’s game of life. Write a function that takes an initial configuration as a table of zeros and ones
(could be bools or ints) and a number of steps  n . The function runs the game of life for  n  steps and returns the
final configuration.
Note: in the original game of life rules, it is supposed that the grid is infinite; instead, for simplicity, you should use
a finite grid instead, and just apply the standard rules even at the borders — which means that the border cells
have only 5 neighbors, and the corner cells only 3.

Tip: for a more straightforward implementation, you’ll need 2 arrays, one for the current configuration and one for
the next one; after each step overwrite the current with the next. For an advanced implementation, you may try a
version that computes the table of neighbors of each cell first, and then uses comparison operators and logic
boolean filters (the operations  & ,  |  and  ~ ) to update the configuration.
Plot the configuration at each step as an image (using  imshow ). Remember to use  plt.clf()  before each new
plot or your plots will rapidly become slow (can you guess why?). This works best if you don’t use inline plotting
(see note in exercise 18) and if you put  plt.pause(0.05)  after each plot in order to actually see what’s going on.
Tip: start with small grids and simple configurations when debugging, and check one step at a time. Here is for
example an interesting self-sustaining pattern, called a glider:
"""

import numpy as np
from nptyping import NDArray
from copy import deepcopy
import matplotlib.pyplot as plt


class Config:
    def __init__(self, matr) -> None:
        self.matr = matr

    def get_neighbor(self, i, j):
        return self.matr[i - 1 : i + 2, j - 1 : j + 2]

    def live(self, i, j):
        self.matr[i, j] = 1

    def kill(self, i, j):
        self.matr[i, j] = 0

    def copy(self):
        return deepcopy(self)

    def display(self):
        plt.clf()
        plt.imshow(self.matr)
        plt.pause(0.05)


def ex26_gameoflife(config: NDArray, n: int):
    current_config = Config(config)
    next_config = None

    # n iterations
    for _ in range(n):
        for i in range(config.shape[0]):
            for j in range(config.shape[1]):
                next_config = current_config.copy()
                neighbors = current_config.get_neighbor(i, j)

                is_live = current_config.matr[i, j]
                live_neighbors = np.sum(neighbors) - is_live

                if live_neighbors > 0:
                    print(f"{i} - {j}: \n", neighbors)
                    print(f"found {live_neighbors} live neighbors")

                if live_neighbors == 3:
                    print("found 3 live neighbors at ")
                    next_config.live(i, j)
                elif live_neighbors > 3 or live_neighbors < 2:
                    next_config.kill(i, j)

                current_config = next_config.copy()
                current_config.display()

    print(current_config.matr)
    print(current_config.get_neighbor(3, 3))
    return current_config
    # create a neighbors table: a n * 9 array where you store
    # in the first element the key cell
    # in the next 8 the neighbors

    # for any cell
    #     if < 2 neighbors: death
    #     if dead and 3 live neighbors: live
    #     else: live


n = 20
c = np.zeros((n, n), dtype=np.int8)
glider = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
m0, m1 = glider.shape
c[:m0, :m1] = glider
ex26_gameoflife(c, 50)
