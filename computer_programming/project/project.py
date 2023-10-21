from generate_data import generate_data
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.axes import Axes
import numpy as np


class Project:
    SEED = "3185382"

    def __init__(self, n: int) -> None:
        if not (isinstance(n, int)):
            raise Exception("n must be an int")

        self.n = n
        self.data = generate_data(n, self.SEED)

    def __repr__(self) -> str:
        return str(self.data)

    def display(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax: Axes
        data, n = self.data, self.n
        X = np.arange(n)
        Y = np.arange(n)
        X, Y = np.meshgrid(X, Y)

        surf = ax.plot_surface(
            X, Y, data, cmap=cm.get_cmap("coolwarm"), linewidth=0, antialiased=False
        )

        fig.colorbar(surf, shrink=0.5, aspect=5)
        print("showing graph")
        plt.show()


def test():
    p = Project(100)
    print(p)
    p.display()


test()
