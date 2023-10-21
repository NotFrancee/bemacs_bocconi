import numpy as np


class Graph:
    def __init__(self, layers: int, nodes_per_layer: list) -> None:
        if not layers == len(nodes_per_layer):
            raise Exception("Length of layers must have same len as array of nodes")

        self.layers = layers

        max_nodes = np.max(nodes_per_layer)
        self.costs = np.zeros((layers - 1, max_nodes, max_nodes), dtype=int)
        self.nodes_per_layer = nodes_per_layer

        self.init_graph()

    def init_graph(self):
        nodes_per_layer, costs = self.nodes_per_layer, self.costs

        for i in range(self.layers - 1):
            source_nodes = nodes_per_layer[i]
            destination_nodes = nodes_per_layer[i + 1]

            weights = np.random.randint(1, 10, size=(source_nodes, destination_nodes))
            costs[i, :(source_nodes), :destination_nodes] = weights


def bottom_up(graph):
    pass
