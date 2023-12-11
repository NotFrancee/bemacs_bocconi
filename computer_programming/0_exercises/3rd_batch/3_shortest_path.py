from typing import Optional
import numpy as np


test_g0 = [np.array([[3.3]])]

test_g1 = [
    np.array(
        [
            [6.0, 8.0, 13.0],
        ]
    ),
    np.array([[9.0, 15.0, np.inf], [8.0, 10.0, 12.0], [np.inf, 8.0, 7.0]]),
    np.array(
        [
            [15.0, np.inf],
            [20.0, 8.0],
            [np.inf, 7.0],
        ]
    ),
    np.array([[3.0], [4.0]]),
]


class DirAcyclicGraph:
    def __init__(self, graph: list[np.ndarray]):
        self.check_graph(graph)
        self.graph = graph

    def check_graph(self, g: list[np.ndarray]):
        if not len(g) > 0:
            raise Exception("empty list")

        if not np.all([a.ndim == 2 for a in g]):
            raise Exception("arrays must have 2 dimensions")

        for i, layer in enumerate(g):
            if i == len(g) - 1:
                continue

            if not layer.shape[1] == g[i + 1].shape[0]:
                raise Exception("shapes do not match")

        return True

    def compute_nodes(
        self,
        graph: Optional[list[np.ndarray]] = None,
    ):
        if graph is None:
            graph = self.graph

        return np.sum([layer.shape[0] for layer in graph])

    def layer2index(
        self,
        layer: int,
        inter_index: int,
    ) -> int:
        graph = self.graph

        # checks
        if layer >= len(graph):
            raise ValueError("layer index out of bounds")

        if inter_index >= graph[layer].shape[0]:
            raise ValueError("inter_index out of bounds")

        # layer and inter_index computation
        layer_index = self.compute_nodes(graph[:layer])

        return int(layer_index + inter_index)

    def index2layer(
        self,
        i: int,
    ):
        graph = self.graph

        if i >= self.compute_nodes(graph):
            raise ValueError("index out of bounds")

        layer_i = 0
        inter_index = i

        while graph[layer_i].shape[0] <= inter_index:
            inter_index -= graph[layer_i].shape[0]
            layer_i += 1

        return (layer_i, inter_index)

    def gen_cost_matrix(
        self,
        fill=np.inf,
        typ=float,
    ):
        graph = self.graph

        c = [np.full(layer.shape[0], fill, dtype=typ) for layer in graph]
        c.append(np.full(graph[-1].shape[1], np.inf))

        return c


def test_graph():
    g = DirAcyclicGraph(test_g1)

    assert g.check_graph(g.graph)

    for node_i in range(g.compute_nodes()):
        node_coords = g.index2layer(node_i)

        node_i_new = g.layer2index(node_coords[0], node_coords[1])

        assert node_i == node_i_new


test_graph()


def shortest_path(g: DirAcyclicGraph):
    c = g.gen_cost_matrix()

    # base case
    c[0][0] = 0

    whence = g.gen_cost_matrix(fill=0, typ=int)
    whence[0][0] = -1

    graph = g.graph

    # forward pass
    for layer_i in range(1, len(graph) + 1):
        n_nodes = len(c[layer_i])
        n_nodes_prev = len(c[layer_i - 1])

        costs = c[layer_i - 1].reshape(-1, 1) + graph[layer_i - 1]
        print(costs)
        print(costs.argmin(axis=0))
        print(costs[costs.argmin(axis=0)])

        for j in range(n_nodes):
            costs = [
                c[layer_i - 1][k] + graph[layer_i - 1][k, j]
                for k in range(n_nodes_prev)
            ]

            min_i = np.argmin(costs)

            c[layer_i][j] = costs[min_i]
            whence[layer_i][j] = min_i

    print(c)
    print(whence)

    path = np.zeros(len(whence), dtype=int)

    # i = len(graph)
    # while whence[i] != -1:
    #     node_i = g.layer2index(i, whence[i])
    #     print(node_i)
    #     print(whence[i])
    #     i -= 1


g = DirAcyclicGraph(test_g1)
shortest_path(g)
