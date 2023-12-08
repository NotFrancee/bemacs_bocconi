import numpy as np

# EXPLANATION
# list of rectangular matrices

# rows: connections of a node with the nodes in the next layer
# columns: n of nodes in the current layer

# data

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


class Graph:
    def __init__(self, graph: list[np.ndarray]):
        self.check_graph(graph)

        self.graph = graph
        self.n_nodes = self.get_n_nodes()

    # function to check if the graph is valid
    def check_graph(self, graph: list[np.ndarray]):
        # non-empty
        if len(graph) == 0:
            return False

        # elements are 2d arrays
        if not np.all(np.array([len(x.shape) for x in graph]) == 2):
            raise Exception("Graph elements are not 2d arrays")

        # shapes have the correct relationships
        is_correct = True
        for layer, next_layer in zip(graph[:-1], graph[1:]):
            if layer.shape[1] != next_layer.shape[0]:
                is_correct = False

        if not is_correct:
            raise Exception("Graph shapes are not correct")

        return True

    def get_n_nodes(self):
        return np.sum([layer.shape[0] for layer in self.graph])

    def layer_to_index(self, layer: int, inter_index: int):
        graph = self.graph
        if layer >= len(graph) or inter_index >= graph[layer].shape[0]:
            raise ValueError("index out of range")

        idx_layer = np.sum([layer.shape[0] for layer in graph[:layer]])

        return idx_layer + inter_index

    def index_to_layer(self, index: int):
        graph, n_nodes = self.graph, self.n_nodes
        nodes_per_layer = [layer.shape[0] for layer in graph]

        if index > n_nodes or index < 0:
            raise ValueError("index out of range")

        layer_idx = 0
        current_inter_idx = index
        while current_inter_idx >= nodes_per_layer[layer_idx]:
            current_inter_idx -= nodes_per_layer[layer_idx]
            layer_idx += 1

        return (layer_idx, current_inter_idx)

    def initialize_c(self):
        c = [np.zeros(layer.shape[0]) + np.inf for layer in self.graph]
        c.append(np.array([np.inf]))

        return c

    def __repr__(self):
        s = []

        for i, layer in enumerate(self.graph):
            s.append(f"layer {i}:")
            s.append(layer.__str__())

        return "\n".join(s)


def test_n_nodes():
    g = Graph(test_g1)
    assert g.get_n_nodes() == 9
    print("test_n_nodes passed")


test_n_nodes()


def test_layer_to_index():
    g = Graph(test_g1)

    n_nodes = g.get_n_nodes()

    for node_idx in range(n_nodes):
        coord = g.index_to_layer(node_idx)
        assert g.layer_to_index(*coord) == node_idx

    print("test_layer_to_index passed")


test_layer_to_index()


# solving using dynamic programming
def forward_pass(graph: Graph):
    g = graph.graph
    print(graph)

    # initialize c with the number of nodes
    c = graph.initialize_c()
    whence = [-np.ones(len(x), dtype=int) for x in c]

    # loop through the layers and then inside the layer
    c[0][0] = 0
    print(c)
    print(whence)

    # iterate through layers
    for l in range(1, len(c)):  # noqa
        nodes_in_layer = len(c[l])

        # iterate through the nodes
        for j in range(nodes_in_layer):
            # calculate cost
            c[l][j] = np.min(
                [c[l - 1][k] + g[l - 1][k, j] for k in range(len(c[l - 1]))]
            )

            # set whence
            whence[l][j] = np.argmin(
                [c[l - 1][k] + g[l - 1][k, j] for k in range(len(c[l - 1]))]
            )

        print("debug", nodes_in_layer)
        print(c[l])
        print(c[l - 1], g[l - 1])
        print(c[l - 1] + g[l - 1])
        assert np.array_equal(c[l], (c[l - 1] + g[l - 1]).squeeze())

    print(c)
    print(whence)
    return whence


def get_best_path(graph: Graph, whence: list):
    print("---RETRIEVING BEST PATH---")
    if whence[-1] == -1:
        return np.inf

    res = [0, graph.n_nodes]

    prev_idx = whence[-1][-1]
    layer = len(whence) - 2

    while prev_idx != -1:
        res.insert(1, graph.layer_to_index(layer, prev_idx))
        layer -= 1
        prev_idx = whence[layer][prev_idx]

    return res


graph = Graph(test_g1)
whence = forward_pass(graph)
path = get_best_path(graph, whence)

print(path)
