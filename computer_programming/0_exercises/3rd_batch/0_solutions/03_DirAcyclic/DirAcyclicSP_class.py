## This is the solution to ex. 3, using a class to store
## the graph.
## The class is called DAWLGraph. All the functions except
## the one computing the shortest path and a couple
## of others have become class methods.
## The final function `shortest_path_cl` is equivalent
## to `shortest_path3` of "DirAcyclicSP.py".

import numpy as np

## Same code as DirAcyclicSP, except that the graph is stored in a class.
## Also, `layer2ind` was renamed `node` and `ind2layer` was renamed `indices`.

test_g0 = [np.array([[3.3]])]

test_g1 = [
        np.array([
                [   6.0,    8.0,   13.0],
                ]),
        np.array([
                [   9.0,   15.0, np.inf],
                [   8.0,   10.0,   12.0],
                [np.inf,    8.0,    7.0]
                ]),
        np.array([
                [  15.0, np.inf],
                [  20.0,    8.0],
                [np.inf,    7.0],
                ]),
        np.array([
                [   3.0],
                [   4.0]
                ])
        ]


def filled(n, x, typ=float):
    a = np.zeros(n, dtype=typ)
    a.fill(x)
    return a

def check_graph(g):
    if not isinstance(g, list):
        raise Exception("g must be a list")
    for gl in g:
        if not isinstance(gl, np.ndarray) or gl.ndim != 2:
            raise Exception("each entry of g must be a 2-d np.array")
    nl = len(g)
    if nl < 1:
        raise Exception("at least 2 layers expected in graph g")
    if g[0].shape[0] != 1:
        raise Exception("g[0] must have only 1 row")
    if g[-1].shape[1] != 1:
        raise Exception("g[1] must have only 1 column")
    for l in range(1, nl):
        if g[l-1].shape[1] != g[l].shape[0]:
            raise Exception("shape mismatch between layers", l-1, "and", l, "in g")
    return True

## A "Direct-Acyclic-Weighted-Layered Graph"
class DAWLGraph:
    def __init__(self, g):
        check_graph(g)
        self.g = g
        self.n = self.num_nodes()

    def __getitem__(self, l):
        return self.g[l]

    def __len__(self):
        return len(self.g)

    def num_nodes(self):
        return sum([gl.shape[0] for gl in self.g]) + 1

    def indices(self, i):
        g = self.g
        if not 0 <= i < self.n:
            raise Exception("out-of-bounds index i")
        l, j = 0, i
        while l < len(g) and j >= g[l].shape[0]:
            j -= g[l].shape[0]
            l += 1
        if l == len(g):
            assert j == 0
        return l, j

    def node(self, l, j):
        g = self.g
        if not 0 <= l <= len(g):
            raise Exception("invalid layer index l")
        if l == len(g):
            if j != 0:
                raise Exception("invalid node index j")
            return self.n - 1
        if not 0 <= j < g[l].shape[0]:
            raise Exception("invalid node index j")
        i = sum([g[k].shape[0] for k in range(l)]) + j
        return i

    def _test_conversions(self):
        g = self.g
        n = self.n
        check_graph(g)
        assert n == self.num_nodes()
        print("test i -> (l,j) -> i")
        for i in range(n):
            l, j = self.indices(i)
            i1 = self.node(l, j)
            print(i, "->", (l,j), "->", i1)
            assert i1 == i
        print("test (l,j) -> i -> (l,j)")
        for l in range(len(g)):
            for j in range(g[l].shape[0]):
                i = self.node(l, j)
                l1, j1 = self.indices(i)
                print((l,j), "->", i, "->", (l1,j1))
                assert (l1, j1) == (l, j)
        l, j = len(g), 0
        i = self.node(l, j)
        l1, j1 = self.indices(i)
        print((l,j), "->", i, "->", (l1,j1))
        assert (l1, j1) == (l, j)
        print("OK")

    def gen_graph_like(self, x0 = 0.0, x = np.inf, typ=float):
        g = self.g
        return [filled(1, x0, typ)] + [filled(gl.shape[1], x, typ) for gl in g]


def shortest_path_cl(g):
    if not isinstance(g, DAWLGraph):
        g = DAWLGraph(g)
    L = len(g)
    c = g.gen_graph_like()
    whence = g.gen_graph_like(x0=0, x=-1, typ=int)
    for l in range(1, L+1):
        w = c[l-1][:,np.newaxis] + g[l-1]
        np.min(w, axis = 0, out = c[l])
        np.argmin(w, axis = 0, out = whence[l])

    l, j = L, 0
    path = [g.node(l, j)]
    while l != 0:
        j = whence[l][j]
        l -= 1
        path.append(g.node(l, j))
    path.reverse()

    return c[-1][0], path
