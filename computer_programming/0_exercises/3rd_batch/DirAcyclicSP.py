## This is the solution to ex. 6, without using classes.
## See "DirAcyclicSP_class.py" for a version using classes.

import numpy as np

## Same graph as in "dir_graph.pdf".
## It's a list of 2-d arrays. Notice that even the first entry is 2-d, although
## it only has 1 row.
## Each entry in the list represents a matrix of costs between two layers.
## The entry `(i,j)` represents the cost from the `i`-th node of the previous
## layer and the `j`-th node of the next layer. This implies that the number
## of columns of one matrix must match the number of rows of the next matrix.
## The definition of the problem also requires that the first matrix has 1 row
## and the last matrix has 1 column.
## Missing links are represented as infinity (`np.inf`).

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

## Example: in the figure, the node labeled 2 is in layer `1` (note: the first
## layer is `0`), and it is connected to the node labeled 6 in layer `2`, with
## a cost of `12.0`. This entry will thus be found in the matrix at index `1`
## (which contains all links between layer `1` and `2`); since node 2 is the
## second node of layer `1`, it has index `1` within that layer; since node 6
## is the third node of layer `2`, it has index `2` within that layer;
## therefore the entry is at position `(1,2)` in the matrix. Overall, we thus
## have the relation:
##    "cost 2->6" := test_c1[1][1,2]

## Another example, an extreme graph with a sinlge link between one input and
## one output:
test_g0 = [np.array([[3.3]])]

### ~~~~~~~ ###

## This function checks that a graph is given in the above form, and fulfills
## all the contstraints on the size of the matrices.
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


## This just returns the number of nodes in the graph
def num_nodes(g):
    #check_graph(g)
    return sum([gl.shape[0] for gl in g]) + 1


## Given a graph represented like in the above example, we want to be able to
## convert from the index of a node in the graph to a pair of numbers, denoting
## a layer index and the position within that layer. We also want to convert in
## other direction.
## For the above graph, we want the mapping:
##    0 <-> (0, 0)
##    1 <-> (1, 0)
##    2 <-> (1, 1)
##    3 <-> (1, 2)
##    4 <-> (2, 0)
##    5 <-> (2, 1)
##    6 <-> (2, 2)
##    7 <-> (3, 0)
##    8 <-> (3, 1)
##    9 <-> (4, 0)
## So let's write two functions that do just that. They will take the cost
## graph as their first argument.

## 1-index form -> 2-indices form
def ind2layer(g, i):
    #check_graph(g)
    if not 0 <= i < num_nodes(g):
        raise Exception("out-of-bounds index i")
    l, j = 0, i
    while l < len(g) and j >= g[l].shape[0]:
        j -= g[l].shape[0]
        l += 1
    if l == len(g):
        assert j == 0
    return l, j

## 2-indices form -> 1-index form
def layer2ind(g, l, j):
    #check_graph(g)
    if not 0 <= l <= len(g):
        raise Exception("invalid layer index l")
    if l == len(g):
        if j != 0:
            raise Exception("invalid node index j")
        return num_nodes(g) - 1
    if not 0 <= j < g[l].shape[0]:
        raise Exception("invalid node index j")
    i = sum([g[k].shape[0] for k in range(l)]) + j
    return i

## Sanity-check: the above two functions must work for all valid inputs, and
## they must be the inverse to each other.
def test_conversions(g):
    check_graph(g)
    n = num_nodes(g)
    print("test i -> (l,j) -> i")
    for i in range(n):
        l, j = ind2layer(g, i)
        i1 = layer2ind(g, l, j)
        print(i, "->", (l,j), "->", i1)
        assert i1 == i
    print("test (l,j) -> i -> (l,j)")
    for l in range(len(g)):
        for j in range(g[l].shape[0]):
            i = layer2ind(g, l, j)
            l1, j1 = ind2layer(g, i)
            print((l,j), "->", i, "->", (l1,j1))
            assert (l1, j1) == (l, j)
    l, j = len(g), 0
    i = layer2ind(g, l, j)
    l1, j1 = ind2layer(g, i)
    print((l,j), "->", i, "->", (l1,j1))
    assert (l1, j1) == (l, j)
    print("OK")

## Generate a structure that mimics the graph, i.e.: a list of 1-d arrays;
## The length of the list is the number of layers in the graph;
## The length of each array in the list is the number of nodes in the corresponding
## layer.
## All elements are initialized to `x` (infinity by default), except for the first
## one, which is initialized to `x0` (zero by default).
## We write a short auxiliary function to help with that
def filled(n, x, typ=float):
    a = np.zeros(n, dtype=typ)
    a.fill(x)
    return a

def gen_graph_like(g, x0 = 0.0, x = np.inf, typ=float):
    # check_graph(g)
    return [filled(1, x0, typ)] + [filled(gl.shape[1], x, typ) for gl in g]

## Shortest path, implementation of the forward pass only (it thus just computes
## the best cost and not the actual path). Also it doesn't use any numpy tricks.
def shortest_path(g):
    check_graph(g)
    L = len(g)
    ## Allocation of the autiliary cost structure
    ## The base case (1st layer) and the rest is filled-in by the gen_graph_like
    ## function.
    c = gen_graph_like(g)
    for l in range(1, L+1):
        cl0 = c[l-1]
        cl1 = c[l]
        for j in range(len(cl1)):
            ## It is crucial here that `cl1[j] == np.inf`.
            ## If we hadn't made sure of that in the cost initialization,
            ## we would have needed to do it here.
            for k in range(len(cl0)):
                w = cl0[k] + g[l-1][k,j]
                if w < cl1[j]:
                    cl1[j] = w
    ## Now the best cost is in the last layer; we are assuming that it
    ## only has 1 node so we look in position 0
    return c[-1][0]

## Same as shortest_path, but it keeps track of the decisions and performs
## the backward pass, so it returns the path too.
def shortest_path2(g):
    check_graph(g)
    L = len(g)
    c = gen_graph_like(g)
    ## Now we need another auxiliary structure to keep track of the argmins.
    ## The shape is the same, the type and initial values are different
    whence = gen_graph_like(g, x0=0, x=-1, typ=int)
    for l in range(1, L+1):
        cl0 = c[l-1]
        cl1 = c[l]
        whl = whence[l]
        for j in range(len(cl1)):
            for k in range(len(cl0)):
                w = cl0[k] + g[l-1][k,j]
                if w < cl1[j]:
                    cl1[j] = w
                    whl[j] = k ## This remembers that the argmin is k

    ## BACKWARD PASS

    ## We use the usual backtracking scheme
    ## We start from the end so the current layer (l) is the last one (L)
    ## and the current node (j) is 0 (we are assuming the last layer has
    ## only one node.
    l, j = L, 0
    ## We store that in the path in the form of a node-index
    path = [layer2ind(g, l, j)]
    ## Keep going until the first layer
    while l != 0:
        ## Get the previous node and add it to the path.
        ## Note that the order of the operations here is quite relevant.
        ## The previous node than will become current in the next iteration.
        j = whence[l][j]                 # path node in the previous layer
        l -= 1                           # go back one layer
        path.append(layer2ind(g, l, j))  # store it in the solution
    ## We used append, thus we revert before returning
    path.reverse()

    return c[-1][0], path

## Same as shortest_path2; leverages numpy syntax to avoid for loops
## (only the forward pass is affected, and as usual the outer for loop
## is essentially inevitable)
def shortest_path3(g):
    check_graph(g)
    L = len(g)
    c = gen_graph_like(g)
    whence = gen_graph_like(g, x0=0, x=-1, typ=int)
    for l in range(1, L+1):
        w = c[l-1][:,np.newaxis] + g[l-1]
        np.min(w, axis = 0, out = c[l])
        np.argmin(w, axis = 0, out = whence[l])

    l, j = L, 0
    path = [layer2ind(g, l, j)]
    while l != 0:
        j = whence[l][j]
        l -= 1
        path.append(layer2ind(g, l, j))
    path.reverse()

    return c[-1][0], path
