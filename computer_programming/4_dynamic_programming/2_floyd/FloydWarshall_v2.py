# Floyd-Warshall dynamic programming algorithm
# ============================================
#
# Input: We are given a directed weighted graph with N nodes.
#        The cost of edge i->j is given in a matrix as w[i,j].
#        If a link i->j is missing, we just set w[i,j] = np.inf.
#        We assume that there are no cycles with negative cost.
#
#  Goal: Find, for each pair of indices (i,j), the minimum cost of
#        a path connecting i to j. Note: it might be infinite...
#        Thus the output needs to be a NxN matrix. In general, it
#        won't be symetrical; its diagonal elements will be 0.
#
# Observations:
#
#  1. For each pair of nodes, the shortest path:
#     a) touches each node at most once (otherwise there would be
#        a cycle, and each cycle costs >= 0)
#     b) thus it has length at most N (including the endpoints)
#
#  2. Let's call c[i,j] the optimal cost of going from i to j. We
#     don't know it yet, it's what we're trying to compute.
#     But we can say that surely c[i,j] <= w[i,j] for every i,j.
#
# Optimal cost computation
# ------------------------
#
# Define a recursive relation. Start by defining the 0-th level
# like this:
#
#   r[0,i,j] = w[i,j]
#
# This is like considering all the paths that go directly from
# i to j without intermediaries.
# Now suppose that you allowed each path to either be direct, or
# to pass via node 0. Call r[1,i,j] the cost of the best result
# between these two options. Thus:
#
#   r[1,i,j] = min(w[i,j], w[i,0]+w[0,j])
#
# That is, we choose between going directly, or going through 0.
# Notice that we can rewrite that like this:
#
#   r[1,i,j] = min(r[0,i,j], r[0,i,0]+r[0,0,j])
#
# Now let's say that we also allow the paths to go thorugh node
# number 1, and we pick the best paths among those that go
# through either 0, or, 1, or neither, or both. Call that
# r[2,i,j]. The key observation is that:
#
#   r[2,i,j] = min(r[1,i,j], r[1,i,1]+r[1,1,j])
#
# That is: the best path is either the best path that goes at most
# through 0, or it's the best path that first goes from i to 1
# (potentially passing trhough 0) and then goes from 1 to j (again,
# potentially passing through 0, but we know that it won't be
# passing thorugh 0 twice anyway).
#
# You can now spot a general fact: We can keep adding nodes one at
# a time, writing
#
#   r[k+1,i,j] = min(r[k,i,j], r[k,i,k]+r[k,k,j])
#
# which is valid for 0 <= k < N.
# When we reach k+1==N, we have allowed the paths to go through all
# nodes potentially, therefore the result is the optimum:
#
#   c[i,j] = r[N,i,j]
#
# And thus after N steps (each of which has to be computed for all i,j)
# we have solved the problem. The complexity is thus O(N^3).
#
# Implementation note : after we have computed the (k+1)-th step, we
# don't need the k-th step any more. Thus we can save memory.
#
# Path reconstruction
# -------------------
#
# This gives us the optimal costs, but we still don't have the
# optimal paths. For that, we need to keep track, at each step k,
# of the decision that we have taken when we have computed the
# minimum (basically: did we choose the first or the second term
# in the min?). In fact, all we need is this: given the optimal
# path from i to j, what is the last element of the path before
# j? We don't need anything else because if we know that our
# path is i->...->z->j, then we can ask the same question about
# the path from i to z, i.e. what is the penultimate element
# before z in that path. Thus if we have that information we
# can reconstruct the whole path. Thus we only need to keep
# an NxN matrix of integers in memory. Call it pred[i,j].
# At the beginning, pred[i,j] is initialized to i if w[i,j] is
# finite, or to -1 if it is infinite (this is a "sentinel value",
# it signals that there is no known path between i and j). At each
# iteration k, we update this pred[i,j] based on whether we accept
# a new route through the new candidate node or not.


import numpy as np

test_w0 = np.array(
    [
        [0.0, 9.0, 3.0, 1.0],
        [-0.5, 0.0, -0.8, 0.1],
        [0.1, 3.3, 0.0, 2.2],
        [np.inf, np.inf, np.inf, 0.0],
    ]
)

test_w1 = np.array(
    [
        [0.0, 20.0, 10.0, 63.0, 72.0, np.inf],
        [np.inf, 0.0, 0.0, 40.0, np.inf, 70.0],
        [np.inf, 5.0, 0.0, 40.0, 34.0, 100.0],
        [np.inf, np.inf, -20.0, 0.0, -5.0, 36.0],
        [np.inf, -31.0, np.inf, 5.0, 0.0, 80.0],
        [np.inf, np.inf, np.inf, np.inf, np.inf, 0.0],
    ]
)


def floydwarshall(w):
    # argument checks:
    # 2-d array, square matrix
    # diagonal terms of w are zero
    # no cycles with negative cost
    n = w.shape[0]
    r = w.copy()
    # pred = np.zeros((n,n), dtype=int)# matrix of
    # integers with the same size as r

    pred = np.arange(n).reshape(n, 1) + np.zeros(n, dtype=int)
    msk = w == np.inf
    pred[msk] = -1

    # for i in range(n):
    #     for j in range(n):
    #         if r[i,j] == np.inf:
    #         # if no connection put -1
    #             pred[i,j] = -1
    #         else:
    #             # if connection exists put i
    #             pred[i,j] = i

    # FORWARD PASS
    for k in range(n):
        r_new = r[:, k].reshape(n, 1) + r[k, :]
        pred_new = pred[k, :] + np.zeros((n, 1), dtype=int)
        msk = r_new < r
        r[msk] = r_new[msk]
        pred[msk] = pred_new[msk]

        # for i in range(n):
        #     for j in range(n):
        #         # r[i,j] = min(r[i,j], r[i,k] + r[k,j])
        #         # r_k = r[i,k] + r[k,j]
        #         if r_new[i,j] < r[i,j]:
        #             r[i,j] = r_new[i,j]
        #             pred[i,j] = pred[k,j]
    return r, pred


def get_opt_path(pred, i, j):
    if pred[i, j] == -1:
        return []
    path = [i, j]
    z = pred[i, j]
    while z != i:
        path.insert(1, z)
        z = pred[i, z]
    return path


def check_all_paths(w):
    r, pred = floydwarshall(w)
    n = w.shape[0]
    # check all pairs
    for i in range(n):
        for j in range(n):
            path = get_opt_path(pred, i, j)
            # check when the path is empty that the cost r is infinite
            if len(path) == 0:
                assert r[i, j] == np.inf
                continue

            # if the path is non-empty, cumulate the cost of all the steps
            # and compare with r. (check they are close enough)
            r_rec = 0.0
            for k in range(len(path) - 1):
                r_rec += w[path[k], path[k + 1]]
            assert abs(r[i, j] - r_rec) < 1e-12

    print("both conditions ok")
    return
