import numpy as np
from copy import deepcopy

class MaxCut:
    def __init__(self, init, seed=None):
        if not (isinstance(init, int) or isinstance(init, np.ndarray)):
            raise Exception("init must be an integer or a 2-d array")

        if seed is not None:
            np.random.seed(seed)

        if isinstance(init, int):
            n = init
            if n <= 0:
                raise Exception("n must be positive")
            ## We can generate a random uniform matrix
            init = np.random.rand(n,n)
            ## But we need to make it symmetric! We copy the upper-triangular
            ## part onto the lower-triangular one
            for i in range(n-1):
                for j in range(i+1, n):
                    init[i,j] = init[j,i]
            ## We also need to set to zero the diagonal
            for i in range(n):
                init[i,i] = 0.0
            ## Alternative way to set the diagonal all at once:
            # init[np.diag_indices(n)] = 0.0

            ## Alternative, much shorter way to do everything at once
            ## ("triu" is short for "upper triangular")
            # init = np.triu(init, 1) + np.triu(init, 1).T

        assert isinstance(init, np.ndarray)

        if init.ndim != 2:
            raise Exception("The init array must be 2-dimensional")
        n = init.shape[0]
        if init.shape[1] != n:
            raise Exception("The init array must be square")
        if not np.array_equal(init, init.T):
            raise Exception("The init array must be symmetric")
        if (init < 0).any():
            raise Exception("The init array elements must be non-negative")
        if (init.diagonal() != 0).any():
            raise Exception("The diagonal entries of the init array must be 0")

        self.n = n
        self.weights = init

        ## We use an array of booleans to represent which elements belong
        ## to one subset and which to the other.
        mask = np.random.rand(n) < 0.5

        ## However, it's more convenient (for reasons which will be clear in
        ## the compute_delta_cost method) to use -1 and +1 as values, rather
        ## than True and False. The conversion can be done easily with same
        ## algebra (this broadcasts):
        cut = 2 * mask - 1

        self.cut = cut

    def init_config(self):
        n = self.n
        ## Same code as in the constructor (we might as well have avoided the
        ## code duplication...)
        mask = np.random.rand(n) < 0.5
        self.cut[:] = 2 * mask - 1

    def cost(self):
        n, weights, cut = self.n, self.weights, self.cut
        c = 0.0

        ## We get the 2 sets by using the cut (converting it back to bools by
        ## using comparisons)
        setA = np.arange(n)[cut > 0]
        setB = np.arange(n)[cut < 0]
        for i in setA:
            for j in setB:
                c += weights[i,j]
        ## We want to *maximize* the cut here, therefore we change the sign
        return -c

    def propose_move(self):
        ## The move proposal will just be one index that we will attempt to flip
        n = self.n
        return np.random.randint(n)

    def accept_move(self, move):
        ## Just flip the one bit in the cut
        i = move
        cut = self.cut
        cut[i] *= -1
        ## Alternatively:
        # cut[i] = -cut[i]

    def compute_delta_cost(self, move):
        i = move
        weights, cut = self.weights, self.cut

        oldv = cut[i]

        ## Let's call setA the one to which entry `i` belongs, and setB the
        ## one to which it would switch by the move.
        ## The cost difference will be given as: remove the cost of the links
        ## between i and all elements in setB, add the cost of all the links
        ## between i and all elements in setA.
        ## All of this is computed easily thanks to our choice for the cut
        ## (write the formulas down on a piece of paper to see that this
        ## it the case!):

        delta_cost = -oldv * np.sum(weights[i, :] * cut)

        ## Alternative writing: use the dot-product (aka inner product) function
        # delta_cost = -oldv * np.dot(weights[i, :], cut)

        ## Yet another way do write the same thing: the `@` operator
        ## represents matrix multiplication, but it's a little weird
        ## (compared to the mathematical definition) and if used with two
        ## 1-d arrays it actually also performs the dot-product...
        # delta_cost = -oldv * (weights[i, :] @ cut)

        return delta_cost

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return "MaxCut:\n" + \
               "  weights=\n" + str(self.weights) + "\n" + \
               "  cut=" + str(self.cut)
