import numpy as np
from copy import deepcopy

## Auxiliary cost function: returns 1 if the input is not
## a permutation of {0, ..., n-1}. This is done by checking
## whether there are repeated elements in it, that is to say
## that the number of unique elements is smaller than the
## length.
## This is used for each row and column
def cost1(a):
    return int(len(np.unique(a)) != len(a))

class LatinSquare:
    def __init__(self, n, seed=None):
        if not isinstance(n, int) or n <= 0:
            raise Exception("n must be a positive integer")
        if seed is not None:
            np.random.seed(seed)
        # The initial table is completely random here
        self.table = np.random.randint(0, n, (n,n))
        self.n = n

    def init_config(self):
        n, table = self.n, self.table
        table[:,:] = np.random.randint(0, n, (n,n))

    def __repr__(self):
        return "LatinSquare:\n" + str(self.table)

    def cost(self):
        n = self.n
        ## The cost is just the number of invalid rows + invalid columns
        invalid = 0
        ## Iterating over a matrix returns the rows (given as views)
        for row in self.table:
            invalid += cost1(row)
        ## Iteration over the columns is the same, but we transpose the matrix
        for col in self.table.T:
            invalid += cost1(col)
        return invalid

    def propose_move(self):
        n = self.n
        ## For our move we just pick an entry at random and we change it.
        ## Here we pick the entry
        i, j = np.random.randint(0, n), np.random.randint(0, n)

        ## Current value
        oldv = self.table[i,j]

        ## New value: change the old value to one other valid value.
        while True:
            newv = np.random.randint(n)
            if newv != oldv:
                break

        ## Alternative version, 1 line of code and no rejections:
        # newv = (oldv + np.random.randint(1, n)) % n

        ## Our move will need to encode the table posision and the new value
        return (i, j, newv)

    def compute_delta_cost(self, move):
        i, j, newv = move # unpack the move
        n = self.n
        table = self.table

        ## Read the current value at (i,j)
        oldv = self.table[i,j]

        ## The contribution of that is the cost associated to its row and column
        oldc_r = cost1(table[i,:])
        oldc_c = cost1(table[:,j])
        oldc = oldc_r + oldc_c

        ## Change the table temporarily
        self.table[i,j] = newv

        ## Compute the new cost contributions
        newc_r = cost1(table[i,:])
        newc_c = cost1(table[:,j])
        newc = newc_r + newc_c

        ## Cost difference, new - old
        delta_c = newc - oldc

        ## Restore the table (we haven't accepted the move yet!)
        self.table[i,j] = oldv

        return delta_c

    def accept_move(self, move):
        i, j, newv = move # unpack the move
        self.table[i,j] = newv

    def copy(self):
        return deepcopy(self)
