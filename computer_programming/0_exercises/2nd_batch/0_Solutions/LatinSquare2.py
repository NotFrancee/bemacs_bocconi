import numpy as np
from copy import deepcopy

## Auxiliary cost function: returns 1 if the input is not
## a permutation of {0, ..., n-1}. This is done by checking
## whether there are repeated elements in it, that is to say
## that the number of unique elements is smaller than the
## length.
## This is used for each row
def cost1(a):
    return int(len(np.unique(a)) != len(a))

class LatinSquare:
    def __init__(self, n):
        if not isinstance(n, int) or n <= 0:
            raise Exception("n must be a positive integer")

        ## Create a table in which each column is [0,1,...,n-2,n-1].
        ## A quick way to do it is via broadcasting, summing a column
        ## of [0,...,n-1] with a row of zeros.
        self.table = np.arange(n).reshape(n,1) + np.zeros(n, dtype=int)
        self.n = n

    def init_config(self):
        n, table = self.n, self.table
        ## We could have used the same code as in `__init__`, but here
        ## we'll shuffle (in-place!) each column of `table` individually,
        ## just to demonstrate.
        for i in range(n):
            np.random.shuffle(table[:,i])

    def __repr__(self):
        return "LatinSquare:\n" + str(self.table)

    def cost(self):
        n = self.n
        ## The cost is just the number of invalid rows + invalid columns
        invalid = 0
        ## But now the columns are always valid, by construciton
        ## (as long as the move is generated correctly). So we only
        ## need to check the rows.
        ## Iterating over a matrix returns the rows (given as views)
        for row in self.table:
            invalid += cost1(row)
        return invalid

    def propose_move(self):
        n = self.n
        ## Our move consists in picking two entries at random in the same column
        ## and swapping them.
        ## So we need to choose one column and two rows
        col = np.random.randint(n)
        while True:
            r1, r2 = np.random.randint(n), np.random.randint(n)
            if r1 != r2:
                break

        ## Our move will need to encode the two table posisions that we swap
        return (col, r1, r2)

    def compute_delta_cost(self, move):
        col, r1, r2 = move # unpack the move
        n = self.n
        table = self.table

        ## The cost contribution of the current configuration comes from
        ## the two rows involved in the move
        oldc1 = cost1(table[r1,:])
        oldc2 = cost1(table[r2,:])
        oldc = oldc1 + oldc2

        ## Change the table temporarily
        table[[r1,r2],col] = table[[r2,r1],col]

        ## Compute the new cost contributions
        newc1 = cost1(table[r1,:])
        newc2 = cost1(table[r2,:])
        newc = newc1 + newc2

        ## Cost difference, new - old
        delta_c = newc - oldc

        ## Restore the table (we haven't accepted the move yet!)
        table[[r1,r2],col] = table[[r2,r1],col]

        return delta_c

    def accept_move(self, move):
        col, r1, r2 = move # unpack the move
        self.table[[r1,r2],col] = self.table[[r2,r1],col]

    def copy(self):
        return deepcopy(self)
