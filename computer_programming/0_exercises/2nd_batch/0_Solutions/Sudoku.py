import numpy as np
from copy import deepcopy

## This file is derived from "LatinSquare3.py".
## The only changes are in the constructor (extra check and extra `sn` attribute),
## in the __repr__ method (unnecessary, purely cosmetic change), and in the
## cost and compute_delta_cost methods (additional sub-squares contributions).
## There is also a change in `cost1` because `len(a)` doesn't work if `a` is
## a 2d-array (it returns the number of rows), so we need to use the `size`
## attribute.

## Auxiliary cost function: returns the number of repeated
## elements in the input, which can be computed as the number
## of elements minus the number of unique elements.
## This is used for each row
def cost1(a):
    return a.size - np.unique(a).size

class Sudoku:
    def __init__(self, n):
        if not isinstance(n, int) or n <= 0 or not int(np.sqrt(n))**2 == n:
            raise Exception("n must be a positive integer and an exact square")

        ## Create a table in which each column is [0,1,...,n-2,n-1].
        ## A quick way to do it is via broadcasting, summing a column
        ## of [0,...,n-1] with a row of zeros.
        self.table = np.arange(n).reshape(n,1) + np.zeros(n, dtype=int)
        self.n = n
        self.sn = int(np.sqrt(n))

    def init_config(self):
        n, table = self.n, self.table
        ## We could have used the same code as in `__init__`, but here
        ## we'll shuffle (in-place!) each column of `table` individually,
        ## just to demonstrate.
        for i in range(n):
            np.random.shuffle(table[:,i])

    def __repr__(self):
        ## Improved printing; all this mess is just to add the sub-squares
        ## printing. Otherwise, it could have been the same as LatinSquare,
        ## just changing the printed name.
        s = ""
        n, sn, table = self.n, self.sn, self.table
        pad = len(str(n-1)) # max required length to fit all the digits
        for i in range(n):
            if i > 0 and i % sn == 0:
                for j in range(n):
                    if j > 0 and j % sn == 0:
                        s += "+-"
                    s += "-" * (pad+1)
                s += "\n"
            for j in range(n):
                if j > 0 and j % sn == 0:
                    s += "| "
                s += "{:>{width}} ".format(table[i,j], width=pad)
            s += "\n"
        return "Sudoku:\n" + s

    def cost(self):
        n, sn = self.n, self.sn
        table = self.table
        ## The cost is just the number repetitions in each row, column and sub-square
        c = 0
        ## This is the same as LatinSquare3
        for row in table:
            c += cost1(row)
        ## This additional part computes the cost of each sub-square.
        ## We have sn x sn sub-squares of size sn x sn
        for si in range(sn):
            i0 = sn * si           # range start, rows
            i1 = i0 + sn           # range end, rows
            for sj in range(sn):
                j0 = sn * sj       # range start, columns
                j1 = j0 + sn       # range end, columns
                ## Indexing with [i0:i1, j0:j1] gives us the sub-square
                ## We can pass it as-is to `cost1` because `unique` accepts
                ## multi-dimensional arrays without problems.
                c += cost1(table[i0:i1, j0:j1])
        return c

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

        ## Our move will need to encode the two table positions that we swap
        return (col, r1, r2)

    def compute_delta_cost(self, move):
        col, r1, r2 = move # unpack the move
        n, sn = self.n, self.sn
        table = self.table

        ## The first cost contribution of the current configuration comes from
        ## the two rows involved in the move
        oldc1 = cost1(table[r1,:])
        oldc2 = cost1(table[r2,:])

        ## The second constribution costs changes associated to the sub-squares
        sj = col // sn             # sub-square column index
        si1 = r1 // sn             # first sub-square row index
        si2 = r2 // sn             # second sub-square row index
        i10 = sn * si1             # range start, rows, first sub-sqaure
        i11 = i10 + sn             # range end, rows, first sub-square
        i20 = sn * si2             # range start, rows, second sub-square
        i21 = i20 + sn             # range end, rows, second sub-square
        j0 = sn * sj               # range start, columns
        j1 = j0 + sn               # range end, columns

        ## The sub-squares costs only change if the swap involves two different sub-squares
        if si1 != si2:
            olds1 = cost1(table[i10:i11, j0:j1])
            olds2 = cost1(table[i20:i21, j0:j1])
        else:
            olds1, olds2 = 0, 0

        oldc = oldc1 + oldc2 + olds1 + olds2

        ## Change the table temporarily
        table[[r1,r2],col] = table[[r2,r1],col]

        ## Compute the new cost contributions of the rows
        newc1 = cost1(table[r1,:])
        newc2 = cost1(table[r2,:])

        ## Compute the new cost contributions of the sub-squares
        if si1 != si2:
            news1 = cost1(table[i10:i11, j0:j1])
            news2 = cost1(table[i20:i21, j0:j1])
        else:
            news1, news2 = 0, 0

        newc = newc1 + newc2 + news1 + news2

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
