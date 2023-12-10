import numpy as np
from copy import deepcopy

## This file is derived from "LatinSquare3.py".
## The only changes are in the constructor (extra check and extra `sn` attribute),
## in the __repr__ method (unnecessary, purely cosmetic change), and in the
## cost and compute_delta_cost methods (additional sub-squares contributions)
## There is also a change in `cost1` because `len(a)` doesn't work if `a` is
## a 2d-array (it returns the number of rows), so we need to use the `size`
## attribute.

## Auxiliary cost function: returns the number of repeated
## elements in the input, which can be computed as the number
## of elements minus the number of unique elements.
## This is used for each row
def cost1(a):
    return a.size - np.unique(a).size

## An auxiliary function two swap two rows in a given column, ensuring that
## they are not fixed (note: this could be greatly improved)
def rand2rows(mask, col, n):
    while True:
        i1, i2 = np.random.randint(n), np.random.randint(n)
        if i1 != i2 and not mask[i1,col] and not mask[i2,col]:
            break
    return i1, i2

class Sudoku:
    def __init__(self, init, r=None):
        if not (isinstance(init, int) or isinstance(init, Sudoku)):
            raise Exception("the first argument must be either an int or a Sudoku")

        if isinstance(init, int):
            if r is not None:
                raise Exception("can not pass r without an init puzzle")
            n = init
            if n <= 0 or not int(np.sqrt(n))**2 == n:
                raise Exception("n must be a positive integer and an exact square")

            ## Create a table in which each column is [0,1,...,n-2,n-1].
            ## A quick way to do it is via broadcasting, summing a column
            ## of [0,...,n-1] with a row of zeros.
            self.table = np.arange(n).reshape(n,1) + np.zeros(n, dtype=int)
            self.n = n
            self.sn = int(np.sqrt(n))
            ## We need to create an empty mask too (see code below)
            self.mask = np.zeros((n,n), dtype=bool)
        else:
            if r is None:
                raise Exception("when passing an init puzzle, r is required")
            if not isinstance(r, float) or not (0 <= r < 1):
                raise Exception("invalid r, must be 0 <= r < 1")
            if init.cost() != 0:
                print("WARNING: the init argument is not a valid puzzle!")
            n = init.n
            self.n = n
            self.sn = int(np.sqrt(n))
            self.table = init.table.copy()

            table = self.table

            ## Create a mask and fill it randomly, each entry being True with
            ## probability `r` (but there is a better way, see below)
            # mask = np.zeros((n,n), dtype=bool)
            # for i in range(n):
            #     for j in range(n):
            #         if np.random.rand() < r:
            #             mask[i,j] = True

            ## Alterative, shorter way, exploiting numpy's broadcasting rules,
            ## remembering that comparisons produce bools (this is also much
            ## faster)
            mask = np.random.rand(n, n) < r

            ## Alternative way to fill the mask that guarantees that there will
            ## be exactly `int(n*n*r)` fixed elements in the mask, distributed
            ## uniformly:
            # mask = np.zeros((n,n), dtype=bool)   # init to False
            # mask.ravel()[0:int(n*n*r)] = True    # set the first n*n*r elements
            # np.random.shuffle(mask.ravel())      # mix them up randomly

            self.mask = mask

            ## For each column, if there are less than 2 free items then there
            ## is not enough freedom to do a swap. In such case, if there is
            ## only one free item, we mark it as fixed anyway.

            ## Version 1: straightforward

            # for j in range(n):
            #     free = 0
            #     for i in range(n):
            #         if not mask[i,j]:
            #             free += 1
            #     if free < 2:
            #         mask[:,j] = True # fix the whole column

            ## Version 2, faster and shorter: summing up an array of bools is the
            ##            same as counting the Trues. We want to avoid the
            ##            situation where in a column there are exactly n-1
            ##            fixed elements: in that case we set everything to True.
            for j in range(n):
                if sum(mask[:,j]) == n-1:
                    mask[:,j] = True

            ## We new check that there is at least some freedom left

            if mask.sum() == n**2:
                raise Exception("Failed to create a meaningful puzzle; try lowering r")

            ## Randomize the columns, while keeping the fixed items fixed.
            ## Version 1: for each column, swap elements for a while...
            ##            (this not very good, and can also be very
            ##             inefficient, but it's easy)

            # for j in range(n):
            #     # we arbitrarily choose to perform 10*n swaps
            #     for k in range(10*n):
            #         i1, i2 = rand2rows(self.mask, j, n)
            #         self.table[[i1,i2]:j] = self.table[[i2,i1]:j]

            ## Version 2: do a random permuatation of the "unmasked" elements
            ##            in each column, and put them back in (this is much
            ##            better)

            for j in range(n):
                mj = ~mask[:,j] # unmasked elements in column j
                                # (the `~` means `not`, inverts True/False)
                v = table[mj,j]
                np.random.shuffle(v)
                table[mj,j] = v

    def init_config(self):
        n, table, mask = self.n, self.table, self.mask
        ## Reshuffle the table, but only the un-masked elements
        for j in range(n):
            mj = ~mask[:,j] # unmasked elements in column j
                            # (the `~` means `not`, inverts True/False)
            v = table[mj,j]
            np.random.shuffle(v)
            table[mj,j] = v

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

    def showpuzzle(self):
        ## This function prints only the fixed entries, so that we get to
        ## see the puzzle. It's basically the same as __repr__.
        s = ""
        n, sn, table, mask = self.n, self.sn, self.table, self.mask
        pad = int(np.ceil(np.log10(n-1)))
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
                if mask[i,j]:
                    v = table[i,j]
                else:
                    v = '?'
                s += "{:>{width}} ".format(v, width=pad)
            s += "\n"
        print("Sudoku puzzle:\n" + s)

    def cost(self):
        n, sn = self.n, self.sn
        table = self.table
        ## The cost is just the number repetitions in each row, column and sub-square
        c = 0
        ## This is the same as LatinSquare3
        for row in self.table:
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
        n, mask = self.n, self.mask
        ## Our move consists in picking two entries at random in the same column
        ## and swapping them.
        ## So we need to choose one column and two rows
        ## However, the column must have enough freedom: there need to be at least
        ## two free places to swap. By construction (see `__init__`) we have already
        ## avoided the case when there is only a single free place. Thus we just
        ## need to ensure that the count of fixed places is less than `n`.
        while True:
            col = np.random.randint(n)
            if mask[:,col].sum() < n: # Summing an array of bools = count the Trues
                break
        r1, r2 = rand2rows(mask, col, n)

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
