import numpy as np
from copy import deepcopy

### A somewhat advanced data structure that allows to
### attempt and perform moves much more efficiently in
### any of the LatinSquare exercises (and could be used
### for Sudoku too).

### The idea is to keep, for each row, a counter (an array of length
### `n`) such that `counter[v]` tells you how many times `v` appears
### in `row`. If `v` does not appear in `row` then `counter[v] == 0`.
### We'll also keep a "counter of the counter" (an array of length
### `n+1`) such that `chist[c]` tells you how many times `c` appears
### in `counter`.
### Notice that `chist[c]` is then telling you how many elements
### of `row` appear repeated `c` times.
### Therefore `chist[0]` tells you how many elements of `row`
### appear 0 times. In other words, how many elements of `row` (among
### the possible ones, 0,1,...,n-1) are missing from `row`.
### This is precisely the information we need to compute the cost.
### Therefore, once these two arrays `counter` and `chist` are built,
### the computation of the cost is immediate, it's O(1) time.

### The usefulness of these structures also relies in the fact that they
### can be updated in O(1) whenever we change an element of the row,
### al illustrated in the following example:
###
### Say that
###   n = 5
###   row = [1, 0, 0, 3, 1]
### Then the structure are:
###   counter = [2, 2, 0, 1, 0]
###   chist = [2, 1, 2, 0, 0, 0]
### Thus the costs are computed from `chist[0]`:
###   cost0 = (2 != 0) = 1    # LatinSquare1 style cost
###   cost1 = chist[0] = 2    # LatinSquare2+3+Sudoku style cost
###
### Now make a move: change 0->4 at index 2
###   row = [1, 0, 4, 3, 1]
### In `counter`, only the indices 0 and 4 are involved, and the changes
### are 2->1 and 0->1 (important for `chist`)
###   counter = [1, 2, 0, 1, 1]
### In `chist`, only the indices 0, 1, 2 are involved, and the changes
### are (see previous comment) chist[0]-=1, chist[1]+=2, chist[2]-=1
###   chist = [1, 3, 1, 0, 0, 0]
### Again the costs are computed immediately:
###   cost0 = (1 != 0) = 1
###   cost1 = chist[0] = 1

### Since we need a data structure like this for each row (for Sudoku, we would
### also need one for each sub-square), we create a class that does this.


## Auxiliary cost function: returns the number of repeated
## elements in the input, which can be computed as the number
## of elements minus the number of unique elements.
## This is used for each row
def cost1(a):
    return len(a) - len(np.unique(a))

## Auxiliary class with a data structure that allows
## to keep track efficiently of how "far" a given
## vector is from being a permutation. The efficiency
## consists in the fact that each update operation
## takes O(1) and the cost can be computed in O(1).
class CostTracker:
    def __init__(self, vec):
        n = len(vec)
        self.n = n
        ## Keep a counter for each element in the
        ## input `vec`, i.e. counter[v] tells you
        ## how many times the number `v` appears
        ## in `vec`.
        counter = np.zeros(n, dtype=int)
        for v in vec:
            assert 0 <= v < n
            counter[v] += 1

        ## Keep a "histogram", i.e. a counter for
        ## the `counter` array that we just built.
        ## `chist[c]` tells you how many elements
        ## appear `c` times in the original `vec`.
        ## It needs `n+1` entries because an
        ## element can appear `n` times in the `vec`.
        ## Notice that `self.chist[0]` contains the
        ## number of elements of `vec` that appear `0`
        ## times, i.e. the number of missing elements,
        ## which is exactly our cost.
        chist = np.zeros(n+1, dtype=int)
        for c in counter:
            chist[c] += 1

        ## Alternative one-liner
        # chist = np.sum(counter == np.arange(n+1)[:,np.newaxis], axis=1)

        self.counter, self.chist = counter, chist
        # self.check()

    def __repr__(self):
        return f"counter={self.counter} chist={self.chist}"

    ## Returns the number of elements that appear 0
    ## times in the input `vec`. It's O(1) time.
    def cost(self):
        ## For the type of cost in LatinSquare1 you would just
        ## return instead `self.chist[0] > 0`.
        return self.chist[0]

    ## Debug function: makes sure that the data structure is consistent
    ##                 (especially useful after updates)
    def check(self):
        n = self.n
        counter, chist = self.counter, self.chist
        assert np.all(counter >= 0)
        assert sum(counter) == n
        assert np.all(chist >= 0)
        assert np.arange(n+1) @ chist == n ## The `@` performs the dot product
        assert np.array_equal(chist, np.sum(counter == np.arange(n+1)[:,np.newaxis], axis=1))

    ## Suppose that, in any point in the original `vec`, an entry
    ## that used to be `old` is now changed to `new`. This updates
    ## the data structure, in O(1) time.
    ## For convenience, it returns the delta_cost of the operation.
    ## A few debug statements are available but commented out.
    def accept_move(self, old, new):
        if old == new:
            return 0 # nothing to do and no cost changes

        # assert 0 <= old < self.n
        # assert 0 <= new < self.n
        counter, chist = self.counter, self.chist

        ## Current cost
        cost0 = self.cost()
        ## Current counter values for old and new
        cold0, cnew0 = counter[old], counter[new]
        ## Updated counter values for old and new: remove old, add new
        cold1, cnew1 = cold0 - 1, cnew0 + 1
        # assert cold0 > 0

        ## Update the counter
        counter[old] = cold1
        counter[new] = cnew1

        ## Update the histogram: remove the current counter values,
        ## add the new ones
        chist[cold0] -= 1
        chist[cnew0] -= 1
        chist[cold1] += 1
        chist[cnew1] += 1

        ## Cost difference
        delta_cost = self.cost() - cost0

        # self.check() # for debugging
        return delta_cost

    ## Computes the delta_cost of changing an entry of `vec` from
    ## `old` to `new`. It actually just performs the change and
    ## then reverts it.
    def compute_delta_cost(self, old, new):
        delta_cost = self.accept_move(old, new)
        self.accept_move(new, old)
        return delta_cost

    ## Copy method
    def copy(self):
        return deepcopy(self)

class LatinSquare:
    def __init__(self, n):
        if not isinstance(n, int) or n <= 0:
            raise Exception("n must be a positive integer")

        ## Create a table in which each column is [0,1,...,n-2,n-1].
        ## A quick way to do it is via broadcasting, summing a column
        ## of [0,...,n-1] with a row of zeros.
        self.table = np.arange(n).reshape(n,1) + np.zeros(n, dtype=int)

        ## For each row, create a CostTracker object so that we can
        ## quickly determine the effect of a move on that row.
        self.ct = [CostTracker(self.table[i,:]) for i in range(n)]

        self.n = n

    def init_config(self):
        n, table, ct = self.n, self.table, self.ct
        ## We could have used the same code as in `__init__`, but here
        ## we'll shuffle (in-place!) each column of `table` individually,
        ## just to demonstrate.
        for j in range(n):
            np.random.shuffle(table[:,j])
        for i in range(n):
            ct[i] = CostTracker(table[i,:])

    def __repr__(self):
        return "LatinSquare:\n" + str(self.table)

    def cost(self):
        n = self.n
        ## The cost is just the number repetitions in each row and column
        c = 0
        ## As for LatinSquare2, the columns are ok by constructions, and
        ## we only need the rows contributions.
        for row in self.table:
            c += cost1(row)

        ## Here we could actually use the tracked costs.
        # c2 = 0
        # for i in range(n):
        #     c2 += self.ct[i].cost()
        # assert c == c2
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

        ## Our move will need to encode the two table posisions that we swap
        return (col, r1, r2)

    def compute_delta_cost(self, move):
        col, r1, r2 = move # unpack the move
        n = self.n
        table, ct = self.table, self.ct

        ## Read out the two values that we're about to swap
        v1, v2 = table[[r1,r2], col]
        ## We have two cost contributions, one for each row
        dc1 = ct[r1].compute_delta_cost(v1, v2) # in row `r1`, v1 -> v2
        dc2 = ct[r2].compute_delta_cost(v2, v1) # in row `r2`, v2 -> v1

        return dc1 + dc2

    def accept_move(self, move):
        col, r1, r2 = move # unpack the move
        table, ct = self.table, self.ct
        ## Read out the two values that we're about to swap
        v1, v2 = table[[r1,r2], col]
        ## Perform the swap in the table
        table[[r1,r2],col] = [v2, v1]
        ## We need to perform the swap in the CostTrackers too
        ct[r1].accept_move(v1, v2)
        ct[r2].accept_move(v2, v1)

    def copy(self):
        return deepcopy(self)
