import numpy as np
from copy import deepcopy
import collections

## Two auxiliary functions to compute the costs associated with the sum
## constraints
def cost1(a, s):
    return abs(a.sum() - s)

def cost2(s1, s):
    return abs(s1 - s)

## A factor which determines how much repetitions are weighted in the cost
## function with respect to the other constraints
repetitioncost = 2

class MagicSquare:
    def __init__(self, n, s, seed=None):
        if not isinstance(n, int) or n <= 0:
            raise Exception("n must be a positive integer")
        if not isinstance(s, int):
            raise Exception("s must be an int")
        if seed is not None:
            np.random.seed(seed)

        self.n = n
        self.s = s

        ## Rather than completing the construction, we delegate to
        ## the `init_config` method.
        ## In this way the function `simann` can be called on the same
        ## object repeatedly and it will reset it every time instead of
        ## starting from the last configuration found.
        self.init_config()

    def init_config(self):
        n, s = self.n, self.s

        ## Initialize the internal table at random. Note that this
        ## will 1) create a new attribute `table` in the object when
        ## called the first time, by the constructor; 2) replace that
        ## attribute (not overwrite it!) when called again later. This
        ## works fine here, but it's important in general to keep a close
        ## eye on these things.
        self.table = np.random.randint(1, s+1, (n,n))

        ## We keep some auiliary quantities to help computing the delta_cost
        ## more efficiently: the current sums along the rows, columns, and
        ## the two diagonals
        self.rsums = self.table.sum(axis=1)
        self.csums = self.table.sum(axis=0)
        self.d1sum = self.table.ravel()[::n+1].sum()
        self.d2sum = self.table.ravel()[n-1:n**2-1:n-1].sum()
        ## We also keep a counter for each element, to keep track of which
        ## elements are already present in the square (we want to avoid
        ## repetitions)
        self.count = collections.Counter(self.table.ravel())

    def cost(self):
        ## NOTE: there is some debug code here (the asserts) that could
        ##       be disabled to make this faster
        n, s = self.n, self.s
        table = self.table

        ## Row sums cost
        c_r = 0
        for row in table:
            c_r += cost1(row, s)
        assert c_r == sum([cost2(rs, s) for rs in self.rsums])

        ## Column sums cost
        c_c = 0
        for col in table.T:
            c_c += cost1(col, s)
        assert c_c == sum([cost2(cs, s) for cs in self.csums])

        ## Diagonals costs
        c_d1 = cost1(table.ravel()[::n+1], s)
        c_d2 = cost1(table.ravel()[n-1:n**2-1:n-1], s)
        assert c_d1 == cost2(self.d1sum, s)
        assert c_d2 == cost2(self.d2sum, s)

        ## Repetitions cost: if all elements are different, the
        ## length of the counter is n**2; if it's less than that,
        ## it means that there are repetitions
        c_count = n**2 - len(self.count)

        ## Grand total
        c = c_r + c_c + c_d1 + c_d2 + repetitioncost * c_count
        return c

    def propose_move(self):
        ## Our move will be: pick a random entry and change it by 1, either
        ## up or down (except if it is already 1, in which case we only change
        ## it up).
        ## The latter case is slightly problematic from the point of view of
        ## maintaining detailed balance in the MCMC, because the probabiliy of going
        ## from 1 to 2 is double the probability of going from 2 to 1. Therefore,
        ## we randomly discard half of the times we pick an element with value 1,
        ## which is equivalent to rejecting the move outright.
        n = self.n
        table = self.table

        while True:
            i, j = np.random.randint(n), np.random.randint(n)

            oldv = table[i,j]
            if oldv == 1:
                if np.random.rand() < 0.5: # randomly discard half the times a 1 is picked
                    continue
                delta = 1
            else:
                delta = 2 * np.random.randint(2) - 1
            break
        newv = oldv + delta

        return (i, j, newv)

    def compute_delta_cost(self, move):
        i, j, newv = move
        n, s = self.n, self.s
        table, count = self.table, self.count
        rsums, csums = self.rsums, self.csums
        d1sum, d2sum = self.d1sum, self.d2sum

        oldv = table[i,j]
        delta = newv - oldv

        ## The contributions that we care about come from the row
        ## and the column, and possibly (if we are in either diagonal)
        ## from the diagonals too. Finally there is the repetitions
        ## count cost

        rsi = rsums[i]
        csj = csums[j]

        oldc_r = cost2(rsi, s)            # row
        oldc_c = cost2(csj, s)            # column
        if i == j:                        # main diagonal
            oldc_d1 = cost2(d1sum, s)
        else:
            oldc_d1 = 0
        if i + j == n - 1:                # anti-diagonal
            oldc_d2 = cost2(d2sum, s)
        else:
            oldc_d2 = 0
        oldc_count = n**2 - len(count)    # repetitions
        oldc = oldc_r + oldc_c + oldc_d1 + oldc_d2 + repetitioncost * oldc_count

        ## Temporarily update the counter
        count[oldv] -= 1
        if count[oldv] == 0: # cf. Counter objects documentation
            del count[oldv]
        count[newv] += 1

        ## Redo the cost computations all over again with the new sums and
        ## the new counter

        newc_r = cost2(rsi + delta, s)
        newc_c = cost2(csj + delta, s)
        if i == j:
            newc_d1 = cost2(d1sum + delta, s)
        else:
            newc_d1 = 0
        if i == n - j - 1:
            newc_d2 = cost2(d2sum + delta, s)
        else:
            newc_d2 = 0
        newc_count = n**2 - len(count)
        newc = newc_r + newc_c + newc_d1 + newc_d2 + repetitioncost * newc_count

        delta_c = newc - oldc

        ## Restore the counter as it was before
        count[oldv] += 1
        count[newv] -= 1
        if count[newv] == 0:
            del count[newv]

        return delta_c

    def accept_move(self, move):
        i, j, newv = move
        ## Here we not only need to update the table, but also all the
        ## auxiliary quantities that we keep track of, like the current
        ## row sums, the repetitions counter etc. Otherwise we would
        ## get into an inconsistent state and the compute_delta_cost
        ## function would return completely wrong results.
        table, count = self.table, self.count
        rsums, csums = self.rsums, self.csums

        ## We change the table
        oldv = table[i,j]
        delta = newv - oldv
        table[i,j] = newv

        ## Update the counter
        count[oldv] -= 1
        if count[oldv] == 0:
            del count[oldv]
        count[newv] += 1

        ## Rows, columns and diagonals sums
        rsums[i] += delta
        csums[j] += delta
        ## WARNING: if we had used d1sum = self.d1sum, then changing d1sum
        ##          would not change self.d1sum here, since d1sum is an
        ##          integer and is immutable!!! (same for d2sum)
        if i == j:
            self.d1sum += delta
        if i + j == self.n - 1:
            self.d2sum += delta

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        ## This could be improved considerably, e.g. by somehow printing
        ## the sums beside the table...
        s = f"MagicSquare:\n{self.table}\n"
        s += f"target = {self.s}\n"
        s += f"row sums = {self.rsums}\n"
        s += f"col sums = {self.csums}\n"
        s += f"diag sums = {self.d1sum}, {self.d2sum}"
        return s

