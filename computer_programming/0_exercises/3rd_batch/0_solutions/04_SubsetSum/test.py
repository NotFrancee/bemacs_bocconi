## A small test on random data to demonstrate that the
## two alternative versions are actually better

import numpy as np
import SubsetSum as ss
import SubsetSum_BETTER as ssb
import SubsetSum_BETTER2 as ssb2
from time import time

def test(n, m, seed = 1236):
    np.random.seed(seed)
    v = np.random.randint(1, m, size=n)
    t1, t2, t3 = 0.0, 0.0, 0.0
    for s in range(v.sum() + 2):
        t = time()
        inds1 = ss.get_subset(v, s)
        t1 += time() - t
        t = time()
        inds2 = ssb.get_subset(v, s)
        t2 += time() - t
        t = time()
        inds3 = ssb2.get_subset(v, s)
        t3 += time() - t
        assert inds1 == inds2
        # print(f"v={v} s={s} inds2={inds2} inds3={inds3}")
        assert (inds2 is None) == (inds3 is None)
    print(f"t1={t1} t2={t2} t3={t3}")

test(10, 10)
test(100, 2)
test(100, 5)
test(100, 8)
test(40, 30)
