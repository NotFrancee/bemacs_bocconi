"""
Implement a slightly sub-optimal version of the
Sieve of Eratosthenes, like this:
1. create an array of bools of length  n , all set to  True
2. set the  0 -th and  1 st element to  False
3. loop over all numbers between  2  and  √(n-1)
(rounded down to the nearest integer, included). For each
value  i , set to  False  all the elements indexed by multiples of  i ,
starting from  i**2 .
The resulting array will only contain  True s in the positions
corresponding to prime numbers.

Advanced: get the indices of those prime numbers with one line of code,
using an  arange  and boolean mask
indexing. (Alternatively, you can use  nonzero  or  flatnonzero ;
alternatively, you can use  argwhere  and
ravel )
"""

import numpy as np
import time


def sieve(n):
    arr = np.ones(n, dtype=bool)

    arr[[0, -1]] = False

    end = int(np.sqrt(n))

    for i in np.arange(2, end):
        # find all multiples of i
        multiples = np.arange(i**2, n, step=i)
        arr[multiples] = False

    primes = np.nonzero(arr)
    print(primes)
    return primes


"""
Same as exercise 24, but this time implement the true sieve,
as described in the Wikipedia page (or any other
resource). The difference is this: in point 3,
instead of looping through all indices between  2  and  √n , use a
while  loop. Start from  i=2 , but then at each cycle
you should get the next  i  as the first index after the current
i  for which your array is  True.

Since the array is made of bools, you can use the  argmax
function from numpy to get the first  True  index in the
array. But beware, you want to find the next
True  after the current  i . Can you do that without creating copies or
using loops?

Try to measure the time difference between this version
of the algorithm and the previous one, with  n == 10**7
or larger. It should be roughly 4 or 5 times faster
(the gain increases with increasing  n ). You may also want to try
a version which uses Python lists and for loops,
and check how much slower that is (on my laptop, roughly 30
times slower for  n==10**7
"""


def real_sieve(n):
    arr = np.ones(n, dtype=bool)
    arr[[0, -1]] = False

    current_i = 1

    while arr[current_i + 1 :].any():  # noqa
        current_slice = arr[current_i + 1 :]  # noqa
        i = np.argmax(current_slice) + current_i + 1

        arr[np.arange(i**2, n, step=i)] = False
        current_i = i

    primes = np.nonzero(arr)
    print(primes)
    return primes


# sieve(1000)
# real_sieve(1000)


def timed_test(n):
    arr = np.power(10, np.arange(n) + 4)
    print(arr)

    for i in arr:
        t = time.time()

        sieve(i)
        t1 = time.time() - t

        real_sieve(i)
        t2 = time.time() - t

        print(f"time1={t1}, time2={t2}")
        print(f"the fast version is {(t2 / t1 - 1):.2f}x  faster")


timed_test(3)
