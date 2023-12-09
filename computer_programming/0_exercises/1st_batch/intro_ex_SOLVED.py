# flake8: noqa

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# 1. Numpy integer types


def ex1_inttypes():
    # NOTE: this exercise was intended to be done mainly at the REPL,
    # but here everything is put inside a function to keep things clean

    a = np.int16(100)  # a signed 16-bit integer
    b = np.uint16(100)  # an unsigned 16-bit integer
    print(f"a = {a} b = {b}")  # when you print them, you don't notice any difference
    print("a == b ? " + str(a == b))  # check that they compare equal

    print()

    maxint16 = 2**15 - 1  # maximum signed 16-bit value
    minint16 = -(2**15)  # minimum signed 16-bit value

    maxuint16 = 2**16 - 1  # maximum unsigned 16-bit value
    minuint16 = 0  # minimum unsigned 16-bit value

    print("converting to signed 16-bit (np.int16):")
    print(f"  max:        {maxint16} -> {np.int16(maxint16)}")
    print(f"  min:        {minint16} -> {np.int16(minint16)}")
    print(f"  beyond max: {maxint16+1} -> {np.int16(maxint16+1)}")
    print(f"  beyond min: {minint16-1} -> {np.int16(minint16-1)}")

    print("converting to unsigned 16-bit (np.uint16):")
    print(f"  max:        {maxuint16} -> {np.uint16(maxuint16)}")
    print(f"  min:        {minuint16} -> {np.uint16(minuint16)}")
    print(f"  beyond max: {maxuint16+1} -> {np.uint16(maxuint16+1)}")
    print(f"  beyond min: {minuint16-1} -> {np.uint16(minuint16-1)}")

    print("take the signed max/min and convert them to unsigned:")
    print(f"  max:        {maxint16} -> {np.uint16(maxint16)}")
    print(f"  min:        {minint16} -> {np.uint16(minint16)}")

    print("take the unsigned max/min and convert them to signed:")
    print(f"  max:        {maxuint16} -> {np.int16(maxuint16)}")
    print(f"  min:        {minuint16} -> {np.int16(minuint16)}")

    print()

    print("mixing signed and unsigned types, and python standard integers too:")
    print(f"  int16 + uint16: {a} + {b} = {a+b} (output type: {type(a+b)})")
    print(f"  int16 - uint16: {a} - {b} = {a-b} (output type: {type(a-b)})")
    print(f"  int16 - uint16: {a} - {2*b} = {a-2*b} (output type: {type(a-2*b)})")
    print(' [notice that the types have been "promoted" to a wider type]')
    print(" [ok but what about an even larger type? Let's go to 32 bits")
    c = np.int32(100)
    d = np.uint32(100)
    print(f"  int32 + uint32: {c} + {d} = {c+d} (output type: {type(c+d)})")
    print(f"  int32 - uint32: {c} - {d} = {c-d} (output type: {type(c-d)})")
    print(f"  int32 - uint32: {c} - {2*d} = {c-2*d} (output type: {type(c-2*d)})")
    print(" [we went to 64 bits now]")
    print(" [what if we go even wider? Let's go to 64 bits inputs]")
    e = np.int64(100)
    f = np.uint64(100)
    print(f"  int64 + uint64: {e} + {f} = {e+f} (output type: {type(e+f)})")
    print(f"  int64 - uint64: {e} - {f} = {e-f} (output type: {type(e-f)})")
    print(f"  int64 - uint64: {e} - {2*f} = {e-2*f} (output type: {type(e-2*f)})")
    print(
        " [all of this may be sort of confusing but occasionally it's good to know, or at least to have some vague idea about]"
    )
    print(" [let's put some standard Python integers in the mix too]")
    print(f"  int64 + int: {e} + {200} = {e+200} (output type: {type(e+200)})")
    print(f"  int64 - int: {e} - {200} = {e-200} (output type: {type(e-200)})")
    print(f"  int64 + int: {e} + {10**30} = {e+10**30} (output type: {type(e+10**30)})")
    print(f"  int64 - int: {e} - {10**30} = {e-10**30} (output type: {type(e-10**30)})")
    print(f"  uint64 + int: {f} + {200} = {f+200} (output type: {type(f+200)})")
    print(f"  uint64 - int: {f} - {200} = {f-200} (output type: {type(f-200)})")
    print(f"  uint64 + int: {f} + 10**30 = {f+10**30} (output type: {type(f+10**30)})")
    print(f"  uint64 - int: {f} - 10**30 = {f-10**30} (output type: {type(f-10**30)})")
    print(
        " [one might keep going, but you get the idea: Python tries to use a result type which has enough range to represent the output]"
    )
    print(" [let's conclude by trying some multiplications too]")
    print(f"  int * int16: 10 * {a} = {10 * a} (output type: {type(10*a)}")
    print(f"  int * uint16: 10 * {b} = {10 * b} (output type: {type(10*b)}")
    print(
        f"  int * int16: 10**30 * {a} = {10**30 * a} (output type: {type(10**30 * a)}"
    )
    print(
        f"  int * uint16: 10**30 * {b} = {10**30 * b} (output type: {type(10**30 * b)}"
    )
    print(
        " [some of the above are intended to show that sometimes the resulting type depends on the value of the input, not only its type]"
    )


# 2. Numpy array creation basics (I)


def ex2_constrbasics1():
    # NOTE: this exercise was intended to be done mainly at the REPL,
    # but here everything is put inside a function to keep things clean

    # pure integers
    lint = [1, 3, 5, 7]
    aint = np.array(lint)
    print()
    print(f"aint = {aint}")
    print(
        f"The element type of np.array({lint}) is {aint.dtype}"
    )  # the .dtype attribute gives the type

    # pure floats
    lfloat = [1.0, 0.0, 2.5]
    afloat = np.array(lfloat)
    print()
    print(f"afloat = {afloat}")
    print(f"The element type of np.array({lfloat}) is {afloat.dtype}")

    # mix of integers and floats
    lmix = [1, 2.0, 3, 4.0]
    amix = np.array(lmix)
    print()
    print(f"amix = {amix}")
    print(f"The element type of np.array({lmix}) is {amix.dtype}")

    # force the result to be floating point with integer inputs
    aforcedfloat = np.array(lint, dtype=float)
    print()
    print(f"aforcedfloat = {aforcedfloat}")
    print(f"The element type of np.array({lint}, dtype=float) is {aforcedfloat.dtype}")

    # force the result to be bool even if the input list is made of floats
    aforcedbool = np.array(lfloat, dtype=bool)
    print()
    print(f"aforcedbool = {aforcedbool}")
    print(f"The element type of np.array({lfloat}, dtype=bool) is {aforcedbool.dtype}")

    # adding strings
    lwithstrings = lmix + ["hello"]
    awithstrings = np.array(lwithstrings)
    print()
    print(f"awithstrings = {awithstrings}")
    print(
        f"The element type of np.array({lwithstrings}) is {awithstrings.dtype} (which is a sort of string-like type, strings are complicated...)"
    )
    print(
        f"The type of one of the elements is type(awithstrings[0]) = {type(awithstrings[0])}"
    )
    print(
        f"Note that isinstance(awithstrings[0], str) = {isinstance(awithstrings[0], str)}"
    )

    # adding other objects, like sets
    lwithsets = lmix + [{0, 1, 2}]
    awithsets = np.array(lwithsets)
    print()
    print(f"awithsets = {awithsets}")
    print(f"The element type of np.array({lwithsets}) is {awithsets.dtype}")

    # retaining the element type even without
    akeeptype = np.array(lmix, dtype=object)
    print()
    print(f"akeeptype = {akeeptype}")
    print(f"The element type of np.array({lmix}, dtype=object) is {akeeptype.dtype}")

    # corner case: empty input list without specifying dtype
    aempty = np.array([])
    print()
    print(f"aempty = {aempty}")
    print(f"The element type of np.array([]) is {aempty.dtype}")


# 3. Numpy array creation basics (II)


def ex3_constrbasics2(n):
    # some explorations with 1-d arrays (vectors)

    a1 = np.ones(n, dtype=float)  # floating points
    print(f" dtype=float, resulting dtype={a1.dtype}, type(a[0])={type(a1[0])}")
    a2 = np.ones(
        n, dtype=int
    )  # "machine" integers, probably a 64-bit in most modern pc's
    print(f" dtype=int, resulting dtype={a2.dtype}, type(a[0])={type(a2[0])}")
    a3 = np.ones(n, dtype=np.int64)  # 64 bit integers, probably the same as before
    print(f" dtype=np.int64, resulting dtype={a3.dtype}, type(a[0])={type(a3[0])}")
    a4 = np.ones(n, dtype=np.int32)  # 32 bit integers
    print(f" dtype=np.int32, resulting dtype={a4.dtype}, type(a[0])={type(a4[0])}")
    a5 = np.ones(n, dtype=bool)  # bool
    print(f" dtype=bool, resulting dtype={a5.dtype}, type(a[0])={type(a5[0])}")
    a6 = np.ones(
        n, dtype=str
    )  # strings (this is kind of stupid, it would have been better if it wasn't defined...)
    print(f" dtype=str, resulting dtype={a6.dtype}, type(a[0])={type(a6[0])}")

    print()

    # move on to matrices

    m1 = np.ones((n, n), dtype=int)  # notice that the first argument is now a tuple
    m2 = 5 * m1  # multiply by an integer
    print(f"integer times an array of integers -> resulting dtype={m2.dtype}")
    m3 = 5.5 * m1  # multiply by a float
    print(f"float times an array of integers -> resulting dtype={m3.dtype}")


# 4. Some broadcasting basics


def ex4_broadcastbasics(n):
    a0 = np.full(n, -1)  # version 0: use the np.full function
    a1 = -np.ones(
        n, dtype=int
    )  # version 1: exploting the "unary minus" (i.e. negation)
    a2 = (-1) * np.ones(n, dtype=int)  # version 2: exploting broadcasted multiplication
    a3 = np.zeros(n, dtype=int) - 1  # version 3: exploting broadcasted subtraction
    a4 = np.ones(n, dtype=int) - 2  # version 3b: same as above basically, just sillier
    return a1


# 5. Reading out array chunks


def ex5_readchunks(l):
    n = len(l)
    a = np.array(l)
    a1 = a[: (n // 2)]  # we need integer division `//` here!
    a2 = a[(n // 2) :]
    b = np.hstack((a1, a2))  # note that the argument is a tuple

    # additional questions
    # Q1
    assert b is not a  # `b` is a different array than `a`
    b[0] += 1  # experiment: change `b[0]`
    assert b[0] != a[0]  # verify that `a[0]` has not changed
    b[0] -= 1  # restore `b[0]` to its original value

    # Q2
    assert np.all(a == b)  # `==` produces an array of bools (a mask)
    #      we need `np.all` to reduce it
    assert np.array_equal(a, b)  # this is actually the preferred way to check
    # for array equality

    # Q3
    a1[0] += 1  # modify an item in the first half
    assert a[0] == a1[0]  # the corresponding item of `a` has changed too!
    assert a1.base is a  # this is because `a1` is a view of `a` (slices produce views)
    assert (
        a2.base is a
    )  # the same is true for `a2` (notice the use of `is` instead of `==`!!)
    a1[0] -= 1  # restore `a[0]` to its original value

    return b


# 6. Slicing in strides (I)


def ex6_slicestrides1(l):
    n = len(l)
    a = np.array(l)
    a1 = a[::2]  # even-indexed elements
    a2 = a[1::2]  # odd-indexed elements

    # additional question
    a1[0] += 1  # change `a1[0]`
    assert a[0] == a1[0]  # `a[0]` has changed because `a1` is a view of `a`
    assert (
        a1.base is a
    )  # another way to see that a1 is a view of a (note that we use `is`)
    a1[0] -= 1  # restore `a[0]` to its original value

    a2[0] += 1  # same with `a2`
    assert a[1] == a2[0]  # the first odd element of `a` is `a2[0]`...
    assert a2.base is a
    a2[0] -= 1

    return a1, a2


# 7. Slicing in strides (II)


def ex7_slicestrides2(l):
    n = len(l)
    a = np.array(l)

    # Q1
    a1 = a[::-1]  # a step of -1 goes in reverse
    assert a1.base is a  # we get a view

    # Q2
    a2 = a[(n // 2 - 1) :: -1]  # first half, reversed: since the step is negative
    #   we start from the middle
    a3 = a[: (n // 2 - 1) : -1]  # second half, reversed: since the step is negative
    #   we start from the end and stop in the middle

    b2 = a[: (n // 2)]  # for checking purposes, we take the first half as in ex. 5
    c2 = b2[::-1]  # then we revert it
    assert np.array_equal(c2, a2)  # we check our expression for `a2`.
    # Note that here we used n//2 while in `a2`
    # we used `n//2-1`!! Make sure you understand why!

    b3 = a[(n // 2) :]  # for checking purposes, we take the second half as in ex. 5
    c3 = b3[::-1]  # then we revert it
    assert np.array_equal(c3, a3)  # we check our expression for `a3`.
    # Note that here we used n//2 while in `a3`
    # we used `n//2-1`!! Make sure you understand why!

    # Q3
    a4 = a[
        n - 2 + n % 2 :: -2
    ]  # even-indexed values, in reverse. We need to start from the last even index!
    a5 = a[
        n - 1 - n % 2 :: -2
    ]  # odd-indexed values, in reverse. We need to start from the last odd index!

    # explanation: consider `a4`. If `n` is even, the last index is `n-1`, which is odd, and then we need
    #     to start from `n-2` since we want the last even index. If `n` is odd, however, the last element
    #     `n-1` is even, which is what we want. The expression `n%2` is 0 for even `n` and 1 for odd `n`.
    #     Therefore, the expression `(1-n%2)` is 1 for even `n` and 0 for odd `n`. Therefore, we can
    #     say that in general we want to start from `n-1-(1-n%2)`, which simplifies to `n-2+n%2`.
    #
    #     now consider `a5`. It's the same except that we want to start from `n-1` if `n` is even, and
    #     from `n-2` if `n` is odd (the opposite of the previous case). So instead of subtracting `(1-n%2)`
    #     we just subtract `n%2`. The resulting expression is `n-1-n%2`.

    b4 = a[::2]  # for checking purposes, we take the even-indexed items as in ex. 6
    c4 = b4[::-1]  # then we revert it
    assert np.array_equal(c4, a4)  # we check our expresison for a4

    b5 = a[1::2]  # for checking purposes, we take the odd-indexed items as in ex. 6
    c5 = b5[::-1]  # then we revert it
    assert np.array_equal(c5, a5)  # we check our expresison for a5

    # Q4
    a6 = a1[::-1]  # the reverse of the reverse of `a`
    assert np.array_equal(a6, a)  # it should clearly be equal to `a`
    assert a6 is not a  # it's not identical to `a` hoewever: it's not the same object
    assert (
        a6.base is a
    )  # in fact, `a6` is a view of `a` (which happens to "view" everything...)
    a6[0] += 1  # indeed, if we change a6...
    assert a6[0] == a[0]  # ...then `a` is also changed
    a6[0] -= 1  # we restore the original value
    a7 = a[
        :
    ]  # here is another way to get exactly the same thing, i.e. a "complete view"
    assert np.array_equal(a7, a6)  # and indeed they are equal
    assert (
        a7 is not a6
    )  # and still they are not identical: they are 2 distinct views which happen
    #   to operate on the same exact data in memory


# 8. Indexing with lists (I)


def ex8_indexinglists1(l):
    n = len(l)
    a = np.array(l)

    # we're assuming n to be odd here:
    assert n % 2 == 1
    # therefore the index of the middle element is n//2
    a1 = a[
        [0, n // 2, n - 1]
    ]  # notice that the outer brackets are for indexing, the inner brackets create a list!

    # explanation: the above is equivalent to:
    # s = [0, n//2, n-1]
    # a1 = a[s]

    # the result is also the same as this (but performed more efficiently):
    # s = [0, n//2, n-1]
    # l1 = [l[i] for i in s]   # this is how you would do more or less the same with Python lists
    # a1 = np.array(l1)

    # checks

    assert len(a1) == 3  # The result will always have 3 elements no matter what.
    # You should check that this is true even if `len(a) == 1`.

    a1[0] += 1  # if we change a1...
    assert a1[0] != a[0]  # ...`a` does not change!
    assert a1.base is None  # ...because `a1` is *not* a view of `a`
    assert a1.base is not a  # ...(another way to see this)
    a1[0] -= 1  # let's restore `a1`

    return a1


# 9. Indexing with lists (II)


def ex9_indexinglists2(l, m):
    n = len(l)
    a = np.array(l)

    inds = np.random.randint(n, size=m)  # ...or np.random.randint(0, n, m)
    a1 = a[inds]
    assert len(a1) == m  # The result must always have length `m`, even when `m > n`

    # Explicit check of all the elements
    for i in range(m):
        assert (
            a1[i] == a[inds[i]]
        )  # This property is what defines the indexing with lists

    # In fact, we can do the same check as above in one line, although a little obfuscated
    assert np.array_equal(a1, np.array([a[inds[i]] for i in range(m)]))

    # Well, the above line can be made even shorter: if we don't explicitly construct an array
    # and compare `a1` with `somelist`, then `somelist` will be automatically converted into an
    # array anyway... (this is true even when you use `==`, e.g. if you did
    # `np.all(a1 == somelist)`)
    assert np.array_equal(a1, [a[inds[i]] for i in range(m)])

    # An alternative way using `np.all`, which is fine here (but generally using
    # np.array_equal is better):
    assert np.all(a1 == [a[inds[i]] for i in range(m)])

    # This one is the same as `np.all`: you can call the `.all` method on the result
    assert (a1 == [a[inds[i]] for i in range(m)]).all()

    return a1


# 10. Indexing with masks (I)


def ex10_indexingmasks1(l):
    n = len(l)
    a = np.array(l)

    a1 = a[a > 0]  # read out the positive elements only

    # explanation: a > 0 returns an array of bools which is True only at indices where `a` is non-negative
    #              using a bool mask inside an indexing expression returns only those elements where the mask
    #              is True. So in this case only the non-negative elements.

    # checks (same as exercise 8)

    assert len(a1) > 0  # these checks don't work if there are no non-negative numbers!!
    a1[0] += 1  # if we change a1...
    assert a1[0] != a[0]  # ...`a` does not change!
    assert a1.base is None  # ...because `a1` is *not* a view of `a`
    assert a1.base is not a  # ...(another way to see this)
    a1[0] -= 1  # let's restore `a1`

    return a1


# 11. Indexing with masks (II)


# first part of the exercise
def ex11_indexingmasks2(l, r):
    n = len(l)
    a = np.array(l)

    s = np.random.rand(n)  # `n` random values uniformly distrivuted in [0,1)
    mask = s < r  # broadcasting happens here; the result is a mask
    a1 = a[mask]

    # explanation: if you have a random uniform number in [0,1) the probability
    #   that it is less than `r` is just `r`. So if you create a random vector
    #   of uniform numbers in [0,1) and compare it to `r` you get an output
    #   vector of bools (a mask) in which each element is `True` with
    #   probability `r`. If you then use this mask to index another vector
    #   of the same length, the effect is that each element will be chosen
    #   with probability `r`, and all chosen elements will end up in a new
    #   resulting vector.

    # checks
    if r == 0:
        assert len(a1) == 0
    elif r == 1:
        assert np.array_equal(a1, a)

    # if we sum up all the items in the mask `mask` we are counting the number
    # of `True` values in there (when you sum bools, they behave like 0 and 1):
    assert len(a1) == mask.sum()

    return a1


# second part of the exercise
def ex11_sampling(l, r, iters=1000):
    k = 0
    for iter in range(iters):
        k += len(ex11_indexingmasks2(l, r))
    avglen = k / iters
    expected = len(l) * r
    print(f"average length with {iters} samples = {avglen}, expected = {expected}")


# 12. Automatic conversions


def ex12_autoconvert():
    # array of floats
    print()
    print("Assign integers to a float array:")
    afloat = np.zeros(5)  # array of floats
    print(f"  before assignment: {afloat[0]}")
    afloat[0] = 55  # assign a normal (small) integer
    print(f"  after assignment (small int): {afloat[0]}")
    afloat[0] = (
        17 * 10**307
    )  # assign a large integer which still fits in the range of floats
    print(f"  after assignment (large-ish int): {afloat[0]}")
    # the next one will fail
    try:
        afloat[0] = (
            19 * 10**307
        )  # assign a large integer which *doesn't* fit in the range of floats
        print(f"  after assignment (too-large int): {afloat[0]}")
    except:
        print(f"  assignment of out-of-range integer failed (as expected)")

    # array of ints
    print()
    print("Assign floats to an integer array:")
    aint = np.zeros(
        5, dtype=int
    )  # array of ints (defults to int32 or int64 depending on machine/OS/python;
    #   just change `int` to `np.int16` or `np.int64` here to experiment with
    #   other types)
    print(f"  before assignment: {aint[0]}")
    aint[0] = 3.0
    print(f"  after assignment (small integer-valued float): {aint[0]}")
    aint[0] = 5.7
    print(f"  after assignment (small non-integer-valued float): {aint[0]}")
    # the next ones will fail
    try:
        aint[0] = 1e19  # this is too large to fit into the int64 range
        print(f"  after assignment (large float): {aint[0]}")
    except:
        print(f"  assignment of out-of-range float failed (as expected)")
    try:
        aint[0] = np.inf  # this is way too large to fit into the int64 range
        print(f"  after assignment (np.inf): {aint[0]}")
    except:
        print(f"  assignment of np.inf failed (as expected)")
    try:
        aint[0] = np.nan  # this is not even a number, whay can you expect...
        print(f"  after assignment (np.nan): {aint[0]}")
    except:
        print(f"  assignment of np.nan failed (as expected)")

    # array of bools
    print()
    print("Assign floats to a bool array:")
    abool = np.zeros(5, dtype=bool)  # array of bools
    print(f"  before assignment: {abool[0]}")
    abool[0] = 3.0
    print(f"  after assignment (small non-zero integer-valued float): {abool[0]}")
    abool[0] = 5.7
    print(f"  after assignment (small non-integer-valued float): {abool[0]}")
    abool[0] = 0.0
    print(f"  after assignment (zero float): {abool[0]}")
    abool[0] = 1e-10
    print(f"  after assignment (small, almost zero, float): {abool[0]}")
    abool[0] = 1e19  # too large for the int64 range, but as a bool: no problem
    print(f"  after assignment (large float): {abool[0]}")
    abool[0] = np.inf  # infinity as a bool? no problem, it's just True (of course)
    print(f"  after assignment (np.inf): {abool[0]}")
    abool[0] = np.nan  # this is not a number, but it can certainly be a bool...
    print(f"  after assignment (np.nan): {abool[0]}")

    # NOTES:
    #  * assigning ints to an array of bools works the same, anything nonzero is True.
    #  * assigning bools to arrays of ints/floats just converts them to 0/1 or 0.0/1.0


# 13. Assigning with slices


def ex13_assignslices(l):
    n = len(l)
    a = np.array(l)
    a1 = a[: (n // 2)]
    a2 = a[(n // 2) :]  # same as ex. 5 up to here

    a[: (n // 2)] = a1[
        ::-1
    ]  # assign the reversed first half into the first half of `a`
    a[(n // 2) :] = a2[
        ::-1
    ]  # assign the reversed second half into the second half of `a`

    # of course, it can be done in just two lines, but see ex. 7 part 2
    # a[:(n//2)] = a[(n//2-1)::-1] # revert the first half
    # a[(n//2):] = a[:(n//2-1):-1] # revert the second half

    # strictly speaking, this one is actually equivalent to the first version above...
    # ...it performs two indexing steps in sequence. This is not good style...
    # a[:(n//2)] = a[:(n//2)][::-1] # revert the first half
    # a[(n//2):] = a[(n//2):][::-1] # revert the second half

    print(f"a = {a}")

    # checks

    # first half
    for i in range(n // 2):
        assert a[i] == l[n // 2 - 1 - i]
    # second half (notice the range using `(n+1)//2`: if `n` is even, nothing changes,
    # but if `n` is odd it goes up one index further than the previous cycle
    for i in range((n + 1) // 2):
        assert a[-i] == l[n // 2 - 1 + i]

    # alternative version: use reversion in the assignment insted (see also ex. 7 part 2)

    a[(n // 2 - 1) :: -1] = a[: (n // 2)]  # revert the first half
    a[: (n // 2 - 1) : -1] = a[(n // 2) :]  # revert the second half

    print(f"a = {a}")

    assert np.array_equal(a, l)  # after the double reversion, we should have
    # gone back to the original situation


# 14. Assigning with lists


def ex14_assignslices(l):
    n = len(l)
    a = np.array(l)

    # see exercise 8
    assert n % 2 == 1
    a[[0, n // 2, n - 1]] = [
        0,
        1,
        2,
    ]  # note: using just `-1` instead of `n-1` would also work

    if n > 1:
        assert a[0] == 0  # this would fail in the case n == 1 (why?)
        assert a[n // 2] == 1  # same here
    assert a[-1] == 2  # this one always succeeds

    # alternative version which has the same effect except in the case `n==1`,
    #   in which case it gives precedence to the middle element instead of the
    #   last one:
    a[[0, -1, n // 2]] = [0, 2, 1]

    if n > 1:
        assert a[0] == 0
        assert a[-1] == 2
    assert a[n // 2] == 1  # now this is the case which is always true

    return a


# 15. Assigning with masks


def ex15_assignmasks(l):
    n = len(l)
    a = np.array(l)

    # see ex. 10
    mask = a < 0  # which values are negative?
    a[mask] = -a[mask]  # flip their sign and put them back

    assert np.all(a >= 0)  # now everything is non-negative (notice the `>=` though!!)
    assert not np.any(a < 0)  # same test done using `any` instead of `all`

    return a


# 16. Broadcasted assignment


def ex16_broadcastassign1(n):
    a = np.zeros(n)
    a[1::2] = 1  # the integer 1 here will be converted to a float, and broadcasted
    # alternative version:
    # a = np.ones(n)
    # a[::2] = 0
    return a


# 17. Broadcasted assignment (II)


def ex17_broadcastassign2(n):
    a = np.zeros(n)  # variants: `a = np.zeros(n, dtype=int)` or `dtype=bool`
    a[: (n // 2)] = 1  # the number 1 here will be converted as appropriate to a
    # float, an integer or a bool
    return a


# 18. More array creation tricks, and some plotting


def ex18_ranges(n):
    # array that excludes the endpoint
    a = np.arange(0.0, 2 * np.pi, 2 * np.pi / n)
    assert len(a) == n

    # array that includes the endpoint
    b = np.linspace(0.0, 2 * np.pi, n)
    assert len(b) == n

    # compute the sin and cos. They are numpy versions thus they broadcast.
    # (this wouldn't work with `math.sin` for example)
    s = np.sin(b)
    c = np.cos(b)

    # Here starts the plotting.

    # clear up previous plots, so that we can call this function repeatedly
    plt.clf()

    # plot the two curves. We use 2 separate commands to be able to pass the
    # `label` option; otherwise we could have used plt.plot(b,s,b,c)
    plt.plot(b, s, label="sin")
    plt.plot(b, c, label="cos")

    # set the x axis label
    plt.xlabel("radians")

    # turn on the legend
    plt.legend()

    # save the figure to a file (this goes inside the current directory, which
    # you get from `os.getcwd()`)
    plt.savefig("sincos.png", dpi=300)


# 19. Normal random numbers

# we need to alias the random module, not to get it
# confused with numpy.random. We call it pyrandom.
import random as pyrandom


def ex19_randnslow():
    return [pyrandom.gauss(0.0, 1.0) for i in range(10**6)]


def ex19_randnfast():
    return random.randn(10**6)


# we will use a very crude and imprecise way of timing our code.
# However, it is simple, and sufficient for our needs.
import time


def ex19_timings():
    t0 = time.time()
    ex19_randnslow()
    t1 = time.time()
    print("python:", t1 - t0)

    t0 = time.time()
    ex19_randnfast()
    t1 = time.time()
    print("numpy:", t1 - t0)


# 20. Normal random numbers, 2-d arrays, computing statistics and more


# Auxiliary function for the "extra" part of this exercise:
# the Gaussian distribution with mean `mu` and standard deviation `sigma`
def gaussian(x, mu, sigma):
    z = (x - mu) / sigma
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(z**2) / 2)


def ex20_gaussstats(samples, bins=40):
    # set rows/columns
    n, m = 10, 20

    # total number of elements of the table
    t1 = n * m
    # total number of elements of the even-indexed sub-table
    # (notice the integer divisions!)
    t2 = (n // 2) * (m // 2)

    # prepare two arrays for collecting the samples
    # (this is because we should always try to avoid growing arrays
    # dynamically if possible!)
    m1 = np.zeros(samples)
    m2 = np.zeros(samples)
    for i in range(samples):
        # generate the random table with Gaussian entries
        a = random.randn(n, m)
        # get the even-indexed sub-table (cf. ex. 6)
        s = a[::2, ::2]
        # collect the two means (`np.mean(a)` would also work here)
        m1[i] = a.mean()
        m2[i] = s.mean()

    # Now that we have collected the results, we can plot some histograms

    # clear up previous plots, so that we can call this function repeatedly
    plt.clf()
    # plot the two histograms
    # the bins are controlled by the optional argument to our function
    # the `histtype` is so that we can see both histograms clearly
    # the `density` option is to get a probability density distribution (see below)
    plt.hist(m2, bins=bins, density=True, histtype="step")
    plt.hist(m1, bins=bins, density=True, histtype="step")

    # Here comes the "extra" part - we compute the expected values of the
    # histograms, which we can compare to the observed ones.

    # the theoretical probability distributions are Gaussians centered in 0
    # and with standard deviation 1 / sqrt(number of samples)
    s1 = 1 / np.sqrt(t1)
    # we generate evenly-spaced points between -5sigma and 5sigma
    # (1000 points are more than enough to get smooth curves)
    x1 = np.linspace(-5 * s1, 5 * s1, 1000)
    # note that we can use the `gaussian` function over the `x1` array directly,
    # since in that function we used the numpy version of `sqrt` and `exp` (and
    # no `for` loops)
    y1 = gaussian(x1, 0.0, s1)

    # same as above for the sub-table
    s2 = 1 / np.sqrt(t2)
    x2 = np.linspace(-5 * s2, 5 * s2, 1000)
    y2 = gaussian(x2, 0.0, s2)

    # superimpose the curves on the previous histograms
    # (with 1000 samples it already looks sort of ok, with 10**4 samples it's
    #  very nice)
    plt.plot(x1, y1, x2, y2)


# 21. More 2-d array indexing patterns questions


def ex21_indexingmore(a):
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise Exception("argument must be a 2-d array")

    # we need at least two lines for this exercise
    a[::2, ::2] = 0  # set the even rows/cols
    a[1::2, 1::2] = 0  # set the odd rows/cols

    # [Extra puzzle]
    # if `a` is not a view and if the number of columns of `a` is odd,
    # we can use a trick. To check whether `a` owns its own data
    # (is not a view), we need to check the `.base` attribute. Actually,
    # we also should check that the internal layout is C-like (i.e. row-major),
    # which can be read out from an internal flag.

    # if a.base is None and a.shape[1] % 2 == 1 and a.flags.c_contiguous:
    #     a.ravel()[::2] = 0

    # Note: if `a` is a view, nasty/unexpected things could happen...


# 22. Stacking


def ex22_stacking(n):
    l = [np.arange(i, i + 5) for i in range(n)]
    return np.vstack(l)  # part 1
    # return np.hstack(l) # part 2


# 23. A more advanced broadcasting puzzle


def ex23_advancedbroadcast(n):
    # reshape version
    return np.arange(5) + np.arange(n).reshape(n, 1)
    # newaxis version
    # return np.arange(5) + np.arange(n)[:,np.newaxis]


# 24. Sieve of Eratosthenes (I)


def ex24_sieve1(n):
    a = np.ones(n, dtype=bool)  # point 1
    a[0:2] = False  # point 2
    for i in range(2, int(np.sqrt(n - 1)) + 1):  # point 3a (loop between 2 and √(n-1)
        a[i**2 :: i] = False  # point 3b (set the multiples of i to False)
        #   start from the next multiple after `i`, i.e. `i**2`
        #   proceed in steps of `i`
        #   note that the assignment is broadcasted (scalar assigned to array)
    return a
    # If you want to return a list of indices instead of a bool mask:
    # return np.arange(n)[a]

    # Return the list of indices, alternative version (I):
    # return np.flatnonzero(a)

    # Return the list of indices, alternative version (II):
    # return np.argwhere(a).ravel()


# 25. Sieve of Eratosthenes (II) - Advanced


def ex25_sieve2(n):
    a = np.ones(n, dtype=bool)
    a[0:2] = False
    i = 2
    while i <= int(np.sqrt(n - 1)) + 1:  # Use a `while` loop this time
        a[i**2 :: i] = False  # Same as before
        i = (
            np.argmax(a[i + 1 :]) + i + 1
        )  # This time, we find the next prime and advance more quickly

    # Explanation of the last line:
    #   we have just decided that `i` is prime. Then `a[i+1:]` is a view of
    #   everything that follows. Calling `argmax` on that returns the position
    #   of the first element that is `True`, because in a bool array the max
    #   is always `True` and argmax returns the first occurrence of the max.
    #   (So this is basically just an ugly trick...)
    #   The reason for adding `i+1` is that `argmax` will return the index of
    #   the first `True` in the view. But the view starts from `i+1`. So to get
    #   the index of the same element in the original array we need to shift it.

    # Return the list of primes (as an array)
    return np.flatnonzero(a)


# Extra: the no-numpy (slow) version of the sieve
def ex25_sievewithlists(n):
    a = [True for i in range(n)]
    a[0:1] = [False, False]
    i = 2
    while i <= int((n - 1) ** 0.5) + 1:
        for j in range(i**2, n, i):
            a[j] = False
        i += 1
        while not a[i]:
            i += 1
    # return the list of primes
    return [i for i in range(n) if a[i]]


# 26. Conway's game of life - Very advanced (!)


# First version (a little more intuitive).
#
# This computes the neighbors of each cell in turn,
# and updates the new configuration according to the
# rules. For this to work, `cnew` must be equal to `c` at
# the beginning.
def conway1step(c, cnew):
    n, m = c.shape
    for i in range(n):
        x1, x2 = max(i - 1, 0), min(i + 2, n)  # x borders
        for j in range(m):
            y1, y2 = max(j - 1, 0), min(j + 2, m)  # y borders
            neighb = c[x1:x2, y1:y2].sum() - c[i, j]  # num. of neighbors
            if c[i, j] and (neighb < 2 or neighb > 3):
                cnew[i, j] = 0
            elif not c[i, j] and neighb == 3:
                cnew[i, j] = 1


def ex26_gameoflife(c, n, pause=1e-3):
    cnew = c.copy()
    for i in range(n):
        conway1step(c, cnew)
        c[:, :] = cnew
        plt.clf()
        plt.imshow(c)
        plt.pause(pause)
    return c


# Second version (a little more numpy-ish; much faster for large n)
#
# This computes the neighbors all at once in a separate array.
# It does this by summing the configuration c shifted by 1 cell in one
# of eight possible directions (north-west, north, nort-east, west, east,
# south-west, south, south-east).
# The problems come from the indices, which are a bit tricky.
#
# At the end, the game of life rules are implemented by filtering the neighbors
# array through some conditions (e.g. neighb == 3) and then applying those
# rules where appropriate by logical operations (The symbol `|` denotes an `or`
# operation; the symbol `&` an `and` operation, the symbol `~` a `not`
# operation). Note that this goes through the creation of some temporary
# arrays, therefore speed could be improved further by avoiding that.
def conway1step2(c, neighb):
    n, m = c.shape
    # reset the neighbors
    neighb.fill(0)
    # compute neighbors by accumulation; see comments below
    for dx in [-1, 0, 1]:
        x1, x2 = max(dx, 0), min(dx, 0)
        for dy in [-1, 0, 1]:
            y1, y2 = max(dy, 0), min(dy, 0)
            if dx == 0 and dy == 0:
                continue  # avoid the center
            neighb[x1 : (n + x2), y1 : (m + y2)] += c[-x2 : (n - x1), -y2 : (m - y1)]

    # the above loop, unrolled and spelled out, does this:
    #
    # neighb[0:n-1, 0:m-1] += c[1:n  , 1:m  ]
    # neighb[0:n-1,  :   ] += c[1:n  ,  :   ]
    # neighb[0:n-1, 1:m  ] += c[1:n  , 0:m-1]
    # neighb[ :   , 0:m-1] += c[ :   , 1:m  ]
    # neighb[ :   , 1:m  ] += c[ :   , 0:m-1]
    # neighb[1:n  , 0:m-1] += c[0:n-1, 1:m  ]
    # neighb[1:n  ,  :   ] += c[0:n-1,  :   ]
    # neighb[1:n  , 1:m  ] += c[0:n-1, 0:m-1]

    # with the neighbors, we can compute the two conditions:
    # under/over-populated regions
    kill = (neighb < 2) | (neighb > 3)
    # breeding regions
    generate = neighb == 3

    # apply the rules (note that these work well because the kill and generate
    # regions do not overlap)
    c &= ~kill  # means "only keep the non-killing regions"
    c |= generate  # means "add the newborns (if not there already)"

    # alternative version (but only works well if c has int elements)
    # c += (1 - c) * generate - c * kill


def ex26_gameoflife2(c, n, pause=1e-3):
    neighb = np.zeros_like(c, dtype=int)
    for i in range(n):
        conway1step2(c, neighb)
        plt.clf()
        plt.imshow(c)
        plt.pause(pause)
    return c
