# %% EXERCISE  1.1 - Fold


def fold(f, x0, n):
    if not isinstance(n, int) or n < 0:
        raise Exception("error must be int and positive")

    prev = x0

    for i in range(n):
        prev = f(prev)

    return prev


## 1.1 tests
from math import sin, cos

assert fold(sin, 0.1, 0) == 0.1
assert fold(sin, 0.1, 1) == sin(0.1)
assert fold(sin, 0.1, 2) == sin(sin((0.1)))
assert fold(cos, 0.1, 3) == cos(cos(cos(0.1)))


# %% EXERCISE  1.2 - Softmax
import math


def softmax_inplace(l):
    s = l.copy()
    den = 0
    for i in range(len(l)):
        den += math.exp(l[i])

    for i in range(len(l)):
        l[i] = math.exp(l[i]) / den

    # CAN I COMPUTE THE EXPs only ONCE?
    print(l)
    for i in range(len(l)):
        s[i] = math.exp(s[i]) / sum(s[:i])
        print(s[i], l[i])

    print(s)


def softmax_inplace_V2(l):
    den = 0
    for i in range(len(l)):
        l[i] = math.exp(l[i])
        den += l[i]

    for i in range(len(l)):
        l[i] /= den


## 1.2 tests

l = [1, 1, 1]
softmax_inplace(l)
assert abs(l[0] - 1 / 3) < 1e-8
assert abs(l[1] - 1 / 3) < 1e-8
assert abs(l[2] - 1 / 3) < 1e-8

# l = [1, 2, 3.5]
# softmax_inplace(l)
# assert abs(l[0] - 0.0628900) < 1e-5
# assert abs(l[1] - 0.17095278) < 1e-5
# assert abs(l[2] - 0.76615720) < 1e-5

# %% EXERCISE 1.3 - Longest Ramp


## 1.3 tests

# assert longestramp([]) == (0, 0)
# assert longestramp([1]) == (0, 1)
# assert longestramp([5, 6, 7]) == (0, 3)
# assert longestramp([1, 3, 1, 2, 3]) == (2, 5)  # TWO RAMPS, THE SECOND WINS
# assert longestramp([2, 3, 3, 1, 6, 1, 1, 3]) == (0, 3) # [2,3,3] COMES BEFORE [1,1,3]
# assert longestramp([3, 2, 5, 5, 1, 1, 4, 5, 2, 6]) == (4, 8)
