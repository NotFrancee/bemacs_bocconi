#%% Ex. 1 - skip multiples

def mysum(m, n):
    s = 0
    for i in range(m, n):
        if i % 3 == 0 or i % 5 == 0:
            continue
        else:
            s += i
    return s

assert mysum(0, 4) == 3
assert mysum(3, 10) == 19

#%% Ex. 2 - hamming distance

def hammingdist(l1, l2):
    if type(l1) != type(l2):
        raise TypeError("l1 and l2 must be of the same type")
    if len(l1) != len(l2):
        raise ValueError("l1 and l2 must be of the same length")
    s = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            s += 1
    return s
    

# hammingdist("abc", ['a','b','c']) # THIS MUST GIVE AN ERROR (WRONG TYPES)
# hammingdist([1, 2, 3], 123) # THIS MUST GIVE AN ERROR (WRONG TYPES)
# hammingdist([1, 2, 3], [1, 4, 2, 3]) # THIS MUST GIVE AN ERROR (WRONG LENGTHS)

assert hammingdist("castle", "battle") == 2
assert hammingdist([0, 3, 2, 1, 0], [0, 1, 2, 3, 1]) == 3
assert hammingdist([], []) == 0

#%% Ex. 3  - Local minima of a list

def localmin(l):
    n = len(l)
    if n == 1:
        return [0]

    minima = []
    for i in range(n):
        if i == 0:
            if l[i] <= l[i+1]:
                minima.append(i)
        elif i == n-1:
            if l[i] <= l[i-1]:
                minima.append(i)
        else:
            if (l[i] <= l[i-1]) and (l[i] <= l[i+1]):
                minima.append(i)
    return minima

assert localmin([]) == []           #  EMPTY LIST: NO MINIMA
assert localmin([5]) == [0]        # 1-ELEMENT LIST: ALWAYS RETURNS [0]
assert localmin([3, 2, 0, 1, 5, 8, 7]) == [2, 6]
assert localmin([1, 3, 3, 5, 9]) == [0, 2]       #CF. EXAMPLE ABOVE
assert localmin([2, 2, 2, 2]) == [0, 1, 2, 3]   # CONSTANT: ALL ELEMENTS ARE MINIMA


#%% Ex 4 - Relative primes

def relprime(a, b):
    for k in range(2, min(a, b)+1):
        if a % k == 0 and b % k == 0:
            return False
    return True

assert relprime(3, 5)    
assert not relprime(6, 4)
assert relprime(15, 26)
assert not relprime(27, 9)
assert relprime(1, 6)


#%% Ex. 5 - Sublist


def issublist(l1, l2):
    n1, n2 = len(l1), len(l2)

    ## The idea is: at each position in l2, check if we find l1 starting there.
    ## The range limit is because after a point, towards the end, there is surely
    ## not enough room for l1 anyway. This ensures that we don't run out of bounds
    ## below when we access l2[i2+i1] (notice that i2+i1 < (n2-n1)+(n1) == n2).

    for i2 in range(n2 - n1 + 1):      # i2 is the position in l2 where l1 may start
        ok = True                      # did we find l1? assume we did, and then check.
        for i1 in range(n1):           # for each element of l1...
            if l1[i1] != l2[i2 + i1]:  # ...if we spot a difference
                ok = False             # ...then we didn't really find it
                break                  # ...we can stop looking (at this i2)
        if ok:                         # if we didn't find any difference...
            return True                # ...then in fact we did find l1!
    return False                       # no luck: l1 is not there

assert issublist([], [])
assert issublist([], [1,2,3])
assert issublist([1,2], [1,2,3,2,1,0])
assert not issublist([1,3], [1,2,3,2,1,0])
assert issublist([2,1,0], [1,2,3,2,1,0])
assert not issublist([1,2,3,4], [1,2,3])
assert issublist([2,3,2,1], [1,2,3,2,1,0])
assert not issublist([2,4,2,1], [1,2,3,2,1,0])
assert issublist([2,4,1], [2,4,1])

# %%
