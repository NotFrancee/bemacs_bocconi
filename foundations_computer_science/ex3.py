def mysum(m, n):
    s = 0

    for i in range(m, n):
        if i % 5 == 0 or i % 3 == 0:
            continue
        s += i

    return s


def hammingdist(l1, l2):
    if not (isinstance(l1, list) and isinstance(l2, list)):
        raise Exception("must be lists")
    if not len(l1) == len(l2):
        raise Exception("lenghts must be equal")

    dist = 0
    for a, b in zip(l1, l2):
        if a != b:
            dist += 1
    return dist


def localmin(l):
    def is_min_rx(i, rx=True):
        offset = i + 1 if rx else i - 1

        while 0 <= offset < len(l):
            r = l[offset]
            print(offset)
            if r != l and r > l[i]:
                return True

            offset += 1 if rx else -1

        return False

    mins = []
    for i in range(len(l)):
        # analyze to the left and right
        if is_min_rx(i, rx=True) and is_min_rx(i, rx=False):
            mins.append(i)

    return mins


# print(mysum(4, 10))
# print(hammingdist([1, 2, 3], [5, 6, 3]))
print(localmin([5, 4, 2, 3, 5, 4]))
