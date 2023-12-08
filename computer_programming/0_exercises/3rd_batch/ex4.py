import numpy as np


def subsum(v: list, s: int):
    # checks
    if not np.any(np.greater_equal(v, 0)):
        raise Exception("v must be >= 0")

    n = len(v)
    m = np.zeros((s + 1, n + 1), dtype=bool)
    m[0, :] = True

    # init whence
    whence = np.zeros((s + 1, n + 1), dtype=int) - 1

    # fwd pass
    for ss in range(1, s + 1):
        for j in range(1, n + 1):
            # print(f"ss: {ss}, item: {v[j-1]}")
            rest = ss - v[j - 1]

            if rest >= 0 and m[rest, j - 1]:
                m[ss, j:] = True
                whence[ss, j:] = j - 1
                break

    # backward pass
    if not m[-1, -1]:
        return None

    prev_idx = whence[-1, -1]
    idxs = []
    st = s

    while prev_idx != -1:
        idxs.append(prev_idx)
        rest = st - v[prev_idx]
        prev_idx = whence[rest, prev_idx]

    print(f"SUM: {s}")
    print(m)
    return idxs


def test_subsum():
    v0, s0 = [3, 7, 2, 10], 5  # true
    v1, s1 = [3, 7, 2, 10], 17  # true
    v2, s2 = [3, 7, 2, 10], 10  # true
    v3, s3 = [3, 7, 2, 10], 21  # false

    idx0 = subsum(v0, s0)
    assert isinstance(idx0, list)
    assert np.take(v0, idx0).sum() == s0

    idx1 = subsum(v1, s1)
    assert isinstance(idx1, list)
    assert np.take(v1, idx1).sum() == s1

    idx2 = subsum(v2, s2)
    assert isinstance(idx2, list)
    assert np.take(v2, idx2).sum() == s2

    idx3 = subsum(v3, s3)
    assert isinstance(idx1, list)
    assert idx3 is None


test_subsum()
