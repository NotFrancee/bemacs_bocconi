import numpy as np

test_w1 = np.array(
    [[1, 1, 1, 1], [2, 1, 2, 3], [0, 1, 2, 0], [0, 0, 0.5, 3], [2, 1, 1, 0]]
)
test_w2 = np.array([[1, 3, 0], [2, 1, 7], [4, 2, 3], [1, 4, 2]])


def best_path(w: np.ndarray):
    # check matrix
    if not np.all(np.greater(w.shape, 0)):
        raise Exception("Matrix must be non-empty and 2dimensional")

    if np.any(w < 0):
        raise Exception("all values must be non-negative")

    n, m = w.shape

    c = np.zeros((n, m))
    c[0, :] = w.cumsum(axis=1)[0, :]
    c[:, 0] = w.cumsum(axis=0)[:, 0]

    whence = np.zeros((n, m))
    path = np.zeros(n + m - 2)

    # forward pass
    for i in range(1, n):
        for j in range(1, m):
            prev_costs = [c[i - 1, j], c[i, j - 1]]
            c[i, j] = w[i, j] + np.min(prev_costs)

    print(c)
    # backward pass

    # checks
    assert len(path == 1) == n - 1
    assert len(path == -1) == m - 1

    # check costs actually sum to the bottom right value


best_path(test_w1)
