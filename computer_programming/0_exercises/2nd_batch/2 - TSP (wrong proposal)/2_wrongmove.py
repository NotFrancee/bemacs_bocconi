import numpy as np
import matplotlib.pyplot as plt


def sample_correct(n):
    while True:
        e1 = np.random.randint(n)
        e2 = np.random.randint(n)

        if e2 < e1:
            e1, e2 = e2, e1

        if e2 > e1 + 1 and not (e1 == 0 and e2 == n - 1):
            break

    return e1, e2


def sample_wrong(n):
    e1 = np.random.randint(n - 2)
    e2 = np.random.randint(e1 + 2, n)

    return e1, e2


def display(correct_data, biased_data, n):
    fig, (row1, row2) = plt.subplots(2, 2)

    for i, ax in enumerate(row1):
        ax.hist(correct_data[i], bins=n - 1)

        if i == 0:
            ax.set_ylabel("Correct Sampl. Method")

    for i, ax in enumerate(row2):
        ax.hist(biased_data[i], bins=n - 1)

        if i == 0:
            ax.set_ylabel("Biased Sampl. Method")
    plt.show()


def test_methods(iters):
    n = 20
    correct_samples = np.zeros((2, iters))
    biased_samples = np.zeros((2, iters))
    for i in range(iters):
        move_correct = sample_correct(n)
        move_biased = sample_wrong(n)

        correct_samples[:, i] = move_correct
        biased_samples[:, i] = move_biased

    display(correct_samples, biased_samples, n)


test_methods(100000)
