import numpy as np
import matplotlib.pyplot as plt

## EXERCISE 2: The wrong move

def move1(n):
    while True:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i > j:
            i, j = j, i
        if i != j:
            break
    return i, j

def move2(n):
    i = np.random.randint(n-1)
    j = np.random.randint(i+1, n)
    return i, j

def destructure(moves):
    ## this takes a list of 2-tuples and returns 2 lists,
    ## one with the first elements and one with the second elements
    mi = [x[0] for x in moves]
    mj = [x[1] for x in moves]
    return mi, mj

def destructure2(moves):
    ## same as the above, but uses some fancy syntax to do it in one line...
    ## (only for the virtuoso python programmer)
    return [list(x) for x in zip(*moves)]

def test_moves(n=10, samples=10**5):
    t1 = [move1(n) for k in range(samples)]
    t2 = [move2(n) for k in range(samples)]

    t1i, t1j = destructure(t1)
    t2i, t2j = destructure(t2)

    ## plot the i's
    plt.clf()
    plt.title("historgram of the i values")
    plt.hist(t1i, bins=n-1, histtype='step', label='move1')
    plt.hist(t2i, bins=n-1, histtype='step', label='move2')
    plt.legend(loc='upper right')
    plt.pause(2)

    ## plot the j's
    plt.clf()
    plt.title("historgram of the j values")
    plt.hist(t1j, bins=n-1, histtype='step', label='move1')
    plt.hist(t2j, bins=n-1, histtype='step', label='move2')
    plt.legend(loc='upper left')
    plt.pause(2)

    ## 2-d plots, first move1 then move2
    #  plt.clf()
    #  plt.title("move1 histogram")
    #  plt.hist2d(t1i, t1j, bins=(n-1, n-1))
    #  plt.pause(2)
    #  plt.title("move2 histogram")
    #  plt.hist2d(t2i, t2j, bins=(n-1, n-1))
    #  plt.pause(2)

    ## same as above, both plots at once
    plt.clf()
    plt.subplot(211)
    plt.title("move1 vs move 2 histogram")
    plt.hist2d(t1i, t1j, bins=(n-1, n-1))
    plt.subplot(212)
    plt.hist2d(t2i, t2j, bins=(n-1, n-1))
    plt.pause(2)
