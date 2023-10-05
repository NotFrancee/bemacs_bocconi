"""
Create a 2-d array of 10x20 random Gaussian numbers using numpy (like for exercise 19, just this time it's 2-d).
Compute their mean (1 line of code).

Now take a sub-table of the first table in which only the elements whose row and column indices are both even are
kept: e.g. elements like  [2,4] ,  [0,8]  are ok, while  [1,2]  or  [3,5]  are not. Do this with a single indexing
expression. Hint: use two slices, one for the rows and one for the columns, and see also exercise 6.

Question: do you get a new independent array or a view? Check your answer by modifying the second array and
verifying whether the first array is also changed, or not. How could you obtain the opposite result, if you wanted to?
Hint: there is an array method that you should use after the slicing.
Compute the mean of this sub-array.

Now put everything in a function that does all of the above operations repeatedly for a number of times (say, 1000
times); i.e. produce a random array and compute its mean and the mean of the “even-indexed sub-table”. The
function should collect the results in two separate 1-d arrays.

Following this, the function should plot a histogram of both these arrays of means. Remember that if you keep
calling  plot  or  hist  the new plots will be superimposed on the previous ones, therefore you will need a
command to clear the figure between tests. You should clearly see a roughly Gaussian shape of the resulting
curves. Look at the available options for  hist here, and try to have the two histograms plotted as lines on top of
each other, i.e. not to have the histogram bars filled. Also try to increase the resolution of the steps by increasing
the number of bins: what happens? How do you obtain smoother curves? Experiment!

Extra: If you know some statistics, you should be able to predict the shape of the two curves, and verify your
prediction by plotting the analytical curves on top of the histograms. In this case, you may want to look at the
option  density  of the  hist  function
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def collect(n: int = 1000):
    means = np.zeros(n)
    means_even = np.zeros(n)

    for i in range(n):
        arr = np.random.randn(10, 20)

        evens = arr[::2, ::2]  # we get a mask

        means[i] = np.mean(arr)
        means_even[i] = np.mean(evens)
    return means, means_even


def display(means: list[tuple[np.array]], bins: int = 100):
    fig, ax = plt.subplots(len(means), 1)

    for i, mean in enumerate(means):
        ax[i].hist(mean[0], bins, density=True, histtype="step")
        ax[i].hist(mean[1], bins, density=True, histtype="step")

    plt.show()


def run(n):
    means = [collect(i) for i in n]

    # experiment bins
    for i in [10, 50, 100]:
        display(means, i)


run([100, 1000, 10000])
