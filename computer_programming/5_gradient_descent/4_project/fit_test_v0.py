import numpy as np
import matplotlib.pyplot as plt  # noqa

# This script provides an example of using a non-linear minimization method
# to perform a least-squares fit


# Suppose that we have some non-linear function that depends on a real input x
# and on some parameters a, b, c
def s(x, a, b, c):
    return a + b * np.sin(c * x)


# Now for the sake of example we generate some data. The general idea of this
# test is:
#   1) we start with some known values of the parameters a, b, c (we call
#      these the "true" values)
#   2) we pick few values of x and get the corresponding function output,
#      using the true values of a, b, c
#   3) we add some "noise" (random jitter) to the output values
#   4) this gives us some data points (an array of x and a corresponding array
#      of y)
#   5) now we want to see if we can recover the original values of a, b, c by
#      just knowing the data points x,y & the general shape of the function s.
#      At least, we wish to get something close to the true values.
#   6) we use a "least squares" cost function to estimate how far we are from
#      the true values (more on this below)
#   7) we thus minimize the cost function using either our grad_desc algorithm
#      or scpiy.optimize.minimize
#   8) this procedure will give us some "optimal" values of a, b, c. We have a
#      look at how close we got to actually guessing the true values.
#
# This mimics the situation in which we have some data points and a model that
# we want to fit, and that model has some unknown parameters that we wish to
# infer from the data.
# In a realistic setting, this can be used to predict the result over inputs
# that we have never observed. Or, if the parameters have an interpretation
# (e.g. the "rate of production" of something), we can use the data to measure
# their value. Or both.
# NOTE: this method is general but determining whether it's appropriate and
# how to actually interpret the results depends on many things (how good is
# the model,
# how good is the cost function for your particular situation, etc.).
# You need a careful statistical analysis before you can make sense of the
# outcome, which is out of scope of this course.
