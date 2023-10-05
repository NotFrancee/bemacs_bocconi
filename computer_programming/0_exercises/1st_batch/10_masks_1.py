"""
Write a function that takes a list as its argument, let's say it is of length  n. Let's also say that the list may contain
both positive and negative numbers. As for the previous exercises, the function should create a 1-d array from this
list. 

Compare the whole array with  0 , i.e. do something like  a > 0 . What is the result? Check this.

Use the result as a mask in an indexing expression. You want to get a new array which contains only the elements
of the original array that are non-negative. For example, if your input list is  [3, -1, -2, 5, 6, -3, 8] , your
output should contain  3 ,  5 ,  6  and  8 . Can you do this with a single indexing expression? (Yes you can!)
Do you get a view or not from this? Write some code that verifies your answer
"""

import numpy as np


def mask(l: np.array):
    positive = l > 0

    return l[positive]


def test_is_view(l):
    m = mask(l)

    m[0] = -5

    print("initial list", l)
    print("final list: ", m)


test_1 = np.array([1, 2, 3, -3, -2, -1, 0])

print(mask(test_1))
print(test_is_view(test_1))
