import numpy as np
import math


class Sudoku(object):
    def __init__(self, data):
        self.data = data

        self.cols = None
        self.rows = None

    def check_size(self):
        data = self.data

        rows = len(data)
        cols = np.array([len(i) for i in data])

        unique_cols = np.unique(cols)

        if not len(unique_cols) == 1 or unique_cols[0] == rows:
            print("meeeh")
            return False

        self.rows = rows
        self.cols = unique_cols[0]

    def check_rows(self):
        unique_els = np.unique(self.data, axis=1)
        print(unique_els)

    def is_valid(self):
        data = self.data

        if not self.check_size():
            return False

        if not math.sqrt(self.rows).is_integer():
            return False

        self.check_rows()


# s = Sudoku([[1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4], [1]])
s = Sudoku([[1, 4, 2, 3], [3, 2, 4, 1], [4, 1, 3, 2], [2, 3, 1, 4]])
s.is_valid()
