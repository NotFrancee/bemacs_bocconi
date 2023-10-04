def insertion_sort(A):
    n = len(A)
    # start from 1 since an array of size 1 would already be sorted
    for j in range(1, n):
        key = A[j]

        i = j - 1  # starting position to scan the list backwards

        # move backwards until you find the right place for insertio
        # or the list finishes
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            i -= 1

        A[i + 1] = key  # put the element in the correct position


A = [7, 3, 4, 2, 5, 1]
print(insertion_sort(A))
print(A)
