def insertionsort(A: list[int]):
    n = len(A)

    for j in range(1, n):
        key = A[j]
        i = j - 1

        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            i -= 1

        A[i + 1] = key


a = [11, 1, 3, 5, 6, 2, 9, 4]
insertionsort(a)
print(a)
