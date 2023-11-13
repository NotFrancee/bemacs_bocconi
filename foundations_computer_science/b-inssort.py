def insertionsort(A):
    n = len(A)
    for j in range(n):
        x = A[j]
        i = j

        while i > 0 and A[i - 1] > x:
            A[i] = A[i - 1]
            i -= 1

        A[i] = x

    return A


A = [5, 4, 6, 1, 9, 8]
insertionsort(A)
print(A)
