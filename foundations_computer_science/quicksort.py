def partition(A, p, r):  # A[p:r]
    x = A[p]  # x is the pivot
    i = p  # i stores the current position where to put items

    for j in range(p + 1, r):
        # this means the item should be on the left of ix
        if A[j] <= x:
            # i gets increased by 1
            # to set the new position where to exchange the items
            i = i + 1
            # exchange A[i] with A[j]

    # now the items have been sorted
    # the items smaller than x are on the left of i
    # the items bigger than x are on the right of i
    # i indicates where to put the pivot
    # exchange A[p] and A[i]
    return i


def quicksort(A, p, r):
    if p < r:
        # builds the subarrays
        # and puts the pivot between them
        q = partition(A, p, r)

        quicksort(A, p, q - 1)
        quicksort(A, q + 1, r)


A = [4, 1, 54, 12, 4, 13, 5]
quicksort(A, 1, len(A))

"""
1) The initial p is 0, so that it starts from the leftmost position
2) partition(A,0,n) gets called
3) pivot is set to the first item in the list
4) j starts scanning the list from (p+1)=1
5) if A[j] <= x then 
    a) i++ (i gets increased)
    b) exchange A[i] and A[j]
    c) what happens: the smaller item gets placed in the leftmost position
    d) j continues to scan the list until it reaches the end of the array
6) at the end of the loop, i is a pointer to the position in which the pivot should be placed
    so, we exchange A[p] with A[i] 
"""
