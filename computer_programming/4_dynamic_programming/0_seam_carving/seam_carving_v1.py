import numpy as np
import numpy.random as rnd


# To import an actual image
# from PIL import Image
# import matplotlib.pyplot as plt

# my_image = Image.open('your image location')
## EXAMPLE:
## my_image = Image.open('/home/luca/Pictures/UP_Bocconi20200319103546.jpg')
# a_image = np.array(bocconi_image) # turn it into an array
# a_image_bw = a_image.mean(axis=2) # turn it to grey-scale


img_test = np.array(
    [
        [1.0, 2.0, 1.0, 1.0],
        [2.0, 0.0, 0.0, 3.0],
        [1.0, 1.0, 2.0, 0.0],
        [2.0, 2.0, 0.5, 1.0],
        [2.0, 1.0, 1.0, 0.0],
    ]
)

# the value for a cell is the avg different between the cell and the adjacent
# e.g. 1 with 2 and 1 adjacent => grad = 1/2abs(2-1) + 1/2(1-1)
# edge cases: the edges have only one adjacent cell. the formula is valid for 0<j<m-1
g_test = np.array(
    [
        [1.0, 1.0, 0.5, 0.0],
        [2.0, 1.0, 1.5, 3.0],
        [0.0, 0.5, 1.5, 2.0],
        [0.0, 0.75, 1.0, 0.5],
        [1.0, 0.5, 0.5, 1.0],
    ]
)

seam_test = np.array([2, 1, 0, 0, 1])

img_carved_test = np.array(
    [
        [1.0, 2.0, 1.0],
        [2.0, 0.0, 3.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.5, 1.0],
        [2.0, 1.0, 0.0],
    ]
)


def gradx(img):
    """Gradient function"""
    if (not isinstance(img, np.ndarray)) or (img.ndim != 2):
        raise Exception("img must be a 2-dim np array!")
    n, m = img.shape
    if m < 2:
        raise Exception("input width must be at least 2")

    g = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            gl = np.abs(img[i, j] - img[i, j - 1]) if j > 0 else 0.0
            gr = np.abs(img[i, j] - img[i, j + 1]) if j < m - 1 else 0.0
            f = 2 if 0 < j < m - 1 else 1.0
            g[i, j] = (gl + gr) / f

    # TASK 1
    # Can we avoid for loops? Vectorize it! (still a few lines required)
    # g[:, 0] = np.abs(img[:, 0], img[:, 1])
    # g[:, 1:-1] = (
    #     np.abs(img[:, 1:-1] - img[:, 0:-2]) + np.abs(img[:, 1:-1] - img[:, 2:])
    # ) / 2

    # g[:, -1] = np.abs(img[:, -1], img[:, -2])

    return g


def test_gradx():
    g = gradx(img_test)
    print(g)
    print(g_test)
    assert np.array_equal(g, g_test)
    print("OK gradx!")


test_gradx()


def carve(img, seam):
    if (not isinstance(img, np.ndarray)) or (img.ndim != 2):
        raise Exception("img must be a 2-dim np array!")
    if (not isinstance(seam, np.ndarray)) or (seam.ndim != 1):
        raise Exception("seam must be a 1-dim np array!")
    n, m = img.shape
    if len(seam) != n:
        raise Exception("the length of seam must match the number of rows of img")

    carved_img = np.zeros((n, m - 1))

    for i in range(n):
        sj = seam[i]
        # TASK 2
        # copy the image rwo by row, but in each row exclude the pixel specified in the seam
        carved_img[i, :sj] = img[i, :sj]
        carved_img[i, sj:] = img[i, sj + 1 :]

    return carved_img


def test_carve():
    carved_img = carve(img_test, seam_test)
    print(carved_img)
    assert np.array_equal(carved_img, img_carved_test)
    print("OK carve!")


test_carve()


def get_seam(grad):
    # THIS FUNCTION should return the optimal Seam:
    # i.e. an array on length equal to the number of rows in the image
    # where each element specifies the pixel to be removed
    n, m = grad.shape

    c = np.zeros((n, m))
    whence = np.zeros((n, m), dtype=int)

    # BASE CASE: copy the first row
    c[0] = grad[0]

    for i in range(1, n):  # this cannot be vectorized
        for j in range(m):
            cl = c[i - 1, j - 1] if j != 0 else np.infty
            ct = c[i - 1, j]
            cr = c[i - 1, j + 1] if j != m - 1 else np.infty

            cmin = min(cl, ct, cr)

            # in case of a tie in this case will favor coming from the left
            # if we want to favor the top we have to put the top in the first if statement
            if cmin == cl:
                whence[i, j] = -1
            elif cmin == cr:
                whence[i, j] = 1
            elif cmin == ct:
                whence[i, j] = 0

            c[i, j] = grad[i, j] + min(cl, ct, cr)

    # BACKWARD PASS
    # we reconstruct the optimal configuration
    seam = np.zeros(n, dtype=int)
    sj = np.argmin(c[-1])
    seam[-1] = sj

    for i in range(n - 2, -1, -1):
        sj = sj + whence[i + 1, sj]
        seam[i] = sj

    return seam


def seam_carve(img):
    g = gradx(img)
    seam = get_seam(g)
    img_reduced = carve(img, seam)
    return img_reduced


def test_get_seam():
    seam = get_seam(g_test)
    print(seam)
    print(seam_test)
    assert np.array_equal(seam, seam_test)
    print("find_seam OK")


test_get_seam()


def test_seam_carve():
    img_carved = seam_carve(img_test)
    assert np.array_equal(img_carved, img_carved_test)
    print("seam carve OK")


# test_seam_carve()
