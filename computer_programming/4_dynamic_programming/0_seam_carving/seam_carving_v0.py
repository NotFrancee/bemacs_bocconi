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
    if (not isinstance(img, np.ndarray)) or (img.ndim != 2):
        raise Exception("img must be a 2-dim np array!")
    n, m = img.shape
    if m < 2:
        raise Exception("input width must be at least 2")

    g = np.zeros((n, m))

    # for i in range(n):
    #     g[i, 0] = np.abs(img[i,0] - img[i,1])
    #     for j in range(1,m-1):
    #         gl = np.abs(img[i, j] - img[i, j-1])
    #         gr = np.abs(img[i, j] - img[i, j+1])
    #         g[i, j] = (gl + gr)/2
    #     g[i, m-1] = np.abs(img[i,m-1] - img[i,m-2])

    for i in range(n):
        for j in range(m):
            gl = np.abs(img[i, j] - img[i, j - 1]) if j > 0 else 0.0
            gr = np.abs(img[i, j] - img[i, j + 1]) if j < m - 1 else 0.0
            f = 2 if 0 < j < m - 1 else 1.0
            g[i, j] = (gl + gr) / f

    # TASK 1
    # Can we avoid for loops? Vectorize it! (still a few lines required)

    return g


def test_gradx():
    g = gradx(img_test)
    assert np.array_equal(g, g_test)
    print("OK gradx!")


# test_gradx()


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

    return carved_img


def test_carve():
    carved_img = carve(img_test, seam_test)
    print(carved_img)
    assert np.array_equal(carved_img, img_carved_test)
    print("OK carve!")


# test_carve()


def get_seam(grad):
    # THIS FUNCTION should return the optimal Seam:
    # i.e. an array on length equal to the number of rows in the image
    # where each element specifies the pixel to be removed
    return 1


def seam_carve(img):
    g = gradx(img)
    seam = get_seam(g)
    img_reduced = carve(img, seam)
    return img_reduced


def test_get_seam():
    seam = get_seam(g_test)
    assert np.array_equal(seam, seam_test)
    print("find_seam OK")


# test_get_seam()


def test_seam_carve():
    img_carved = seam_carve(img_test)
    assert np.array_equal(img_carved, img_carved_test)
    print("seam carve OK")


# test_seam_carve()
