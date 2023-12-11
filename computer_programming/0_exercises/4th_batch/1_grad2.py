def g(x):
    # print("g evaluated")
    return x**4 + 4 * x**3 + x**2 - 10 * x + 1


def grad_g(x):
    return 4 * x**3 + 12 * x**2 + 2 * x - 10


def grad2_g(x):
    return 12 * x**2 + 24 * x + 2


def grad(f, x, delta=1e-5):
    return (f(x + delta) - f(x - delta)) / (2 * delta)


def grad2_naive1(x):
    # print("-naive1-")
    return grad(grad_g, x)


def grad2_naive2(f, x, delta):
    # print("-naive2-")
    gf = lambda xx: grad(f, xx, delta)  # noqa

    return grad(gf, x, delta)


def grad2(f, x, delta):
    # print("-smart1-")
    return (-2 * f(x) + f(x + 2 * delta) + f(x - 2 * delta)) / (4 * delta**2)


# x0 = 1
# delta = 1e-1

# naive1 = grad2_naive1(x0)
# naive2 = grad2_naive2(g, x0, delta)
# v1 = grad2(g, x0, delta)

# assert naive1 - naive2 < 1e-5
# assert naive2 - v1 < 1e-5


def test():
    deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    x0 = 1

    for delta in deltas:
        real = grad2_g(x0)
        naive = grad2_naive2(g, x0, delta)
        smart = grad2(g, x0, delta)

        print(f"delta {delta}")
        print(f"naive (bps): {(naive - real) / real * 100 * 100:.5f}")
        print(f"smart (bps): {(smart - real) / real * 100 * 100:.5f}")


test()
