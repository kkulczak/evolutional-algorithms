import numpy as np


def griewank(X):
    return (
        1
        + (X ** 2).sum(axis=1) / 4000
        - np.prod(
            np.cos(
                X / np.sqrt(np.arange(1, X.shape[-1] + 1))
            ),
            axis=1
        )
    )


def rastrigin(v):  # min @ 0*
    return 10*v.size + (v**2 - 10*np.cos(2*np.pi*v)).sum(1)


def schwefel(v):  # min @ 420.9687*
    return 418.9829*v.size - (v*np.sin(np.sqrt(np.abs(v)))).sum(1)


def levy(v):  # min @ 1*
    w = 1 + (v - 1)/4
    return np.sin(np.pi*w[0])**2 + (w[v.size - 1] - 1)**2 * (1 + (np.sin(2*np.pi*w[v.size - 1]))**2) + ((w[1:-1] - 1)**2 * (1 + 10*(np.sin(np.pi * w[1:-1] + 1)**2))).sum(1)


if __name__ == "__main__":
    # pop = np.array([
    #     [1, 2, 3, 0, 1, 0],
    #     [5, 6, 6, 1, 0, 0],
    #     [4, 4, 4, 0, 0, 1]
    # ], dtype=np.float64)
    # x = pop - np.arange(1, pop.shape[-1] + 1)  # [np.newaxis, :]
    # print(x)
    a = np.array([1, -2, -4, 6])
    print(a / a.sum())
    pop = np.vstack((
        np.zeros((1, 15)),
        np.ones((1, 15)) * 600,
        np.arange(15)[np.newaxis, :]
    ))
    print(Griewank().function(pop))
