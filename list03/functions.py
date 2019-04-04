import numpy as np


class Griewank():
    def function(self, X):
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
