import numpy as np


def griewank(X):
    """ minimum @ 0
    """

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


# def rastrigin(v):  # min @ 0*
#     return 10*v.size + (v**2 - 10*np.cos(2*np.pi*v)).sum(1)


# def schwefel(v):  # min @ 0*
#     return 418.9829*v.size - (v*np.sin(np.sqrt(np.abs(v)))).sum(1)

def rastrigin(P):
    n = P.shape[-1]
    return (
        P**2 - 10 * np.cos(2*np.pi * P)
    ).sum(axis=1) + 10*n


def schwefel(P):
    n = P.shape[-1]
    return np.sum(
        P * np.sin(
            np.sqrt(
                np.abs(P)
            )
        ),
        axis=1
    ) + 418.9829*n


def sphere(P):
    return np.sum(P**2, axis=1)


def dp(P):
    return (P[:, 0]-1) ** 2 + np.sum(
        (
            2 * P[:, 1:]**2 - P[:, :-1]**2
        ) ** 2 *
        np.arange(2, P.shape[-1] + 1), axis=1
    )


tests = [
    {
        'population_evaluation': griewank,
        'individual_size': 80,
        'domain': (-500, 500)
    },
    { 
        'population_evaluation': rastrigin,
        'individual_size': 60,
        'domain': (-5.12, 5.12)
    },
    { 
        'population_evaluation': schwefel,
        'individual_size': 60,
        'domain': (-500, 500)
    },
    { 
        'population_evaluation': dp,
        'individual_size': 500,
        'domain': (-10, 10)
    },
    { 
        'population_evaluation': sphere,
        'individual_size': 60,
        'domain': (-5.12, 5.12)
    }
]

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
