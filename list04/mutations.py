import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import *


def plot_mutation(
        mutations,
        domain_X=np.arange(-5, 5, 0.25),
        domain_Y=np.arange(-5, 5, 0.25),
        objective_function=objective_function_F1a,
        original_individual=np.array([[1, 1]]),
):
    X, Y = np.meshgrid(domain_X, domain_Y)
    Z = - objective_function(np.vstack([X.ravel(),
                                        Y.ravel()]).T).reshape(X.shape[0], X.shape[1])
    plt.figure(figsize=(6, 6))
    plt.contour(X, Y, Z, 50)
    plt.plot(mutations[:, 0], mutations[:, 1], 'ro')
    plt.plot(original_individual[0, 0],
             original_individual[0, 1], 'k*', markersize=24)
    # plt.title('Mutation 1')
    plt.show()


def mutate(
    sigma,
    original_individual=np.array([[1, 1]]),
    N=250,
    d=2,
):
    if (
            isinstance(sigma, int)
            or isinstance(sigma, float)
            or len(sigma.shape) == 1
    ):
        return original_individual + sigma * np.random.randn(N, d)

    return (
        original_individual
        + np.dot(
            np.random.randn(N, d),
            np.linalg.cholesky(sigma).T
        )
    )


def benchmark_mutation(
    sigma,
    objective_function=objective_function_F1a,
    original_individual=np.array([[1, 1]])
):
    indv_res = objective_function(original_individual)[0]

    x = np.array([objective_function(mutate(sigma)) for i in range(100)])
    better_than_indv = (x < indv_res).sum(axis=1)
    best_from_pop = x.max(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    ax1.hist(better_than_indv / 250)
    ax1.set_title('Amount of better children')
    ax2.hist(best_from_pop)
    ax2.set_title('Bests From Children')
    fig.suptitle(
        objective_function.__name__,
        verticalalignment='top',
        fontsize=15,
        y=1.05

    )
    fig.tight_layout()
    plt.show()
