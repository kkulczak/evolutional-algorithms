import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from notebook_utils import *
from functions import *

N = 250
d = 2

objective_function = objective_function_F1a
original_individual = np.array([[1, 1]])


def example_1():
    sigma = 0.25
    mutations = original_individual + sigma * np.random.randn(N, d)
    domain_X = np.arange(-5, 5, 0.25)
    domain_Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(domain_X, domain_Y)
    Z = - objective_function(
        np.vstack([
            X.ravel(),
            Y.ravel()
        ]).T
    ).reshape(X.shape[0], X.shape[1])

    plt.figure(figsize=(9, 9))
    plt.contour(X, Y, Z, 50)
    plt.plot(mutations[:, 0], mutations[:, 1], 'ro')
    plt.plot(original_individual[0, 0],
             original_individual[0, 1], 'k*', markersize=24)
    plt.title('Mutation 1')
    plt.show()
