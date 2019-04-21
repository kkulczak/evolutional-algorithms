import numpy as np


def objective_function_F1(X):
    # Sphere function (minimum at 0)
    return - np.sum(X**2, axis=1)


def objective_function_F1a(X):
    # Sphere function - modified
    return - (X[:, 0]**2 + 9*X[:, 1]**2)


def objective_function_F1b(X):
    # Sphere function - modified
    return - (X[:, 0]**2 + 625*X[:, 1]**2)


def objective_function_F1c(X):
    # Sphere function - modified
    return - (X[:, 0]**2 + 2*X[:, 1]**2 - 2 * X[:, 0] * X[:, 1])


def objective_function_F6(X):
    # Rastrigin function (minimum at 0)
    return - 10.0 * X.shape[1] - np.sum(X**2, axis=1) + 10.0 * np.sum(np.cos(2 * np.pi * X), axis=1)


def objective_function_F7(X):
    # Schwefel function (minimum at 420.9687)
    # (REMARK: should be considered only on [-500, 500]^d, because there are better minima outside)
    return - 418.9829 * X.shape[1] + np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def objective_function_F8(X):
    # Griewank function (minimum at 0)
    return - 1 - np.sum(X**2 / 4000, axis=1) + np.prod(np.cos(X / np.sqrt(np.linspace(1, X.shape[1], X.shape[1]))), axis=1)
