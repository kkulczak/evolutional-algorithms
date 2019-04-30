import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import *


def es(
    objective_function,
    chromosome_length,
    population_size,
    number_of_iterations,
    number_of_offspring,
    number_of_parents,
    sigma,
    tau,
    tau_0,
    log_frequency=1,
    verbose=True
):

    best_solution = np.empty((1, chromosome_length))
    best_solution_objective_value = 0.00

    log_objective_values = np.empty((number_of_iterations, 4))
    log_best_solutions = np.empty((number_of_iterations, chromosome_length))
    log_best_sigmas = np.empty((number_of_iterations, chromosome_length))

    # generating an initial population
    current_population_solutions = 100.0 * \
        np.random.rand(population_size, chromosome_length)
    current_population_sigmas = sigma * \
        np.ones((population_size, chromosome_length))

    # evaluating the objective function on the current population
    current_population_objective_values = objective_function(
        current_population_solutions)

    for t in range(number_of_iterations):

        # selecting the parent indices by the roulette wheel method
        fitness_values = current_population_objective_values - \
            current_population_objective_values.min()
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = 1.0 / population_size * np.ones(population_size)
        parent_indices = np.random.choice(
            population_size,
            (number_of_offspring, number_of_parents),
            True,
            fitness_values
        ).astype(np.int64)

        # creating the children population by Global Intermediere Recombination
        children_population_solutions = np.zeros(
            (number_of_offspring, chromosome_length)
        )
        children_population_sigmas = np.zeros(
            (number_of_offspring, chromosome_length)
        )
        for i in range(number_of_offspring):
            children_population_solutions[i, :] = current_population_solutions[parent_indices[i, :], :].mean(
                axis=0)
            children_population_sigmas[i, :] = current_population_sigmas[parent_indices[i, :], :].mean(
                axis=0)

        # mutating the children population by adding random gaussian noise
        children_population_sigmas = (
            children_population_sigmas
            * np.exp(
                tau * np.random.randn(number_of_offspring, chromosome_length)
                + tau_0 * np.random.randn(number_of_offspring, 1)
            )
        )
        children_population_solutions = children_population_solutions + \
            children_population_sigmas * \
            np.random.randn(number_of_offspring, chromosome_length)

        # evaluating the objective function on the children population
        children_population_objective_values = objective_function(
            children_population_solutions
        )

        # replacing the current population by (Mu + Lambda) Replacement
        current_population_objective_values = np.hstack(
            [
                current_population_objective_values,
                children_population_objective_values
            ]
        )
        current_population_solutions = np.vstack(
            [current_population_solutions, children_population_solutions])
        current_population_sigmas = np.vstack(
            [current_population_sigmas, children_population_sigmas])

        I = np.argsort(current_population_objective_values)[::-1]
        current_population_solutions = current_population_solutions[I[:population_size], :]
        current_population_sigmas = current_population_sigmas[I[:population_size], :]
        current_population_objective_values = current_population_objective_values[
            I[:population_size]]

        # recording some statistics
        if best_solution_objective_value < current_population_objective_values[0]:
            best_solution = current_population_solutions[0, :]
            best_solution_objective_value = current_population_objective_values[0]
        log_objective_values[t, :] = [
            current_population_objective_values.min(),
            current_population_objective_values.max(),
            current_population_objective_values.mean(),
            current_population_objective_values.std()
        ]
        log_best_solutions[t, :] = current_population_solutions[0, :]
        log_best_sigmas[t, :] = current_population_sigmas[0, :]

        if verbose and np.mod(t + 1, log_frequency) == 0:
            print("Iteration %04d : best score = %0.8f, mean score = %0.8f." % (
                t + 1, log_objective_values[:t+1, 1].max(), log_objective_values[t, 2]))

    return (
        best_solution_objective_value,
        best_solution,
        log_objective_values,
        log_best_solutions,
        log_best_sigmas
    )


def plot_es(res):
    (
        best_objective_value,
        best_chromosome,
        history_objective_values,
        history_best_chromosome,
        history_best_sigmas
    ) = res
    plt.figure(figsize=(18, 4))
    plt.plot(history_objective_values[:, 0], label='min')
    plt.plot(history_objective_values[:, 1], label='max')
    plt.plot(history_objective_values[:, 2], label='mean')
    plt.xlabel('iteration')
    plt.ylabel('objective function value')
    plt.title('min/avg/max objective function values')
    plt.legend()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(history_best_sigmas)
    plt.xlabel('iteration')
    plt.ylabel('sigma value')
    plt.title('best sigmas')
    plt.show()


def run_es(
    d,
    N,
    T,
    func,
    log_frequency=10,
    number_of_parents=2,
    plot=True,
    sigma=50.0,
    verbose=True,
    K=1

):
    result = es(
        objective_function=func,
        chromosome_length=d,
        population_size=N,
        number_of_iterations=T,
        number_of_offspring=2*N,
        number_of_parents=number_of_parents,
        sigma=sigma,
        tau=K/np.sqrt(2*d),
        tau_0=K/np.sqrt(2*np.sqrt(d)),
        log_frequency=log_frequency,
        verbose=verbose
    )
    (
        best_objective_value,
        best_chromosome,
        history_objective_values,
        history_best_chromosome,
        history_best_sigmas
    ) = result
    if plot:
        plot_es(result)
    return result


def plot_3D_benchmark_function(objective_function, domain_X, domain_Y, title):
    plt.figure(figsize=(12, 8))
    ax = plt.gca(projection='3d')
    X, Y = np.meshgrid(domain_X, domain_Y)
    Z = - objective_function(np.vstack([X.ravel(),
                                        Y.ravel()]).T).reshape(X.shape[0], X.shape[1])
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.hot, linewidth=0, antialiased=True)
    plt.title(title)
    plt.show()


def plot_contour_benchmark_function(objective_function, domain_X, domain_Y, title):
    plt.figure(figsize=(9, 9))
    X, Y = np.meshgrid(domain_X, domain_Y)
    Z = - objective_function(np.vstack([X.ravel(),
                                        Y.ravel()]).T).reshape(X.shape[0], X.shape[1])
    plt.contour(X, Y, Z, 50)
    plt.title(title)
    plt.show()
