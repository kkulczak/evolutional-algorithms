import numpy as np
import matplotlib.pyplot as plt
import time
from MutationLocalSearch import mutation_local_search


def SGA(
    population_size,
    chromosome_length,
    number_of_offspring,
    crossover_probability,
    mutation_probability,
    number_of_iterations,
    tsp_objective_function,
    operator,
    mutation,
    local_search_mut_propability=0,
    local_search_mut_K=2,
    verbose=False,
    *args,
    **kwargs
):


    time0 = time.time()
    costs = np.zeros((number_of_iterations, population_size))
    best_objective_value = np.Inf
    best_chromosome = np.zeros((1, chromosome_length))

    # generating an initial population
    current_population = np.zeros(
        (population_size, chromosome_length), dtype=np.int64)
    for i in range(population_size):
        current_population[i, :] = np.random.permutation(chromosome_length)

    # evaluating the objective function on the current population
    objective_values = np.zeros(population_size)
    for i in range(population_size):
        objective_values[i] = tsp_objective_function(current_population[i, :])

    for t in range(number_of_iterations):

        # selecting the parent indices by the roulette wheel method
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(population_size) / population_size
        parent_indices = np.random.choice(
            population_size, number_of_offspring, True, fitness_values).astype(np.int64)

        # creating the children population
        children_population = np.zeros(
            (number_of_offspring, chromosome_length), dtype=np.int64)
        for i in range(int(number_of_offspring/2)):
            if np.random.random() < crossover_probability:
                children_population[2*i, :], children_population[2*i+1, :] = operator(
                    current_population[parent_indices[2*i], :].copy(), current_population[parent_indices[2*i+1], :].copy())
            else:
                children_population[2*i, :], children_population[2*i+1, :] = current_population[parent_indices[2*i],
                                                                                                :].copy(), current_population[parent_indices[2*i+1]].copy()
        if np.mod(number_of_offspring, 2) == 1:
            children_population[-1, :] = \
                current_population[parent_indices[-1], :]

        # mutating the children population
        for i in range(number_of_offspring):
            if np.random.random() < mutation_probability:
                children_population[i, :] = mutation(
                    children_population[i, :])
        
        # local search_mutation
        for i in range(number_of_offspring):
            if np.random.random() < local_search_mut_propability:
                children_population[i, :] = mutation_local_search(
                    children_population[i, :],
                    tsp_objective_function,
                    K=local_search_mut_K
                )


        # evaluating the objective function on the children population
        children_objective_values = np.zeros(number_of_offspring)
        for i in range(number_of_offspring):
            children_objective_values[i] = tsp_objective_function(
                children_population[i, :])

        # replacing the current population by (Mu + Lambda) Replacement
        objective_values = np.hstack(
            [objective_values, children_objective_values])
        current_population = np.vstack(
            [current_population, children_population])

        I = np.argsort(objective_values)
        current_population = current_population[I[:population_size], :]
        objective_values = objective_values[I[:population_size]]

        # recording some statistics
        if best_objective_value < objective_values[0]:
            best_objective_value = objective_values[0]
            best_chromosome = current_population[0, :]

        if verbose:
            print('%3d %14.8f %12.8f %12.8f %12.8f %12.8f' % (t, time.time() - time0, objective_values.min(),
                                                              objective_values.mean(), objective_values.max(), objective_values.std()))
        costs[t, :] = objective_values

    return np.min(costs), costs
