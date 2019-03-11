import numpy as np

def pbil(
    f,
    vec_size=10,
    prop_mut=0.1,
    pop_size=100,
    iterations=100,
    theta1=0.1,
    theta2=0.05,
    theta3=0,
    **kwargs
):
    propability_vector = np.ones(vec_size) * 0.5
    best_scores_history = []
    propabilities_vector_history = []
    for i in range(iterations):
        new_population = (np.random.rand(pop_size, vec_size) < propability_vector).astype(np.int64)

        pop_score = f(new_population)
        best = new_population[np.argmax(pop_score)]
    #         history 
        best_scores_history.append(pop_score.max())
        propabilities_vector_history.append(propability_vector)
    #     Best update
        propability_vector = np.clip(
            propability_vector * (1 - theta1) + best * theta1,
            0,
            1
        )
    # mutation
        mutated_ids = np.random.rand(vec_size) < theta2
        propability_vector[mutated_ids] = np.clip(
            (
                propability_vector[mutated_ids] * (1 - theta3) +
                (np.random.rand(mutated_ids.sum()) < 0.5) * theta3
            ),
            0,
            1
        )
    
    
    return f(best[np.newaxis, :])[0], best_scores_history,propabilities_vector_history, best, propability_vector


