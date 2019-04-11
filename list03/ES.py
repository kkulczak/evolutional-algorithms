import numpy as np

PARAMETERS_TO_SAVE = (np.min, np.max, np.mean)


def parent_selection(P, scores, Lambda):
    fitness_values = scores.max() - scores
    if fitness_values.sum() > 0:
        fitness_values = fitness_values / fitness_values.sum()
    else:
        fitness_values = np.ones(scores.size) / scores.size
    children_ids = np.random.choice(
        range(scores.size),
        size=Lambda,
        replace=True,
        p=fitness_values
    )
    return P[children_ids]


def mutation(P, Tau, Tau0):
    # copy_P = P.copy()
    ind_size = (P.shape[-1] // 2)
    xs = P[:, :ind_size]
    sigmas = P[:, ind_size:]

    epsilons_i = np.random.randn(*sigmas.shape) * (Tau)
    epsilons_0 = np.random.randn(sigmas.shape[0], 1) * (Tau0)
    sigmas *= np.exp(epsilons_0 + epsilons_i)

    xs += np.random.randn(*xs.shape) * (sigmas)


def replacement(pop, scores, Mu):
    chosen = np.argsort(scores)[:Mu]
    return pop[chosen], scores[chosen]


def save_logs(scores, logs):
    for x in PARAMETERS_TO_SAVE:
        logs[x.__name__].append(x(scores))


def ES_mu_lambda(
    population_evaluation,
    individual_size,
    iterations=50,
    plus=True,
    Mu=400,
    Lambda=800,
    K=0.5,
    domain=(0, 1),
    verbose=False,
    seed=None,
    **kwargs


):
    np.random.seed(seed)
    Tau = K / np.sqrt(2 * individual_size)
    Tau0 = K / np.sqrt(2 * np.sqrt(individual_size))
    logs = {x.__name__: [] for x in PARAMETERS_TO_SAVE}

    P = np.hstack((
        np.random.uniform(
            low=domain[0],
            high=domain[1],
            size=(Mu, individual_size)
        ),
        np.random.uniform(
            low=1,
            high=domain[1] / 4,
            size=(Mu, individual_size)
        )
    ))

    P_scores = population_evaluation(P[:, :individual_size])

    for i in range(iterations):
        if verbose:
            print('.', end='')
        save_logs(P_scores, logs)
        children_P = parent_selection(P, P_scores, Lambda)
        mutation(children_P, Tau, Tau0)
        
        children_scores = population_evaluation(children_P[:, :individual_size])
        
        if plus:
            P, P_scores = replacement(
                np.vstack((P, children_P)),
                np.hstack((P_scores, children_scores)),
                Mu
            )
        else:
            P, P_scores = replacement(
                children_P,
                children_scores,
                Mu
            )

    save_logs(P_scores, logs)
    return P_scores.min(), P[P_scores.argmin()], logs


def test():
    res = ES_mu_lambda(
        lambda x: (x ** 2).sum(axis=1),
        10
    )
    print(res)


if __name__ == "__main__":
    pop = np.array([
        [1, 2, 3, 0, 1, 0],
        [5, 6, 6, 1, 0, 0],
        [4, 4, 4, 0, 0, 1]
    ], dtype=np.float64)
    test()
