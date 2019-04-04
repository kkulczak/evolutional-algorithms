import numpy as np

PARAMETERS_TO_SAVE = (np.min, np.max, np.mean)


def parent_selection(P, scores, Lambda):
    props = 1 / scores
    children_ids = np.random.choice(
        range(scores.size),
        size=Lambda,
        replace=True,
        p=(props / props.sum()).squeeze()
    )
    return P[children_ids]


def mutation(P, Tau, Tau0):
    # copy_P = P.copy()
    ind_size = (P.shape[-1] // 2)
    xs = P[:, :ind_size]
    sigmas = P[:, ind_size:]

    epsilons_i = np.random.randn(*sigmas.shape) * Tau
    epsilons_0 = np.random.randn(sigmas.shape[0], 1) * Tau0
    sigmas *= np.exp(epsilons_0 + epsilons_i)

    xs += np.random.randn(*xs.shape) * sigmas


def replacement(From, Mu):
    ids = np.random.choice(
        np.arange(From.shape[0]),
        size=Mu,
        replace=False
    )
    return From[ids]


def save_logs(scores, logs):
    for x in PARAMETERS_TO_SAVE:
        logs[x.__name__].append(x(scores))


def ES_mu_lambda(
    population_evaluation,
    individual_size,
    iterations=50,
    plus=True,
    Mu=500,
    Lambda=300,
    Tau=1,
    Tau0=1,
    domain=(0, 1),
    verbose=False,
    **kwargs


):
    logs = {x.__name__: [] for x in PARAMETERS_TO_SAVE}

    P = np.hstack((
        np.random.uniform(
            low=domain[0],
            high=[1],
            size=(Mu, individual_size)
        ),
        np.random.rand(*(Mu, individual_size))
    ))

    scores = population_evaluation(P[:individual_size])

    for i in range(iterations):
        if verbose:
            print('.', end='')
        save_logs(scores, logs)
        children_P = parent_selection(P, scores, Lambda)
        mutation(children_P, Tau, Tau0)
        if plus:
            P = replacement(np.vstack((P, children_P)))
        else:
            P = children_P
        scores = population_evaluation(P[:individual_size])
    save_logs(scores, logs)
    return scores.min(), P[scores.argmin()], logs


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
