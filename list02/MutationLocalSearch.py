import numpy as np
import itertools


def mutation_local_search(P, f_target, K=3, verbose=False, **kwargs):
    print('PING')
    def perms(comb):
        where_changing = np.array(comb)
        it = (
            (
                where_changing,
                where_changing[list(x)]
            ) for x in
            itertools.permutations(np.arange(K))
        )
        next(it)
        return it

    combinations = itertools.combinations(
        np.arange(len(P)),
        K
    )

    population = (
        x for c in combinations
        for x in perms(c)
    )

    working_p = P.copy()
    cur_best_p = P
    cur_best_val = f_target(P)

    for target, value in population:
        working_p[target] = P[value]
        score = f_target(working_p)
        if score < cur_best_val:
            cur_best_val = score
            cur_best_p = working_p.copy()
        working_p[target] = P[target]

    if verbose:   
        return cur_best_p, cur_best_val

    return cur_best_p

def test():
    def f(x):
        return (x[-3] + x[-2] ** 2 + (x[-1] + 1) ** 3) 
    P = np.arange(10)
    temp = mutation_local_search(P, f, K=4)
    if temp[-1] != 9:
        print("Wrong opt should be 9")
        assert False

if __name__ == "__main__":
    test()
