import numpy as np

def get_group_indices(individual_size, a=None, b=None, x=None, **kwargs):
    if a is None and b is None:
        a, b = np.sort(
            np.random.choice(
                range(individual_size),
                size=2,
                replace=False)
        )

    x = np.random.choice(range(3), size=1, replace=False)[0] if x is None else x 
    if x == 0:
        b = a
        a = 0
    elif x == 2:
        a = b
        b = individual_size
    return a,b


def PMX(ind1, ind2, **kwargs):
    a, b = get_group_indices(len(ind1), **kwargs)
    
    O1 = ind1.copy()
    O2 = ind2.copy()
    O1[a:b] = ind2[a:b]
    O2[a:b] = ind1[a:b]

    ind1_mapping = dict(zip(ind1[a:b], ind2[a:b]))
    ind2_mamping = dict(zip(ind2[a:b], ind1[a:b]))

    indexes_of_rewritten_groups = np.concatenate((
        np.arange(0, a),
        np.arange(b, len(ind1))
    ))

    for i in indexes_of_rewritten_groups:
        while O1[i] in ind2_mamping:
            O1[i] = ind2_mamping[O1[i]]
        while O2[i] in ind1_mapping:
            O2[i] = ind1_mapping[O2[i]]

    return O1, O2




def test_case_slides():
    P1 = np.array([1, 2, 3, 10, 4, 5, 6, 7, 11, 8, 9])
    P2 = np.array([4, 5, 2, 10, 1, 8, 7, 6, 11, 9, 3])
    target_O1 = np.array([4, 2, 3, 10, 1, 8, 7, 6, 11, 5, 9])
    target_O2 = np.array([1, 8, 2, 10, 4, 5, 6, 7, 11, 9, 3])
    O1, O2 = PMX(P1, P2, a=3, b=9, x=1)
    if not all(target_O1 == O1) or not all(target_O2 == O2):
        print(target_O1)
        print(O1, 'o1')
        print(target_O2)
        print(O2, 'o2')
        assert False

if __name__ == "__main__":
    test_case_slides()