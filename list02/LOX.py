import numpy as np
from PMX import get_group_indices


def LOX(ind1, ind2, **kwargs):
    """Linear Order Crossover (LOX)
        Copy chosen group from one parent.
        Then copy all unused genes from second parent.

    Args:
        ind1 ([type]): [description]
        ind2 ([type]): [description]
    
    Returns:
        [type]: [description]
    """

    a, b = get_group_indices(len(ind1), **kwargs)
    
    O1 = ind1.copy()
    O2 = ind2.copy()

    copied_from_ind1 = frozenset(ind1[a:b])
    copied_from_ind2 = frozenset(ind2[a:b])

    ordered_indexes_sequence = np.concatenate((
        np.arange(0, a),
        np.arange(b, len(ind1))
    ))
    
    j = 0
    for i in ordered_indexes_sequence:
        while ind2[j] in copied_from_ind1:
            j = (j + 1) % len(ind1)
        O1[i] = ind2[j]
        j = (j + 1) % len(ind1)
    
    j = 0
    for i in ordered_indexes_sequence:
        while ind1[j] in copied_from_ind2:
            j = (j + 1) % len(ind1)            
        O2[i] = ind1[j]
        j = (j + 1) % len(ind1)

    return O1, O2

def test_case(x=1):
    P1 = np.array([44,12,24,42,14,32,31,43,33,11,34,21,41,13,23,22])
    P2 = np.array([41,22,24,32,14,12,31,23,13,11,44,21,34,43,33,42])
    target_O1 = np.array([41,22,24,14,12,32,31,43,33,11,34,23,13,44,21,42])
    O1, O2 = LOX(P1,P2, a=5, b=11, x=x)
    if not all(target_O1 == O1):
        print(target_O1, ' target_O1')
        print(O1, ' O1')
        assert False

if __name__ == "__main__":
    test_case()