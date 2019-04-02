import numpy as np
from PMX import get_group_indices


def OX(ind1, ind2, **kwargs):
    a, b = get_group_indices(len(ind1), **kwargs)

    O1 = ind1.copy()
    O2 = ind2.copy()

    copied_from_ind1 = frozenset(ind1[a:b])
    copied_from_ind2 = frozenset(ind2[a:b])

    ordered_indexes_sequence = np.concatenate((
        np.arange(b, len(ind1)),
        np.arange(0, b)
    ))
    
    j = b % len(ind1)
    for i in ordered_indexes_sequence:
        if ind2[i] not in copied_from_ind1:
            O1[j] = ind2[i]
            j = (j + 1) % len(ind1)
    assert j == a

    j = b % len(ind1)
    for i in ordered_indexes_sequence:
        if ind1[i] not in copied_from_ind2:
            O2[j] = ind1[i]
            j = (j + 1) % len(ind1)
    assert j == a
    
    return O1, O2


def test_case(x=1):
    P1 = np.array([1, 2, 3, 10, 4, 5, 6, 7, 11, 8, 9])
    P2 = np.array([4, 5, 2, 10, 1, 8, 7, 6, 11, 9, 3])
    target_O1 = np.array([2, 1, 8, 10, 4, 5, 6, 7, 11, 9, 3])
    target_O2 = np.array([3, 4, 5, 10, 1, 8, 7, 6, 11, 9, 2])
    O1, O2 = OX(P1, P2, a=3, b=9, x=x)
    if not all(target_O1 == O1) or not all(target_O2 == O2):
        print(target_O1, ' target_O1')
        print(O1, ' O1')
        print(target_O2, ' target_02')
        print(O2, ' O2')
        assert False


def test_case_2():
    P1 = [
    18, 28, 24, 25, 2, 30, 35, 7, 12, 3, 1, 49, 10, 36, 50, 45, 14, 4, 13, 19, 29, 11, 34, 40,
    9, 27, 8, 33, 26, 22, 47, 41, 44, 42, 21, 20, 32, 15, 23, 46, 5, 43, 39, 17, 31, 0, 51, 38,
    37, 48, 16, 6]
    P2 = [33, 47, 12, 48, 51, 25, 28, 34, 50, 10, 39, 16, 45, 19, 0, 15, 44, 31, 41, 40, 20, 27, 14, 5, 9, 43, 1, 26, 37, 8, 30, 4, 22, 32, 17, 6, 23, 42, 21, 18, 24, 7, 29, 36, 49, 2, 38, 11, 13, 46, 35, 3]
    OX(P1, P2, x=2)


if __name__ == "__main__":
    test_case()
    # test_case_2()