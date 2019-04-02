import numpy as np
import time
import multiprocessing
import matplotlib.pyplot as plt
from SGA import SGA

def compare(func_list, tries=10):
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        res = np.array(p.map(f, enumerate([func_list] * tries)))
        print(res.shape)
    return res

def f(arg):
    i, func_list = arg
    res = []
    for f, kwargs in func_list:
            res.append(f(**kwargs)[0])
    print(i)
    return res



def reverse_sequence_mutation(p):
    a = np.random.choice(len(p), 2, False)
    i, j = a.min(), a.max()
    q = p.copy()
    q[i:j+1] = q[i:j+1][::-1]
    return q

def rand_mutation(p):
    i, j = np.random.choice(len(p), 2, False)
    q = p.copy()
    q[i] = p[j]
    q[j] = p[i]
    return q

def tsp_objective_function(p):
    s = 0.0
    for i in range(n):
        s += A[p[i-1], p[i]]
    return s

def tsp_objective_function_2(p, data, n):
    s = 0.0
    for i in range(n):
        s += data[p[i-1], p[i]]
    return s

def plot_scores(**kwargs):
    t0 = time.time()
    best, costs = SGA(**kwargs)
    print(kwargs['title'], best, time.time() - t0)
    x, y = costs.shape
    plt.figure(figsize=(15,5))
    plt.plot(range(x), costs.min(axis=1))
    plt.plot(range(x), costs.max(axis=1))
    plt.plot(range(x), costs.mean(axis=1))
    plt.show()