import numpy as np
import matplotlib.pyplot as plt
from ES import ES_mu_lambda
from multiprocessing import Pool


def plot_hist(data):
    for option_name, data in data.items():
        plt.hist([x[0] for x in data], label=option_name)
    plt.legend()
    plt.show()


def gen_title(x):
    return f'{x["population_evaluation"].__name__}_{x["individual_size"]}'


def _lambda_ES(x):
    return ES_mu_lambda(**x)


def run_multiproc_hist(kwargs, cores=8):
    with Pool(cores) as p:
        data = p.map(_lambda_ES, [kwargs] * 8)
    plot_one(data[0], title=gen_title(kwargs))
    plot_hist({gen_title(kwargs): data})


def plot_one(data, title=None):
    mins = data[-1]['amin']
    best_id = np.argmin(mins)

    plt.figure(figsize=(15, 5))
    for name, vals in data[-1].items():
        # if name == 'amax':
        #     continue
        plt.plot(range(len(vals)), vals, label=name)
    plt.text(best_id, mins[best_id], mins[best_id], weight='bold', fontsize=14)
    plt.scatter(best_id, mins[best_id], c='r')
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()
