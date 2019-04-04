import numpy as np
import matplotlib.pyplot as plt


def plot_hist(data):
    for option_name, data in data.items():
        plt.hist([x[0] for x in data], label=option_name)
    plt.legend()
    plt.show()


def plot_one(data):
    mins = data[-1]['amin']
    best_id = np.argmin(mins)

    plt.figure(figsize=(15, 5))
    for name, vals in data[-1].items():
        plt.plot(range(len(vals)), vals, label=name)
    plt.text( best_id, mins[best_id], mins[best_id], weight='bold', fontsize=14 )
    plt.scatter(best_id, mins[best_id], c='r')
    plt.legend()
    plt.show()
