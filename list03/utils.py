import numpy as np
import matplotlib.pyplot as plt


def plot_hist(data):
    for option_name, data in data.items():
        plt.hist([x[0] for x in data], label=option_name)
    plt.show()


def plot_one(data):
    for name, vals in data[-1]:
        plt.plot(vals, label=name)
    plt.show()
