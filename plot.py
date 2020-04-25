import matplotlib.pyplot as plt
import numpy as np


def plot_regret_advertising(opt, reward_per_experiment):
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")

    plt.plot(np.cumsum(np.mean(opt - reward_per_experiment, axis=0)), "r")
    plt.show()
