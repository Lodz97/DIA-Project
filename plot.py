import matplotlib.pyplot as plt
import numpy as np


def plot_regret_advertising(opt, reward_per_experiment):
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")

    plt.plot(np.cumsum(np.mean(opt - reward_per_experiment, axis=0)), "r")
    plt.show()


def plot_regret_comparison(opt_per_phase, reward_per_experiment, sw_reward_per_experiment):
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")

    opt_per_round = np.zeros(len(reward_per_experiment[0]))
    n_phases = len(opt_per_phase)
    for i in range(0, n_phases):
        opt_per_round[i*n_phases: (i+1)*n_phases] = opt_per_phase[i]

    cum_regret = np.cumsum(np.mean(opt_per_round - reward_per_experiment, axis=0))
    sw_cum_regret = np.cumsum(np.mean(opt_per_round - sw_reward_per_experiment, axis=0))

    plt.plot(cum_regret, "r")
    plt.plot(sw_cum_regret, "b")
    plt.show()

