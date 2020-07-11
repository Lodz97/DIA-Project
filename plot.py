import matplotlib.pyplot as plt
import numpy as np
"""
    This module defines method to plot the result of the different experiments.
"""


def plot_regret_advertising(opt, reward_per_experiment):
    """
    plot the cumulative regret and the comparison between the optimal reward and the obtained one.
    """
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - reward_per_experiment, axis=0)), "b")

    plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    mean_reward = np.mean(reward_per_experiment, axis=0)
    opt = np.ones(len(mean_reward)) * opt
    plt.plot(opt, 'r', label=u'Optimal Reward')
    plt.plot(mean_reward, 'b', label=u'GPTS Reward')
    plt.legend(loc='lower right')

    plt.show()


def plot_regret_comparison(opt_per_phase, reward_per_experiment, sw_reward_per_experiment):
    """
        plot the comparison of the regret between the GPTS and SW_GPTS in a non stationary environment3
    """
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")

    opt_per_round = np.zeros(len(reward_per_experiment[0]))
    n_phases = len(opt_per_phase)
    phase_len = int(len(reward_per_experiment[0])/n_phases)

    for i in range(0, n_phases):
        opt_per_round[i*phase_len: (i+1)*phase_len] = opt_per_phase[i]

    cum_regret = np.cumsum(np.mean(opt_per_round - reward_per_experiment, axis=0))
    sw_cum_regret = np.cumsum(np.mean(opt_per_round - sw_reward_per_experiment, axis=0))

    plt.plot(cum_regret, "r", label=u'Stationary Regret')
    plt.plot(sw_cum_regret, "b", label=u'SW Regret')
    plt.legend(loc='lower right')
    plt.show()


def plot_cum_regret(opt, algorithm):
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - algorithm, axis=0)), 'r')
    plt.show()


def plot_regret_per_arm(opt, algorithm, plot_info):
    plt.figure(0)
    plt.ylabel("Regret comparison")
    plt.xlabel("Days")
    arms = np.mean(algorithm, axis=0)
    print(len(arms))
    print(len(arms[0]))
    for element in range(0, len(plot_info)):
        arm = plot_info[element]
        plt.plot(np.cumsum(opt - arms[element]), label=u'{arm}'.format(arm=arm))
        plt.legend(loc='best')
    plt.show()
