import matplotlib.pyplot as plt
import numpy as np

"""
This file defines the possible functions that map a budget value 
to the corresponding expected number of clicks
"""


def func(x, bound, slope):
    """
    :param x: point to estimate
    :param bound: number of clicks with max budget
    :param slope: slope of the curve
    :return: f(x)
    """
    return (1 - np.exp(-slope * x)) * bound


def func_man_eu(x):
    return func(x, 1000, 5)


def func_man_usa(x):
    return func(x, 500, 3)


def func_woman(x):
    return func(x, 200, 2)


budget = np.linspace(0, 1.0, 20)
plt.figure()
plt.plot(budget, func_man_eu(budget), 'r:', label='Man EU')
plt.plot(budget, func_man_usa(budget), 'b:', label='Man USA')
plt.plot(budget, func_woman(budget), 'g:', label='Woman')
plt.title("Clicks / Budget curve")
plt.xlabel("Normalized Budget")
plt.ylabel("Number of clicks")
plt.legend()
plt.show()



"""
Samples test


def clicks_distr(x, bound, slope, noise_std):
    return func(x, bound, slope) + np.random.normal(0, noise_std, size=func(x, bound, slope).shape)


n_obs = 20
budget = np.linspace(0, 1.0, 20)
x_obs = np.array([])
y_obs = np.array([])
noise_std = 10
bound = 1000  # Number of clicks with max budget
slope = 5.0

for i in range (1, n_obs):
    new_x_obs = np.random.choice(budget, 1)
    new_y_obs = clicks_distr(new_x_obs, bound, slope, noise_std)
    x_obs = np.append(x_obs, new_x_obs)
    y_obs = np.append(y_obs, new_y_obs)
    x_pred = np.atleast_2d(budget).T
    plt.figure(i)
    plt.plot(x_pred, func(x_pred, bound, slope), 'r:', label='func')
    plt.plot(x_obs, y_obs, 'ro', label='obs')
    plt.title("Clicks - budget curve")
    plt.xlabel("Normalized Budget")
    plt.ylabel("Number of clicks")
    plt.show()
"""
