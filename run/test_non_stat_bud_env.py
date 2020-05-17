from environment.NonStationaryBudgetEnvironment import NonStationaryBudgetEnvironment
import numpy as np
import SystemConfiguration
from environment.ClickFunction import ClickFunction
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config = SystemConfiguration.SystemConfiguration()

    param_sub_c1 = config.init_sub_campaign("sub_campaign1")
    budget_sub_c1 = np.linspace(param_sub_c1["min_budget"], param_sub_c1["max_budget"], param_sub_c1["n_arms"])

    func_c1_p1 = ClickFunction(*config.init_function("func_man_eu_p1"))
    func_c1_p2 = ClickFunction(*config.init_function("func_man_eu_p2"))
    func_c1_p3 = ClickFunction(*config.init_function("func_man_eu_p3"))

    sigma = config.init_noise()
    t_horizon = config.init_advertising_experiment3()["t_horizon"]

    sub_c1 = NonStationaryBudgetEnvironment(budget_sub_c1, sigma, [func_c1_p1, func_c1_p2, func_c1_p3], t_horizon)
    reward = []

    for t in range(0, t_horizon):
        reward.append(sub_c1.round(80))

    plt.plot(reward, 'r')
    plt.show()