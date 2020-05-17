import operator
import matplotlib.pyplot as plt
from Learners.GPTSLearner import GPTSLearner
from Learners.SWGPTSLearner import SWGPTSLearner
from environment.BudgetEnvironment import BudgetEnvironment
import numpy as np
from environment.ClickFunction import ClickFunction

if __name__ == '__main__':

    budget_sub = np.linspace(10, 80, 8)
    bdg_sub = list(budget_sub)
    func = ClickFunction(250, 0.03)
    sigma = 5
    t_horizon = 80
    theta, l_scale = 192277, 1.53
    collected_reward = []
    sw_collected_reward = []
    optimum = func.apply_func(80)

    for n in range(100):

        env = BudgetEnvironment(budget=budget_sub, sigma=sigma, func=func)
        gpl = GPTSLearner(n_arms=len(budget_sub), arms=budget_sub, noise_std=sigma, kernel_theta=theta,
                          len_scale=l_scale)
        swgpl = SWGPTSLearner(n_arms=len(budget_sub), arms=budget_sub, noise_std=sigma, kernel_theta=theta,
                              len_scale=l_scale, window_size=3*int(np.sqrt(t_horizon)))
        for t in range(0, t_horizon):
            pulled_arms = gpl.pull_arm()
            sw_pulled_arms = swgpl.pull_arm()
            # print("pulled arms")
            # print(pulled_arms)
            best_arm = int(max(pulled_arms.items(), key=operator.itemgetter(1))[0])
            sw_best_arm = int(max(sw_pulled_arms.items(), key=operator.itemgetter(1))[0])
            print(sw_best_arm)

            idx_best_arm = bdg_sub.index(best_arm)
            sw_idx_best_arm = bdg_sub.index(sw_best_arm)
            reward = env.round(best_arm)
            sw_reward = env.round(sw_best_arm)
            """print("best arm")
            print(best_arm)
            print("sw best_arm")
            print(sw_best_arm)"""
            gpl.update(idx_best_arm, reward)
            swgpl.update(sw_idx_best_arm, sw_reward)
            swgpl.plot_process(func, t)
        print(n)

        collected_reward.append(gpl._collected_rewards)
        sw_collected_reward.append(swgpl._collected_rewards)
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(optimum - collected_reward, axis=0)), "r")
    plt.plot(np.cumsum(np.mean(optimum - sw_collected_reward, axis=0)), "b")
    plt.show()



