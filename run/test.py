import operator
import matplotlib.pyplot as plt
from Learners.GPTSLearner import GPTSLearner
from environment.BudgetEnvironment import BudgetEnvironment
from SystemConfiguration import SystemConfiguration
import numpy as np
from environment.ClickFunction import ClickFunction

if __name__ == '__main__':

    budget_sub = np.linspace(10, 80, 8)
    func = ClickFunction(250, 0.03)
    sigma = 0.5
    t_horizon = 60
    theta, l_scale = 1, 1

    env = BudgetEnvironment(budget=budget_sub, sigma=sigma, func=func)
    gpl = GPTSLearner(n_arms=len(budget_sub), arms=budget_sub, noise_std=sigma, kernel_theta=theta, len_scale=l_scale)

    for t in range(0, t_horizon):
        pulled_arms = gpl.pull_arm()
        #print("pulled arms")
        #print(pulled_arms)
        best_arm = int(max(pulled_arms.items(), key=operator.itemgetter(1))[0])
        budget_sub = list(budget_sub)
        idx_best_arm = budget_sub.index(best_arm)
        reward = env.round(best_arm)
        print("best arm")
        print(best_arm)
        gpl.update(idx_best_arm, reward)

        if (t+1) % 10 == 0:
            x_pred = np.atleast_2d(gpl.scaled_arms).T
            x_bdgt = np.atleast_2d(budget_sub).T
            y_pred, sigma = gpl._gp.predict(x_pred, return_std=True)
            x_obs = gpl._pulled_arms
            y_obs = gpl._collected_rewards
            plt.figure(t+1)
            plt.plot(x_pred, func.apply_func(x_bdgt), "r:", label=r'$true function$')
            plt.plot(x_pred, y_pred, 'b-', label=u'Predicted values')
            x_obs = np.atleast_2d(x_obs).T
            plt.plot(x_obs.ravel(), y_obs.ravel(), 'ro', label=u'Observed values')
            plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                     np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]), alpha=.5, fc='b',
                     ec='None',
                     label='95% conf interval')
            plt.show()


