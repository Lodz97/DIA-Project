import numpy as np
import matplotlib.pyplot as plt
from environment.PricingEnvironment import PricingEnvironment
from learners.PricingTSLearner import PricingTSLearner
from learners.PricingGreedyLearner import PricingGreedyLearner
from environment.ConversionRate import *

if __name__ == "__main__":

    # Possible prices (JSON)
    price_array = np.array([20.0, 25.0, 30.0, 10, 15])
    # Percentage of user belonging to each class, depending on budget allocation, which is fixed (JSON)
    perc = [0.4, 0.0, 0.6]

    # Profit given by appropriate function
    profit_array = interp_marginal_profit(0)(price_array)
    # Coversion rates corresponding to prices, give by appropriate function
    rate_man_eu = interp_man_eu(1)(price_array) / 100
    rate_man_usa = interp_man_usa(1)(price_array) / 100
    rate_woman = interp_woman(1)(price_array) / 100
    # Combine curves
    p = (perc[0] * rate_man_eu + perc[1] * rate_man_usa + perc[2] * rate_woman)
    show_total_profit(perc)
    # p = np.array([0.15, 0.1, 0.1, 0.35])  DIA video data
    n_arms = len(p)
    opt = np.max(profit_array * p)

    # Time interval (JSON)
    T = 1000
    # Number of experiments (JSON)
    n_experiments = 50
    ts_rewards_per_experiment = []
    gr_rewards_per_experiment = []

    for e in range(0, n_experiments):
        env = PricingEnvironment(n_arms=n_arms, probabilities=p)
        ts_learner = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
        gr_learner = PricingGreedyLearner(n_arms=n_arms, profit_array=profit_array)
        for t in range(0, T):
            # Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            # Greedy Learner
            pulled_arm = gr_learner.pull_arm()
            reward = env.round(pulled_arm)
            gr_learner.update(pulled_arm, reward)

        ts_rewards_per_experiment.append(ts_learner._collected_rewards)
        gr_rewards_per_experiment.append(gr_learner._collected_rewards)

    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
    plt.plot(np.cumsum(np.mean((opt - gr_rewards_per_experiment), axis=0)), 'g')
    plt.legend(["TS", "Greedy"])
    plt.show()
