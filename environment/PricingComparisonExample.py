import numpy as np
import matplotlib.pyplot as plt
from environment.PricingEnvironment import *
from environment.PricingTS_Learner import *
from environment.PricingGreedyLearner import *
from environment.ConversionRate import *


# Possible prices
price_array = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
# Coversion rates corresponding to prices, give by appropriate function
rate_man_eu = curve_man_eu()(price_array)
rate_man_usa = curve_man_usa()(price_array)
rate_woman = curve_woman()(price_array)
# Percentage of user belonging to each class, depending on budget allocation, which is fixed
perc = [0.5, 0.3, 0.2]
# Combine curves
p = (perc[0] * rate_man_eu + perc[1] * rate_man_usa + perc[2] * rate_woman) / 100
# p = np.array([0.15, 0.1, 0.1, 0.35])  DIA video data
n_arms = len(p)
opt = np.max(p)

T = 1000

n_experiments = 1000
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []


for e in range(0, n_experiments):
    env = PricingEnvironment(n_arms=n_arms, probabilities=p)
    ts_learner = PricingTSLearner(n_arms=n_arms)
    gr_learner = PricingGreedyLearner(n_arms=n_arms)
    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)


plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()
