import numpy as np
import matplotlib.pyplot as plt
from environment.PricingEnvironment import *
from environment.PricingTS_Learner import *
from environment.PricingGreedyLearner import *
from environment.ConversionRate import *


# Possible prices
price_array = np.array([20.0, 25.0, 30.0, 10, 15])
# Percentage of user belonging to each class, depending on budget allocation, which is fixed
perc = [0.5, 0, 0.5]

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

# Disaggregate optimum
opt_eu = np.max(profit_array * rate_man_eu)
opt_usa = np.max(profit_array * rate_man_usa)
opt_woman = np.max(profit_array * rate_woman)
opts = [opt_eu, opt_usa, opt_woman]
opt_multi = perc[0] * opt_eu + perc[1] * opt_usa + perc[2] * opt_woman

T = 3000
n_experiments = 20
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

ts_rewards_per_experiment_multi = []

for e in range(0, n_experiments):
    env = PricingEnvironment(n_arms=n_arms, probabilities=p)
    ts_learner = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    gr_learner = PricingGreedyLearner(n_arms=n_arms, profit_array=profit_array)

    # Create 3 different environments
    env_eu = PricingEnvironment(n_arms=n_arms, probabilities=rate_man_eu)
    env_usa = PricingEnvironment(n_arms=n_arms, probabilities=rate_man_usa)
    env_woman = PricingEnvironment(n_arms=n_arms, probabilities=rate_woman)
    envs = [env_eu, env_usa, env_woman]
    # Learner used as container to aggregate the 3 learners data
    ts_learner_multi = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    # Create 3 diffent TS learners
    ts_learner_eu = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_usa = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_woman = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learners = [ts_learner_eu, ts_learner_usa, ts_learner_woman]

    for t in range(0, T):
        # Thompson Sampling Learner
        # Extract randomly from a disaggregate curve
        idx = np.random.choice(range(0, 3), p=perc)
        # Aggragate learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        # Choose one of disaggregate learners
        pulled_arm = ts_learners[idx].pull_arm()
        reward = envs[idx].round(pulled_arm)
        ts_learners[idx].update(pulled_arm, reward)
        ts_learner_multi.update(pulled_arm, reward)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ts_rewards_per_experiment_multi.append(ts_learner_multi.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)


plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt_multi - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt_multi - ts_rewards_per_experiment_multi, axis=0)), 'b')
plt.plot(np.cumsum(np.mean((opt - gr_rewards_per_experiment), axis=0)), 'g')
plt.legend(["TS_Aggregate", "TS_Discrimination", "Greedy"])
plt.show()

plt.figure(1)
plt.ylabel("Expected reward")
plt.xlabel("t")
plot = np.array([])
plot2 = np.array([])
for t in range(0, T):
    plot = np.append(plot, np.cumsum(np.mean(ts_rewards_per_experiment, axis=0))[t] / (t+1))
    plot2 = np.append(plot2, np.cumsum(np.mean(ts_rewards_per_experiment_multi, axis=0))[t] / (t+1))
plt.plot(plot, 'r')
plt.plot(plot2, 'b')
plt.legend(["TS_Aggregate", "TS_Discrimination"])
plt.show()
