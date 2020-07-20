import numpy as np
import matplotlib.pyplot as plt
from environment.PricingEnvironment import *
from environment.PricingTS_Learner import *
from environment.PricingGreedyLearner import *
from environment.ConversionRate import *

# Possible prices
price_array = np.array([10, 15, 20.0, 25.0, 30.0])
# Percentage of user belonging to each class, depending on budget allocation, which is fixed
perc = [0.3, 0.5, 0.2]

#profit_array = interp_marginal_profit(0)(price_array)
profit_array = price_array
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

T = 4000
n_experiments = 50
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

ts_rewards_per_experiment_multi = []
ts_rewards_per_experiment_eu_usa = []
ts_rewards_per_experiment_eu_woman = []
ts_rewards_per_experiment_usa_woman = []
ts_rewards_per_experiment_context = []

low_bound_agg_avg = []
low_bound_disagg_avg = []
low_bound_agg_eu_usa_avg = []
low_bound_agg_eu_woman_avg = []
low_bound_agg_usa_woman_avg = []

for e in range(0, n_experiments):
    print(e)
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
    # Learners used to aggregate 2 learners each
    ts_learner_multi_eu_usa = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_multi_eu_woman = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_multi_usa_woman = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    # Create 6 different TS learners
    ts_learner_eu = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_usa = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_woman = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_eu_usa = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_eu_woman = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learner_usa_woman = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
    ts_learners = [ts_learner_eu, ts_learner_usa, ts_learner_woman, ts_learner_eu_usa, ts_learner_eu_woman,
                   ts_learner_usa_woman]
    # Learner to contain regret of best context
    ts_learner_multi_context = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)

    low_bound_agg = np.array([])
    low_bound_disagg = np.array([])
    low_bound_agg_eu_usa = np.array([])
    low_bound_agg_eu_woman = np.array([])
    low_bound_agg_usa_woman = np.array([])
    # 0->eu, 1->us, 2->woman, 3->aggregate, 4->aggregate eu usa, 5->aggregate eu woman, 6->aggregate usa woman
    exp_revs = [[], [], [], [], [], [], []]
    # 0->aggregate, 1->disaggregate, 2->eu usa, 3->eu woman, 4->usa woman
    context = 0
    context_array = np.array([])
    context_val_array = np.array([])
    context_val2_array = np.array([])
    context_val3_array = np.array([])
    context_val4_array = np.array([])
    context_val5_array = np.array([])

    for t in range(0, T):
        idx = np.random.choice(range(0, 3), p=perc)
        if context == 0:
            pulled_arm = ts_learner.pull_arm()
            reward = envs[idx].round(pulled_arm)
        if context == 1:
            pulled_arm = ts_learners[idx].pull_arm()
            reward = envs[idx].round(pulled_arm)
        # Choose partially aggregate learners
        if idx == 0:
            # eu customer, update eu usa
            if context == 2:
                pulled_arm = ts_learners[3].pull_arm()
                reward = envs[idx].round(pulled_arm)
            # eu customer, update eu woman
            if context == 3:
                pulled_arm = ts_learners[4].pull_arm()
                reward = envs[idx].round(pulled_arm)
            # eu customer, update usa woman
            if context == 4:
                pulled_arm = ts_learners[idx].pull_arm()
                reward = envs[idx].round(pulled_arm)
        if idx == 1:
            # usa customer, update eu usa
            if context == 2:
                pulled_arm = ts_learners[3].pull_arm()
                reward = envs[idx].round(pulled_arm)
            # usa customer, update eu woman
            if context == 3:
                pulled_arm = ts_learners[idx].pull_arm()
                reward = envs[idx].round(pulled_arm)
            # usa customer, update usa woman
            if context == 4:
                pulled_arm = ts_learners[5].pull_arm()
                reward = envs[idx].round(pulled_arm)
        if idx == 2:
            # woman customer, update eu usa
            if context == 2:
                pulled_arm = ts_learners[idx].pull_arm()
                reward = envs[idx].round(pulled_arm)
            # woman customer, update eu woman
            if context == 3:
                pulled_arm = ts_learners[4].pull_arm()
                reward = envs[idx].round(pulled_arm)
            # woman customer, update usa woman
            if context == 4:
                pulled_arm = ts_learners[5].pull_arm()
                reward = envs[idx].round(pulled_arm)


        # Thompson Sampling Learner
        # Extract randomly from a disaggregate curve

        # Aggragate learner

        if context == 0:
            ts_learner.update(pulled_arm, reward)
            ts_learner_multi_context.update(pulled_arm, reward)
            exp_revs[3].append(profit_array[pulled_arm] * reward)

        # Choose one of disaggregate learners

        if context == 1:
            ts_learners[idx].update(pulled_arm, reward)
            ts_learner_multi.update(pulled_arm, reward)
            ts_learner_multi_context.update(pulled_arm, reward)
            exp_revs[idx].append(profit_array[pulled_arm] * reward)

        # Choose partially aggregate learners
        if idx == 0:
            # eu customer, update eu usa

            if context == 2:
                ts_learners[3].update(pulled_arm, reward)
                ts_learner_multi_eu_usa.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[4].append(profit_array[pulled_arm] * reward)
            # eu customer, update eu woman

            if context == 3:
                ts_learners[4].update(pulled_arm, reward)
                ts_learner_multi_eu_woman.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[5].append(profit_array[pulled_arm] * reward)
            # eu customer, update usa woman

            if context == 4:
                ts_learners[idx].update(pulled_arm, reward)
                ts_learner_multi_usa_woman.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[idx].append(profit_array[pulled_arm] * reward)
        if idx == 1:
            # usa customer, update eu usa

            if context == 2:
                ts_learners[3].update(pulled_arm, reward)
                ts_learner_multi_eu_usa.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[4].append(profit_array[pulled_arm] * reward)
            # usa customer, update eu woman

            if context == 3:
                ts_learners[idx].update(pulled_arm, reward)
                ts_learner_multi_eu_woman.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[idx].append(profit_array[pulled_arm] * reward)
            # usa customer, update usa woman

            if context == 4:
                ts_learners[5].update(pulled_arm, reward)
                ts_learner_multi_usa_woman.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[6].append(profit_array[pulled_arm] * reward)
        if idx == 2:
            # woman customer, update eu usa

            if context == 2:
                ts_learners[idx].update(pulled_arm, reward)
                ts_learner_multi_eu_usa.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[idx].append(profit_array[pulled_arm] * reward)
            # woman customer, update eu woman

            if context == 3:
                ts_learners[4].update(pulled_arm, reward)
                ts_learner_multi_eu_woman.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[5].append(profit_array[pulled_arm] * reward)
            # woman customer, update usa woman

            if context == 4:
                ts_learners[5].update(pulled_arm, reward)
                ts_learner_multi_usa_woman.update(pulled_arm, reward)
                ts_learner_multi_context.update(pulled_arm, reward)
                exp_revs[6].append(profit_array[pulled_arm] * reward)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

        if (t + 1) % 7 == 0:
            # To avoid divisions by 0, wait till there is enough data
            if (len(exp_revs[0]) != 0 and len(exp_revs[1]) != 0 and len(exp_revs[2]) != 0 and len(exp_revs[3]) != 0 and
                    len(exp_revs[4]) != 0 and len(exp_revs[5]) != 0 and len(exp_revs[6]) != 0):
                low_bound_agg = np.append(low_bound_agg, np.mean(exp_revs[3]) * (1 - np.sqrt(-np.log(0.05) / (2 * (t + 1)))))
                low_bound_disagg = np.append(low_bound_disagg, perc[0] * np.mean(exp_revs[0]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[0])))) + \
                                   perc[1] * np.mean(exp_revs[1]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[1])))) + \
                                   perc[2] * np.mean(exp_revs[2]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[2])))))
                low_bound_agg_eu_usa = np.append(low_bound_agg_eu_usa, (perc[0] + perc[1]) * np.mean(exp_revs[4]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[4])))) + \
                                   perc[2] * np.mean(exp_revs[2]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[2])))))
                low_bound_agg_eu_woman = np.append(low_bound_agg_eu_woman, (perc[0] + perc[2]) * np.mean(exp_revs[5]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[5])))) + \
                                   perc[1] * np.mean(exp_revs[1]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[1])))))
                low_bound_agg_usa_woman = np.append(low_bound_agg_usa_woman, (perc[1] + perc[2]) * np.mean(exp_revs[6]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[6])))) + \
                                   perc[0] * np.mean(exp_revs[0]) * (1 - np.sqrt(-np.log(0.05) / (2 * len(exp_revs[0])))))
            else:
                low_bound_agg = np.append(low_bound_agg, 0)
                low_bound_disagg = np.append(low_bound_disagg, 0)
                low_bound_agg_eu_usa = np.append(low_bound_agg_eu_usa, 0)
                low_bound_agg_eu_woman = np.append(low_bound_agg_eu_woman, 0)
                low_bound_agg_usa_woman = np.append(low_bound_agg_usa_woman, 0)

            x = np.random.uniform(0, 1)
            if x <= (1 - t / T):
                context = np.random.randint(0, 5)
                context_array = np.append(context_array, context)
            else:
                context = np.argmax([low_bound_agg[-1], low_bound_disagg[-1], low_bound_agg_eu_usa[-1],
                                     low_bound_agg_eu_woman[-1], low_bound_agg_usa_woman[-1]])
                context_array = np.append(context_array, context)
                context_val_array = np.append(context_val_array, low_bound_agg[-1])
                context_val2_array = np.append(context_val_array, low_bound_disagg[-1])
                context_val3_array = np.append(context_val_array, low_bound_agg_eu_usa[-1])
                context_val4_array = np.append(context_val_array, low_bound_agg_eu_woman[-1])
                context_val5_array = np.append(context_val_array, low_bound_agg_usa_woman[-1])

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ts_rewards_per_experiment_multi.append(ts_learner_multi.collected_rewards)
    ts_rewards_per_experiment_eu_usa.append(ts_learner_multi_eu_usa.collected_rewards)
    ts_rewards_per_experiment_eu_woman.append(ts_learner_multi_eu_woman.collected_rewards)
    ts_rewards_per_experiment_usa_woman.append(ts_learner_multi_usa_woman.collected_rewards)
    ts_rewards_per_experiment_context.append(ts_learner_multi_context.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

    low_bound_agg_avg.append(low_bound_agg)
    low_bound_disagg_avg.append(low_bound_disagg)
    low_bound_agg_eu_usa_avg.append(low_bound_agg_eu_usa)
    low_bound_agg_eu_woman_avg.append(low_bound_agg_eu_woman)
    low_bound_agg_usa_woman_avg.append(low_bound_agg_usa_woman)

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
#plt.plot(np.cumsum(np.mean(opt_multi - ts_rewards_per_experiment, axis=0)), 'r')
#plt.plot(np.cumsum(np.mean(opt_multi - ts_rewards_per_experiment_multi, axis=0)), 'b')
plt.plot(np.cumsum(np.mean(opt_multi - ts_rewards_per_experiment_context, axis=0)), 'c')
#plt.plot(np.cumsum(np.mean((opt_multi - gr_rewards_per_experiment), axis=0)), 'g')
plt.legend(["TS_Context"])
plt.show()

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
#plt.plot(np.cumsum(np.mean(opt_multi - ts_rewards_per_experiment, axis=0)), 'r')
#plt.plot(np.cumsum(np.mean(opt_multi - ts_rewards_per_experiment_multi, axis=0)), 'b')
plt.plot(np.ones(T) * opt_multi, 'b')
plt.plot((np.mean( ts_rewards_per_experiment_context, axis=0)), 'r')
#plt.plot(np.cumsum(np.mean((opt_multi - gr_rewards_per_experiment), axis=0)), 'g')
plt.legend(["TS_Context"])
plt.show()

plt.figure(50)
plt.ylabel("Reward")
plt.xlabel("t")
plot = np.array([])
plot2 = np.array([])
for t in range(0, T):
    plot = np.append(plot, np.cumsum(np.mean(ts_rewards_per_experiment_context, axis=0))[t] / (t + 1))
    plot2 = np.append(plot2, opt_multi)
plt.plot(plot, 'r')
plt.plot(plot2, 'b')
plt.legend(["TS_Context", "Clairvoyant"])
plt.show()

'''
print("ciao")
plt.figure(1)
plt.ylabel("Expected reward")
plt.xlabel("t")
plot = np.array([])
plot2 = np.array([])
plot3 = np.array([])
plot4 = np.array([])
plot5 = np.array([])
for t in range(0, T):
    plot = np.append(plot, np.cumsum(np.mean(ts_rewards_per_experiment, axis=0))[t] / (t + 1))
    plot2 = np.append(plot2, np.cumsum(np.mean(ts_rewards_per_experiment_multi, axis=0))[t] / (t + 1))
    plot3 = np.append(plot3, np.cumsum(np.mean(ts_rewards_per_experiment_eu_usa, axis=0))[t] / (t + 1))
    plot4 = np.append(plot4, np.cumsum(np.mean(ts_rewards_per_experiment_eu_woman, axis=0))[t] / (t + 1))
    plot5 = np.append(plot5, np.cumsum(np.mean(ts_rewards_per_experiment_usa_woman, axis=0))[t] / (t + 1))
plt.plot(plot, 'r')
plt.plot(plot2, 'b')
plt.plot(plot3, 'g')
plt.plot(plot4, 'c')
plt.plot(plot5, 'm')
plt.legend(["TS_Aggregate", "TS_Discrimination", "TS_Agg_eu_usa", "TS_Agg_eu_woman", "TS_Agg_usa_woman"])
plt.show()'''

plt.figure(2)
plt.ylabel("Reward Lower Bound")
plt.xlabel("w")
plt.plot((np.mean(low_bound_agg_avg, axis=0)), 'r')
plt.plot((np.mean(low_bound_disagg_avg, axis=0)), 'b')
plt.plot((np.mean(low_bound_agg_eu_usa_avg, axis=0)), 'g')
plt.plot((np.mean(low_bound_agg_eu_woman_avg, axis=0)), 'c')
plt.plot((np.mean(low_bound_agg_usa_woman_avg, axis=0)), 'm')
plt.legend(["Lower bound aggregate", "Lower bound disaggregate", "Lower bound aggregate eu usa",
            "Lower bound aggregate eu woman", "Lower bound aggregate usa woman"])
plt.show()


'''
for i in range (0, n_experiments):
    plt.figure(i + 3)
    plt.ylabel("Reward Lower Bound")
    plt.xlabel("w")
    plt.plot(low_bound_agg_avg[i], 'r')
    plt.plot(low_bound_disagg_avg[i],  'b')
    plt.plot(low_bound_agg_eu_usa_avg[i],  'g')
    plt.plot(low_bound_agg_eu_woman_avg[i], 'c')
    plt.plot(low_bound_agg_usa_woman_avg[i],  'm')
    plt.legend(["Lower bound aggregate", "Lower bound disaggregate", "Lower bound aggregate eu usa",
                "Lower bound aggregate eu woman", "Lower bound aggregate usa woman"])
    plt.show()
'''