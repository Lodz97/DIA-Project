from environment.PricingEnvironment import PricingEnvironment
from learners.PricingTSLearner import PricingTSLearner
from learners.PricingGreedyLearner import PricingGreedyLearner
from environment.ConversionRate import *
from configuration.SysConfPricing import SysConfPricing
from utility import estimate_daily_n_click
from utility.estimate_daily_n_click import weight
import numpy as np


if __name__ == "__main__":

    conf = SysConfPricing("/home/orso/Documents/POLIMI/DataIntelligenceApplication/DIA-Project/configuration/")
    profit_array = conf.get_profit() #not used
    arms_prob = conf.get_function() #list prob of arms for each user
    arms = conf.get_arms_price()

    #   show_total_profit(conf.get_users_percentages())
    T, n_experiments = conf.get_experiment_pricing_info()
    ts_rewards_per_experiment = []
    gr_rewards_per_experiment = []
    n_click = estimate_daily_n_click.n_click_for_days(T)

    prob_user = weight(np.mean(n_click, axis=0))

    p = conf.get_aggregate_function(prob_user)
    n_arms = len(p)
    opt = np.max(p*arms)

    for e in range(0, n_experiments):
        env = PricingEnvironment(n_arms=n_arms, probabilities=p)
        environment_ts = [PricingEnvironment(n_arms=n_arms, probabilities=p) for p in arms_prob]
        environment_gr = [PricingEnvironment(n_arms=n_arms, probabilities=p) for p in arms_prob]
        ts_learner = PricingTSLearner(n_arms=n_arms, arms=arms)
        gr_learner = PricingGreedyLearner(n_arms=n_arms, profit_array=profit_array)
        click = np.array([sum(el) for el in n_click])

        for t in range(0, T):
            env_click = [k for k in range(0, len(n_click[0]))]
            while click[t] != 0:
                i = np.random.choice(a=env_click, p=prob_user)
                click[t] += -1

                pulled_arm = ts_learner.pull_arm()
                reward = environment_ts[i].round(pulled_arm)
                ts_learner.update(pulled_arm, reward)

                pulled_arm = gr_learner.pull_arm()
                reward = environment_gr[i].round(pulled_arm)
                gr_learner.update(pulled_arm, reward)

        ts_rewards_per_experiment.append(ts_learner._collected_rewards)
        gr_rewards_per_experiment.append(gr_learner._collected_rewards)


    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
    #plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
    plt.legend(["TS", "Greedy"])
    plt.show()
