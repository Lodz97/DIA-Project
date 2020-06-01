from environment.PricingEnvironment import PricingEnvironment
from learners.PricingTSLearner import PricingTSLearner
from learners.PricingGreedyLearner import PricingGreedyLearner
from environment.ConversionRate import *
from configuration.SysConfPricing import SysConfPricing
from utility import estimate_daily_n_click

if __name__ == "__main__":

    conf = SysConfPricing("/home//orso/Documents/POLIMI/DataIntelligenceApplication/DIA-Project/configuration/")
    profit_array = conf.get_profit()
    p = conf.get_aggregate_function()
    #   show_total_profit(conf.get_users_percentages())
    n_arms = len(p)
    opt = np.max(profit_array * p)

    T, n_experiments = conf.get_experiment_info()
    T = int(estimate_daily_n_click.n_click_for_days(T))
    ts_rewards_per_experiment = []
    gr_rewards_per_experiment = []

    for e in range(0, n_experiments):
        env = PricingEnvironment(n_arms=n_arms, probabilities=p)
        ts_learner = PricingTSLearner(n_arms=n_arms, profit_array=profit_array)
        gr_learner = PricingGreedyLearner(n_arms=n_arms, profit_array=profit_array)

        for t in range(0, T):

            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

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
