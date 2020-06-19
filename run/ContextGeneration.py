from learners.SuperSetContext import SuperSetContext
from utility import estimate_daily_n_click
from configuration.SysConfPricing import SysConfPricing
from utility.estimate_daily_n_click import weight
import numpy as np
from context_feature_generation import generate_context_feature
from environment.PricingEnvironment import PricingEnvironment
import plot


if __name__ == "__main__":

    conf = SysConfPricing("/home/orso/Documents/POLIMI/DataIntelligenceApplication/DIA-Project/configuration/")
    arms = conf.get_arms_price()
    arms_user_prob = conf.get_function()

    T, n_experiments, number_week = conf.get_experiment_context_info()
    string_partition = generate_context_feature()

    collected_reward = []
    collected_reward_experiment = []
    prob_user_over_experiment = []

    for e in range(0, n_experiments):
        environment = dict(zip(string_partition[0], [PricingEnvironment(n_arms=len(arms), probabilities=p) for p in arms_user_prob]))
        context = SuperSetContext(string_partition, arms, 0)
        for week in range(0, number_week):
            n_click = estimate_daily_n_click.n_click_for_days(T)
            click = np.array([sum(el) for el in n_click])  # total number of user each day
            prob_user = weight(np.mean(n_click, axis=0))
            context.active_context()

            tmp_reward = []
            for t in range(0, T):
                while click[t] != 0:  # the user of the day are not terminated
                    i = np.random.choice(a=string_partition[0], p=prob_user)
                    click[t] += -1

                    pulled_arm = context.pull_arm(i)
                    reward = environment[i].round(pulled_arm)
                    context.update(i, pulled_arm, reward)

            collected_reward = collected_reward + context.collected_reward()
            prob_user_over_experiment.append(prob_user)
        collected_reward_experiment.append(collected_reward)

    prob_user_over_experiment = np.mean(prob_user_over_experiment, axis=0)
    opt = 0
    for i in range(0, len(prob_user_over_experiment)):
        opt = opt + np.max(arms_user_prob[i] * arms) * prob_user_over_experiment[i]
    plot.plot_cum_regret(opt, collected_reward_experiment)