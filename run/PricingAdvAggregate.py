from configuration.SysConfAdv import SysConfAdv
from configuration.SysConfPricing import SysConfPricing
from environment.BudgetEnvironment import BudgetEnvironment
from learners .CombinatorialLearner import CombinatorialLearner
from learners.PricingTSLearner import PricingTSLearner
from learners.AggregateLearner import AggregateLearner
from environment.PricingEnvironment import PricingEnvironment
from utility import estimate_daily_n_click
from utility.estimate_daily_n_click import weight
import numpy as np
from learners.GPTSLearner import GPTSLearner
from run.Advertising import get_optimum
import plot
import sys


def update_value_budget(campaign, value_click):
    tmp = []
    for i in range(0, len(campaign)):
        dct = dict(zip(campaign[i].clicks_budget.keys(), np.array(list(campaign[i].clicks_budget.values()))
                       * value_click[i]))
        tmp.append(dct)
    return tmp


def get_real_value_clicks(disaggregate_user_prob, pricing_arms):
    tmp = []
    for k in range(0, len(disaggregate_user_prob)):
        tmp.append(np.array(disaggregate_user_prob[k]) * np.array(pricing_arms))

    value_click_arms = []
    for idk in range(0, len(pricing_arms)):
        value_click_arms.append([tmp[id][idk] for id in range(0, len(disaggregate_user_prob))])
    return value_click_arms


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        sys.exit(1)
    pricing_conf = SysConfPricing(path + "/DIA-Project/configuration/")
    pricing_arms = pricing_conf.get_arms_price()
    user_prob = [0.3, 0.5, 0.2]
    arms_user_prob = pricing_conf.get_aggregate_function(user_prob)
    disaggregate_user_prob = [[0.5, 0.7, 0.9, 0.35, 0.2], [0.75, 0.9, 0.85, 0.8, 0.7], [0.95, 0.8, 0.2, 0.1, 0.05]]
    print(arms_user_prob)

    config = SysConfAdv(path + "/DIA-Project/configuration/")

    budget = config.budget_sub_campaign()
    functions = config.function()

    sigma = config.init_noise()
    experiment_params = config.init_advertising_experiment()
    kernel = config.init_learner_kernel()
    campaign = []
    T_HORIZON = experiment_params["t_horizon"]      # n of days
    N_OF_EXPERIMENTS = experiment_params["n_experiment"]
    combinatorial_reward_experiment = []

    for n in range(0, N_OF_EXPERIMENTS):
        print("EXP")
        print(n)
        pricing_learner_aggregate = PricingTSLearner(n_arms=len(pricing_arms), arms=pricing_arms)
        pricing_learner_disaggregate = AggregateLearner(key_list=[["man_usa"], ["man_eu"], ["woman"]], arms=pricing_arms,
                                                        confidence=0, total_aggregate=False)
        pricing_env = dict(zip(["man_eu", "man_usa", "woman"],
                               [PricingEnvironment(n_arms=len(pricing_arms), probabilities=p)
                                for p in disaggregate_user_prob]))
        daily_number_click = sum(estimate_daily_n_click.n_click_for_days(1)[0])
        campaign = []
        for idx in range(0, len(budget)):
            campaign.append(BudgetEnvironment(budget[idx], sigma, functions[idx]))

        learners = []
        for k in range(0, len(budget)):
            learners.append(GPTSLearner(len(budget[k]), budget[k], sigma, kernel[k][0], kernel[k][1]))

        comb_learner = CombinatorialLearner(campaign, learners, experiment_params["cum_budget"])

        collected_reward_adv = []

        for day in range(0, T_HORIZON):
            #print("DAY")
            #print(day)
            while daily_number_click != 0:  # the user of the day are not terminated
                i = np.random.choice(a=["man_eu", "man_usa", "woman"], p=user_prob)
                daily_number_click += -1

                #pulled_arm = pricing_learner_aggregate.pull_arm()
                pulled_arm = np.random.choice([0, 1, 2, 3, 4])
                reward = pricing_env[i].round(pulled_arm)
                pricing_learner_aggregate.update(pulled_arm, reward)
                pricing_learner_disaggregate.update(i, pulled_arm, reward)

            value_click = []
            for j in range(0, len(pricing_arms)):
                value_click.append(pricing_learner_disaggregate.get_reward_arm(j))

            bud_super_arm = []
            value_number_click_arm = []
            for j in range(0, len(pricing_arms)):
                comb_learner.sc_value_per_click = value_click[j]
                temp1, temp2 = comb_learner.knapsacks_solver_value()
                bud_super_arm.append(temp1)
                value_number_click_arm.append(temp2)

            idx_max = np.argmax(value_number_click_arm)

            rewards = comb_learner.get_realization(bud_super_arm[idx_max])
            comb_learner.update(bud_super_arm[idx_max], rewards, day)

            daily_number_click = int(sum(rewards))
            user_prob = estimate_daily_n_click.weight(rewards)

            collected_reward_adv.append(sum([rewards[i] * value_click[idx_max][i] for i in range(0, len(rewards))]))

        combinatorial_reward_experiment.append(collected_reward_adv)

    values = get_real_value_clicks(disaggregate_user_prob, pricing_arms)
    optimum = []
    for k in range(0, len(pricing_arms)):
        optimum.append(get_optimum(update_value_budget(campaign, values[k]), experiment_params["cum_budget"]))

    opt = max(optimum)
    plot.plot_regret_advertising(opt, combinatorial_reward_experiment)





