from configuration.SysConfAdv import SysConfAdv
from configuration.SysConfPricing import SysConfPricing
from environment.BudgetEnvironment import BudgetEnvironment
from learners .CombinatorialLearner import CombinatorialLearner
from learners.AggregateLearner import AggregateLearner
from environment.PricingEnvironment import PricingEnvironment
from utility import estimate_daily_n_click
import sys
import numpy as np
from learners.GPTSLearner import GPTSLearner
from run.Advertising import get_optimum
import plot


def update_value_budget(value_click, campaign):
    tmp = []
    for i in range(0, len(value_click)):
        dct = dict(zip(campaign[i].clicks_budget.keys(), np.array(list(campaign[i].clicks_budget.values())) * value_click[i]))
        tmp.append(dct)
    return tmp


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print("ERROR: JSON path required")
        sys.exit(1)

    pricing_conf = SysConfPricing(path + "/configuration/")
    pricing_arms = pricing_conf.get_arms_price()
    arms_user_prob = [[0.5, 0.7, 0.9, 0.35, 0.2], [0.75, 0.9, 0.85, 0.8, 0.7], [0.95, 0.8, 0.2, 0.1, 0.05]]
    user_prob = [0.3, 0.5, 0.2]

    config = SysConfAdv(path + "/configuration/")

    budget = config.budget_sub_campaign()
    functions = config.function()

    sigma = config.init_noise()
    experiment_params = config.init_advertising_experiment()
    kernel = config.init_learner_kernel()
    campaign = []
    T_HORIZON = experiment_params["t_horizon"]      # n of days
    N_OF_EXPERIMENTS = experiment_params["n_experiment"]
    combinatorial_reward_experiment = []

    opt = []
    for i in range(0, len(user_prob)):
        opt.append(max(np.array(arms_user_prob[i]) * pricing_arms))

    for n in range(0, N_OF_EXPERIMENTS):
        print("EXP")
        print(n)
        pricing_learner = AggregateLearner(key_list=[["man_usa"], ["man_eu"], ["woman"]], arms=pricing_arms, confidence=0,
                                       total_aggregate=False)   # NB confidence is not important for this point
        pricing_env = dict(zip(["man_eu", "man_usa", "woman"],
                           [PricingEnvironment(n_arms=len(pricing_arms), probabilities=p) for p in arms_user_prob]))
        daily_number_click = sum(estimate_daily_n_click.n_click_for_days(1, path)[0])
        campaign = []
        for idx in range(0, len(budget)):
            campaign.append(BudgetEnvironment(budget[idx], sigma, functions[idx]))

        learners = []
        for k in range(0, len(budget)):
            learners.append(GPTSLearner(len(budget[k]), budget[k], sigma, kernel[k][0], kernel[k][1]))

        comb_learner = CombinatorialLearner(campaign, learners, experiment_params["cum_budget"])

        collected_reward_adv = []
        pulled_arm = {key: np.random.choice([i for i in range(0, len(pricing_arms))])
                      for key in ["man_eu", "man_usa", "woman"]}

        for day in range(0, T_HORIZON):
            while daily_number_click != 0:  # the user of the day are not terminated
                i = np.random.choice(a=["man_eu", "man_usa", "woman"], p=user_prob)
                daily_number_click += -1

                reward = pricing_env[i].round(pulled_arm[i])
                pricing_learner.update(i, pulled_arm[i], reward)

            arms_idx, value_click = pricing_learner.get_sample_best_arms()
            comb_learner.sc_value_per_click = value_click

            bud_super_arm = comb_learner.knapsacks_solver()
            rewards = comb_learner.get_realization(bud_super_arm)
            comb_learner.update(bud_super_arm, rewards, day)

            daily_number_click = int(sum(rewards))
            user_prob = estimate_daily_n_click.weight(rewards)
            collected_reward_adv.append(sum([rewards[i] * value_click[i] for i in range(0, len(rewards))]))
            for i,el in enumerate(["man_eu", "man_usa", "woman"]):
                pulled_arm[el] = arms_idx[i]
        combinatorial_reward_experiment.append(collected_reward_adv)

    optimum = get_optimum(update_value_budget(opt, campaign),
                          experiment_params["cum_budget"])

    plot.plot_regret_advertising(optimum, combinatorial_reward_experiment)




