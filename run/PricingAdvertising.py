from configuration.SysConfAdv import SysConfAdv
from configuration.SysConfPricing import SysConfPricing
from environment.BudgetEnvironment import BudgetEnvironment
from learners .CombinatorialLearner import CombinatorialLearner
from learners.AggregateLearner import AggregateLearner
from environment.PricingEnvironment import PricingEnvironment
from utility import estimate_daily_n_click
from utility.estimate_daily_n_click import weight
import numpy as np
from learners.GPTSLearner import GPTSLearner

## CREATE
## total_disaggregate_learner ---> AggregateLearner
## pricing environment ---> PricingEnvironment
## advertising environment ---> BudgetEnvironment
## adv learner ---> GPTSLearner
## optimizer --> KnapsackSolver
## run pricing with random number of users

# loop
## 0) advertising_environment: number of users per class
## 1) run pricing day: value clicks (opt_price*conversion_rate)
## 2) run advertising: estimate_of_click_number
## 3) solve optimization: optimal budget(and each sub campaign budget)

if __name__ == "__main__":
    pricing_conf = SysConfPricing("/home/mattia/PyProjects/DIA-Project/configuration/")
    pricing_arms = pricing_conf.get_arms_price()
    arms_user_prob = [[0.5, 0.7, 0.9, 0.35, 0.2], [0.75, 0.9, 0.85, 0.8, 0.7], [0.95, 0.8, 0.2, 0.1, 0.05]]
    user_prob = [0.3, 0.5, 0.2]

    config = SysConfAdv("/home/mattia/PyProjects/DIA-Project/configuration/")

    budget = config.budget_sub_campaign()
    functions = config.function()

    sigma = config.init_noise()
    experiment_params = config.init_advertising_experiment()
    kernel = config.init_learner_kernel()
    campaign = []
    T_HORIZON = experiment_params["t_horizon"]      # n of days
    N_OF_EXPERIMENTS = experiment_params["n_experiment"]

    for n in range(0, N_OF_EXPERIMENTS):

        pricing_learner = AggregateLearner(key_list=[["man_usa"], ["man_eu"], ["woman"]], arms=pricing_arms, confidence=0,
                                       total_aggregate=False)   # NB confidence is not important for this point
        pricing_env = dict(zip(["man_eu", "man_usa", "woman"],
                           [PricingEnvironment(n_arms=len(pricing_arms), probabilities=p) for p in arms_user_prob]))
        daily_number_click = sum(estimate_daily_n_click.n_click_for_days(1))
        campaign = []
        for idx in range(0, len(budget)):
            campaign.append(BudgetEnvironment(budget[idx], sigma, functions[idx]))

        learners = []
        for k in range(0, len(budget)):
            learners.append(GPTSLearner(len(budget[k]), budget[k], sigma, kernel[k][0], kernel[k][1]))

        comb_learner = CombinatorialLearner(campaign, learners, experiment_params["cum_budget"])

        for day in range(0, T_HORIZON):

            # pricing problem
            while daily_number_click != 0:  # the user of the day are not terminated
                i = np.random.choice(a=["man_eu", "man_usa", "woman"], p=user_prob)
                daily_number_click += -1

                pulled_arm = pricing_learner.pull_arm(i)
                reward = pricing_env[i].round(pulled_arm)
            # select best arms + use value clicks
            # advertising
            # todo daily_number_click update
            # todo user_prob update
