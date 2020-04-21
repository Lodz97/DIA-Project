import numpy as np
import matplotlib as plt
from environment import ClickFunction, BudgetEnvironment
from Learners import CombinatorialLearner, GPTSLearner
import SystemConfiguration
# ERROR
# it does not compile!!!!
if __name__ == "__main__":

    config = SystemConfiguration.SystemConfiguration("/../")

    param_sub_c1 = config.init_sub_campaign("sub_campaign1")
    budget_sub_c1 = np.linspace(param_sub_c1["min_budget"], param_sub_c1["max_budget"], param_sub_c1["n_arms"])
    param_sub_c2 = config.init_sub_campaign("sub_campaign2")
    budget_sub_c2 = np.linspace(param_sub_c2["min_budget"], param_sub_c2["max_budget"], param_sub_c2["n_arms"])
    param_sub_c3 = config.init_sub_campaign("sub_campaign3")
    budget_sub_c3 = np.linspace(param_sub_c3["min_budget"], param_sub_c3["max_budget"], param_sub_c3["n_arms"])

    sigma = config.init_noise()
    total_budget = config.init_total_budget()

    func_c1 = ClickFunction.ClickFunction(*config.init_function("func_man_eu"))
    func_c2 = ClickFunction.ClickFunction(*config.init_function("func_man_usa"))
    func_c3 = ClickFunction.ClickFunction(*config.init_function("func_woman"))

    combinatorial_reward_experiment = []

    for i in range(0, config.init_advertising_experiment2()["n_experiment"]):

        sub_c1 = BudgetEnvironment.BudgetEnvironment(budget_sub_c1, sigma, func_c1)
        sub_c2 = BudgetEnvironment.BudgetEnvironment(budget_sub_c2, sigma, func_c2)
        sub_c3 = BudgetEnvironment.BudgetEnvironment(budget_sub_c3, sigma, func_c3)
        campaign = [sub_c1, sub_c2, sub_c3]

        gpts_l1 = GPTSLearner.GPTSLearner(len(budget_sub_c1), budget_sub_c1, sigma)
        gpts_l2 = GPTSLearner.GPTSLearner(len(budget_sub_c2), budget_sub_c2, sigma)
        gpts_l3 = GPTSLearner.GPTSLearner(len(budget_sub_c3), budget_sub_c3, sigma)
        learners = [gpts_l1, gpts_l2, gpts_l3]

        comb_learner = CombinatorialLearner.CombinatorialLearner(campaign, learners, total_budget)

        for t in range(0, config.init_advertising_experiment2()["t_horizon"]):
            super_arm = comb_learner.knapsacks_solver()
            rewards = comb_learner.get_realization(super_arm)
            comb_learner.update(super_arm, rewards)

        combinatorial_reward_experiment.append(comb_learner.collected_reward)

    print(combinatorial_reward_experiment)