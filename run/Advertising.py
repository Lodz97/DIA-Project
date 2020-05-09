import numpy as np
from environment import ClickFunction, BudgetEnvironment
from Learners import CombinatorialLearner, GPTSLearner
import SystemConfiguration
from combinatorial_solver import KnapsackSolver
import plot


def get_optimum(dic_budget, cumulative_budget):
    k_sol = KnapsackSolver.KnapsackSolver(dic_budget, cumulative_budget)
    tmp = k_sol.solve(dic_budget)
    print(tmp[0])
    return tmp[1]


if __name__ == "__main__":

    config = SystemConfiguration.SystemConfiguration("/home/orso/Documents/POLIMI/DataIntelligenceApplication/DIA-Project/run/")

    param_sub_c1 = config.init_sub_campaign("sub_campaign1")
    budget_sub_c1 = np.linspace(param_sub_c1["min_budget"], param_sub_c1["max_budget"], param_sub_c1["n_arms"])
    param_sub_c2 = config.init_sub_campaign("sub_campaign2")
    budget_sub_c2 = np.linspace(param_sub_c2["min_budget"], param_sub_c2["max_budget"], param_sub_c2["n_arms"])
    param_sub_c3 = config.init_sub_campaign("sub_campaign3")
    budget_sub_c3 = np.linspace(param_sub_c3["min_budget"], param_sub_c3["max_budget"], param_sub_c3["n_arms"])

    sigma = config.init_noise()

    func_c1 = ClickFunction.ClickFunction(*config.init_function("func_man_eu_p1"))
    func_c2 = ClickFunction.ClickFunction(*config.init_function("func_man_usa_p1"))
    func_c3 = ClickFunction.ClickFunction(*config.init_function("func_woman_p1"))

    experiment_params = config.init_advertising_experiment2()
    t_horizon = experiment_params["t_horizon"]
    cum_budget = experiment_params["cumulative_budget"]
    combinatorial_reward_experiment = []
    campaign = []
    for i in range(0, config.init_advertising_experiment2()["n_experiment"]):

        sub_c1 = BudgetEnvironment.BudgetEnvironment(budget_sub_c1, sigma, func_c1)
        sub_c2 = BudgetEnvironment.BudgetEnvironment(budget_sub_c2, sigma, func_c2)
        sub_c3 = BudgetEnvironment.BudgetEnvironment(budget_sub_c3, sigma, func_c3)
        campaign = [sub_c1, sub_c2, sub_c3]

        theta, l_scale = config.init_learner_kernel()
        gpts_l1 = GPTSLearner.GPTSLearner(len(budget_sub_c1), budget_sub_c1, sigma, theta, l_scale)
        gpts_l2 = GPTSLearner.GPTSLearner(len(budget_sub_c2), budget_sub_c2, sigma, theta, l_scale)
        gpts_l3 = GPTSLearner.GPTSLearner(len(budget_sub_c3), budget_sub_c3, sigma, theta, l_scale)
        learners = [gpts_l1, gpts_l2, gpts_l3]

        comb_learner = CombinatorialLearner.CombinatorialLearner(campaign, learners, cum_budget)

        for t in range(0, t_horizon):
            super_arm = comb_learner.knapsacks_solver()
            rewards = comb_learner.get_realization(super_arm)
            comb_learner.update(super_arm, rewards,t)
            #if t % 10 == 0:
            #    comb_learner.plot_regression(t, [func_c1,func_c2,func_c3])
        combinatorial_reward_experiment.append(comb_learner.collected_reward)
        print(i)

    optimum = get_optimum([campaign[0].clicks_budget, campaign[1].clicks_budget, campaign[2].clicks_budget], cum_budget)
    print(optimum)
    plot.plot_regret_advertising(optimum, combinatorial_reward_experiment)
