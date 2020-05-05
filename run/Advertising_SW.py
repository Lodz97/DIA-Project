import SystemConfiguration
import numpy as np
from environment.ClickFunction import ClickFunction
from environment.NonStationaryBudgetEnvironment import NonStationaryBudgetEnvironment
from Learners import GPTSLearner, SWGPTSLearner
from combinatorial_solver.KnapsackSolver import KnapsackSolver
from Learners.CombinatorialLearner import CombinatorialLearner
import plot


def get_optimum(dic_budget):
    reward_non_stationary = []
    l_tmp = [element[0] for element in dic_budget]
    solver = KnapsackSolver(l_tmp)
    for idx in range(len(dic_budget[0])):
        l_tmp = [element[idx] for element in dic_budget]
        reward_non_stationary.append(solver.solve(l_tmp)[1])

    return reward_non_stationary


if __name__ == "__main__":

    config = SystemConfiguration.SystemConfiguration()

    param_sub_c1 = config.init_sub_campaign("sub_campaign1")
    budget_sub_c1 = np.linspace(param_sub_c1["min_budget"], param_sub_c1["max_budget"], param_sub_c1["n_arms"])
    param_sub_c2 = config.init_sub_campaign("sub_campaign2")
    budget_sub_c2 = np.linspace(param_sub_c2["min_budget"], param_sub_c2["max_budget"], param_sub_c2["n_arms"])
    param_sub_c3 = config.init_sub_campaign("sub_campaign3")
    budget_sub_c3 = np.linspace(param_sub_c3["min_budget"], param_sub_c3["max_budget"], param_sub_c3["n_arms"])

    sigma = config.init_noise()
    t_horizon = config.init_advertising_experiment3()["t_horizon"]
    n_experiment = config.init_advertising_experiment3()["n_experiment"]
    # TODO
    # must window size be multiplied by the number of phases????
    window_size = 3*int(np.sqrt(t_horizon))

    func_list1 = config.function()
    function_plot = ClickFunction.function_list_by_phase(func_list1)
    
    combinatorial_reward_experiment = []
    sw_combinatorial_reward_experiment = []
    campaign = []

    for i in range(0, n_experiment):

        sub_c1 = NonStationaryBudgetEnvironment(budget_sub_c1, sigma, func_list1[0], t_horizon)
        sub_c2 = NonStationaryBudgetEnvironment(budget_sub_c2, sigma, func_list1[1], t_horizon)
        sub_c3 = NonStationaryBudgetEnvironment(budget_sub_c3, sigma, func_list1[2], t_horizon)

        campaign = [sub_c1, sub_c2, sub_c3]
        theta, l_scale = config.init_learner_kernel()
        gpts_l1 = GPTSLearner.GPTSLearner(len(budget_sub_c1), budget_sub_c1, sigma, theta, l_scale)
        gpts_l2 = GPTSLearner.GPTSLearner(len(budget_sub_c2), budget_sub_c2, sigma, theta, l_scale)
        gpts_l3 = GPTSLearner.GPTSLearner(len(budget_sub_c3), budget_sub_c3, sigma, theta, l_scale)
        learners = [gpts_l1, gpts_l2, gpts_l3]

        theta, l_scale = config.init_learner_kernel()
        sw_gpts_l1 = SWGPTSLearner.SWGPTSLearner(len(budget_sub_c1), budget_sub_c1, sigma, theta, l_scale, window_size)
        sw_gpts_l2 = SWGPTSLearner.SWGPTSLearner(len(budget_sub_c2), budget_sub_c2, sigma, theta, l_scale, window_size)
        sw_gpts_l3 = SWGPTSLearner.SWGPTSLearner(len(budget_sub_c3), budget_sub_c3, sigma, theta, l_scale, window_size)

        sw_learners = [sw_gpts_l1, sw_gpts_l2, sw_gpts_l3]

        comb_learner = CombinatorialLearner(campaign, learners)
        sw_comb_learner = CombinatorialLearner(campaign, sw_learners)

        for t in range(0, t_horizon):
            super_arm = comb_learner.knapsacks_solver()
            rewards = comb_learner.get_realization(super_arm)
            comb_learner.update(super_arm, rewards, t)
            sw_super_arm = sw_comb_learner.knapsacks_solver()
            sw_rewards = sw_comb_learner.get_realization(sw_super_arm)
            sw_comb_learner.update(sw_super_arm, sw_rewards, t)
            #print("non stat")
            #print(sw_super_arm)
            if t % 20 == 0:
                tmp = 40
                sw_comb_learner.plot_regression(t, function_plot[int(t/tmp)])
                #comb_learner.plot_regression(t, function_plot[int(t/tmp)])
        combinatorial_reward_experiment.append(comb_learner.collected_reward)
        sw_combinatorial_reward_experiment.append(sw_comb_learner.collected_reward)
        print(i)

    optimum = get_optimum([campaign[0].list_clicks_budget, campaign[1].list_clicks_budget, campaign[2].list_clicks_budget])
    plot.plot_regret_comparison(optimum, combinatorial_reward_experiment, sw_combinatorial_reward_experiment)
