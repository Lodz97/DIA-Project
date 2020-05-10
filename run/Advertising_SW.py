from configuration import SysConfAdvSW
import numpy as np
from environment.NonStationaryBudgetEnvironment import NonStationaryBudgetEnvironment
from Learners import GPTSLearner, SWGPTSLearner
from combinatorial_solver.KnapsackSolver import KnapsackSolver
from Learners.CombinatorialLearner import CombinatorialLearner
import plot


def get_optimum(dic_budget, cum_budget):
    print(dic_budget)
    reward_non_stationary = []
    l_tmp = [element[0] for element in dic_budget]
    solver = KnapsackSolver(l_tmp, cum_budget)
    for idx in range(len(dic_budget[0])):
        l_tmp = [element[idx] for element in dic_budget]
        reward_non_stationary.append(solver.solve(l_tmp)[1])

    return reward_non_stationary


if __name__ == "__main__":

    config = SysConfAdvSW.SysConfAdvSW("/home/orso/Documents/POLIMI/DataIntelligenceApplication/DIA-Project/configuration/")
    sigma = config.init_noise()
    experiment_params = config.init_advertising_experiment()
    window_size = 3*int(np.sqrt(experiment_params["t_horizon"]))

    kernel = config.init_learner_kernel()
    budget = config.budget_sub_campaign()
    func_list = config.function()
    function_plot = config.function_list_by_phase(func_list)
    combinatorial_reward_experiment = []
    sw_combinatorial_reward_experiment = []
    campaign = []

    for i in range(0, experiment_params["n_experiment"]):
        campaign = []
        for idx in range(0, len(budget)):
            campaign.append(NonStationaryBudgetEnvironment(budget[idx], sigma, func_list[idx],
                                                           experiment_params["t_horizon"]))

        learners = []
        sw_learners = []
        for k in range(0, len(budget)):
            learners.append(GPTSLearner.GPTSLearner(len(budget[k]), budget[k], sigma, kernel[k][0], kernel[k][1]))
            sw_learners.append(SWGPTSLearner.SWGPTSLearner(len(budget[k]), budget[k], sigma, kernel[k][0], kernel[k][1],
                                                           window_size))

        comb_learner = CombinatorialLearner(campaign, learners, experiment_params["cum_budget"])
        sw_comb_learner = CombinatorialLearner(campaign, sw_learners, experiment_params["cum_budget"])

        for t in range(0, experiment_params["t_horizon"]):
            super_arm = comb_learner.knapsacks_solver()
            rewards = comb_learner.get_realization(super_arm)
            comb_learner.update(super_arm, rewards, t)
            sw_super_arm = sw_comb_learner.knapsacks_solver()
            sw_rewards = sw_comb_learner.get_realization(sw_super_arm)
            sw_comb_learner.update(sw_super_arm, sw_rewards, t)
            #if t % 20 == 0:
                #tmp = 130
                #sw_comb_learner.plot_regression(t, function_plot[int(t/tmp)])
                #comb_learner.plot_regression(t, function_plot[int(t/tmp)])
        combinatorial_reward_experiment.append(comb_learner.collected_reward)
        sw_combinatorial_reward_experiment.append(sw_comb_learner.collected_reward)
        print(i)

    optimum = get_optimum([campaign[0].list_clicks_budget, campaign[1].list_clicks_budget,
                           campaign[2].list_clicks_budget], experiment_params["cum_budget"])
    print(optimum)
    plot.plot_regret_comparison(optimum, combinatorial_reward_experiment, sw_combinatorial_reward_experiment)
