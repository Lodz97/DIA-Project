from environment import BudgetEnvironment
from learners import CombinatorialLearner, GPTSLearner
from configuration import SysConfAdv
from combinatorial_solver import KnapsackSolver
import plot


def get_optimum(dic_budget, cumulative_budget):
    k_sol = KnapsackSolver.KnapsackSolver(dic_budget, cumulative_budget)
    tmp = k_sol.solve(dic_budget)
    return tmp[1]


if __name__ == "__main__":
    config = SysConfAdv.SysConfAdv("C:\\Users\\Giacomo\\PycharmProjects\\DIA-Project-GIT\\configuration\\")

    budget = config.budget_sub_campaign()
    functions = config.function()

    sigma = config.init_noise()
    experiment_params = config.init_advertising_experiment()
    kernel = config.init_learner_kernel()
    combinatorial_reward_experiment = []
    campaign = []

    for i in range(0, experiment_params["n_experiment"]):
        campaign = []
        for idx in range(0, len(budget)):
            campaign.append(BudgetEnvironment.BudgetEnvironment(budget[idx], sigma, functions[idx]))

        learners = []
        for k in range(0, len(budget)):
            learners.append(GPTSLearner.GPTSLearner(len(budget[k]), budget[k], sigma, kernel[k][0], kernel[k][1]))

        comb_learner = CombinatorialLearner.CombinatorialLearner(campaign, learners, experiment_params["cum_budget"])

        for t in range(0, experiment_params["t_horizon"]):
            super_arm = comb_learner.knapsacks_solver()
            rewards = comb_learner.get_realization(super_arm)
            comb_learner.update(super_arm, rewards, t)
            #if t % 10 == 0:
                #comb_learner.plot_regression(t, functions)

        combinatorial_reward_experiment.append(comb_learner.collected_reward)

    optimum = get_optimum([campaign[0].clicks_budget, campaign[1].clicks_budget, campaign[2].clicks_budget],
                          experiment_params["cum_budget"])
    plot.plot_regret_advertising(optimum, combinatorial_reward_experiment)
