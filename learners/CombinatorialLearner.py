import itertools
import numpy as np
from combinatorial_solver.KnapsackSolver import KnapsackSolver


class CombinatorialLearner:
    """
    A class which perform the combinatorial GP bandit for advertising
    Attributes
    ----------
    __budget_env : list[] (of BudgetEnvironment)
        represent the sub campaigns
    __gp_learner : list[] (of GPTSLearner)
        represents the learner for each sub campaigns
    __knapsacks_solver : KnapsackSolver
    __collected_reward : list
        total reward of the campaign
    """
    def __init__(self, budget_environments, gp_learner, cum_budget):
        self.__budget_env = budget_environments
        self.__gp_learner = gp_learner
        self.collected_reward = []
        self.__knapsacks_solver = KnapsackSolver([x.clicks_budget for x in budget_environments], cum_budget)
        self.sc_value_per_click = np.ones(len(gp_learner))

    def __set_super_arms(self):
        return [x.arms for x in self.__gp_learner]

    def collect_sample(self):
        """
        :return: list[] (list of dictionaries)
            the expected number of click w.r.t the budgets for each sub campaigns
        """
        sample = []
        for idx in range(0, len(self.__gp_learner)):
            tmp = self.__gp_learner[idx].pull_arm()
            dct = dict(zip(tmp.keys(), np.array(list(tmp.values()))*self.sc_value_per_click[idx]))
            sample.append(dct)
        return sample

    def knapsacks_solver(self):
        """
        :return: tuple
            the best super arm selected at the current round
        """
        samples = self.collect_sample()
        return self.__knapsacks_solver.solve(samples)[0]

    def get_realization(self, super_arm):
        """
        :param super_arm: tuple
            best super-arm
        :return: list[]
            the reward of the arm selected (one for each sub campaign)
        """
        reward = []
        for i in range(0, len(self.__budget_env)):
            reward.append(self.__budget_env[i].round(super_arm[i]))
        self.collected_reward.append(sum(reward))
        return reward

    def update(self, super_arm, reward, t):
        """
        Updates the model
        :param super_arm: list[]
            best super-arm
        :param reward: list[]
            reward of the arm selected
        :return:
        """
        for i in range(0, len(self.__gp_learner)):
            idx = np.where(self.__gp_learner[i].arms == super_arm[i])[0][0]     # it returns a tuple of arrays
            self.__gp_learner[i].update(idx, reward[i])

    def plot_regression(self, t, func_list):
        for idx in range(0, len(func_list)):
            self.__gp_learner[idx].plot_process(func_list[idx], t)


    @property
    def collected_reward(self):
        return self.__collected_reward

    @collected_reward.setter
    def collected_reward(self, value):
        self.__collected_reward = value