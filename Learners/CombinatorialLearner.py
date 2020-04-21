import itertools
import numpy as np


class CombinatorialLearner:
    """
    A class which perform the combinatorial GP bandit for advertising
    Attributes
    ----------
    __budget_env : list[] (of BudgetEnvironment)
        represent the sub campaigns
    __gp_learner : list[] (of GPTSLearner)
        represents the learner for each sub campaigns
    __daily_budget : float
        the total daily budget for the campaign
    """
    def __init__(self, budget_environments, gp_learner, budget):
        self.__budget_env = budget_environments
        self.__gp_learner = gp_learner
        self.__daily_budget = budget
        self.__super_arms = [x for x in itertools.product(*self.__set_super_arms()) if sum(x) <= self.__daily_budget]
        self.collected_reward = []

    def __set_super_arms(self):
        return [x.arms for x in self.__gp_learner]

    def collect_sample(self):
        """
        :return: list[] (list of dictionaries)
            the expected number of click w.r.t the budgets for each sub campaigns
        """
        sample = []
        for gpl in self.__gp_learner:
            sample = sample.append(gpl.pull_arms)
        return sample

    def knapsacks_solver(self):
        """
        :return: tuple
            the best super arm selected at the current round
        """
        samples = self.collect_sample()
        reward_list = np.array([])
        for s_arm in self.__super_arms:
            reward = 0
            for idx in range(0, len(s_arm)):
                reward += samples[idx][s_arm[idx]]
            reward_list = np.append(reward_list, reward)

        best_super_arm = self.__super_arms[int(np.argmax(reward_list))]
        return best_super_arm

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
        self.collected_reward.append(reward)
        return reward

    def update(self, super_arm, reward):
        """
        Updates the model
        :param super_arm: list[]
            best super-arm
        :param reward: list[]
            reward of the arm selected
        :return:
        """
        for i in range(0, len(self.__gp_learner)):
            self.__gp_learner[i].update(super_arm[i], reward[i])

    @property
    def collected_reward(self):
        return self._collected_reward

    @collected_reward.setter
    def collected_reward(self, value):
        self._collected_reward = value