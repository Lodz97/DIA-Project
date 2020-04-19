import itertools
import numpy as np


class CombinatorialLearner:
    """
    A class which perform the combinatorial GP bandits for advertising
    Attributes
    ----------
    __budget_env : list[] (of BudgetEnvironment)
        represent the subcampaigns
    __gp_learner : list[] (of GPTSLearner)
        represents the learner for each subcampaigns
    __daily_budget : float
        the total daily budget for the campaign
    """
    def __init__(self, budget_environments, gp_learner, budget):
        self.__budget_env = budget_environments
        self.__gp_learner = gp_learner
        self.__daily_budget = budget
        self.__super_arms = [x for x in itertools.product(*self.__set_super_arms()) if sum(x) <= self.__daily_budget]

    def __set_super_arms(self):
        return [x.arms for x in self.__gp_learner]

    def collect_sample(self):
        """
        Returns
        ----------
        sample : list[] (list of dictionaries)
            the expected number of click w.r.t the budgets for each subcampaigns
        """
        sample = []
        for gpl in self.__gp_learner:
            sample = sample.append(gpl.pull_arms)
        return sample

    def knapsacks_solver(self):
        """
        Returns
        ----------
        best_super_arm : tuple
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
        Parameters
        super_arm : tuple
            best super-arm
        ----------
        Returns
        ----------
        reward : list[]
            the reward of the arm selected (one for each sub campaign)
        """
        reward = []
        for i in range(0, len(self.__budget_env)):
            reward.append(self.__budget_env[i].round(super_arm[i]))
        return reward

    def update(self, super_arm, reward):
        """
        Updates the model
        Parameters
        ----------
        super_arm : list[]
            best super-arm
        reward : list[]
            reward of the arm selected
        """
        for i in range(0, len(self.__gp_learner)):
            self.__gp_learner[i].update(super_arm[i], reward[i])
