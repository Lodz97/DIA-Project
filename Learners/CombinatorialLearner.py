

class CombinatorialLearner:
    """
    A class which perform the combinatorial GP bandits for advertising
    Attributes
    ----------
    __budget_env : list[] (of BudgetEnvironment)
        represent the subcampaigns
    __GP_learner : list[] (of GPTSLearner)
        represents the learner for each subcampaigns
    __daily_budget : float
        the total daily budget for the campaign
    """
    def __init__(self, budget_environments, GP_learner, budget):
        self.__budget_env = budget_environments
        self.__GP_learner = GP_learner
        self.__daily_budgets = budget

    def collect_sample(self):
        """
        Returns
        ----------
        sample : list[]
            the expected number of click w.r.t the budgets for each subcampaigns
        """
        sample = []
        for gpl in self.__GP_learner:
            sample = sample.append(gpl.pull_arms)
        return sample

    def knapsacks_solver(self):
        pass
        #return superArm

    def get_realization(self, superArm):
        """
        Parameters
        superArm : lis[]
            best superarm
        ----------
        Returns
        ----------
        reward : list[]
            the reward of the arm selected (one for each subcampigns)
        """
        reward = []
        for i in range(0,len(self.__budget_env)):
            reward = reward.append(self.__budget_env[i].round(superArm[i]))
        return reward

    def update(self, superArm, reward):
        """
        Updates the model
        Parameters
        ----------
        superArm : list[]
            best superarm
        reward : list[]
            reward of the arm selected
        """
        for i in range(0, len(self.__GP_learner)):
            self.__GP_learner[i].update(superArm[i], reward[i])
