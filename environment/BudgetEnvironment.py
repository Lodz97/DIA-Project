from environment.AbstractClassEnvironment import AbstractClassEnvironment
from environment.ClickFunction import ClickFunction
import numpy as np


class BudgetEnvironment(AbstractClassEnvironment):
    """
    A class to represent the Gaussian Process between the number of click w.r.t budgets.
    The bid is fixed

    Attributes
    ----------
    __clicks_budget: dic
        {key = the possible budgets (arms) : value = means of each budget value w.r.t number of clicks
    __sigma : int
        the noise of the process
    """
    def __init__(self, budget, sigma, func):
        """
        :param func: method
            maps budget values to the corresponding expected number of clicks
        """
        AbstractClassEnvironment.__init__(self)
        self.__sigma = sigma
        tmp = func.apply_func(budget)
        self.clicks_budget = {budget[i]: tmp[i] for i in range(0, len(budget))}

    def round(self, pulled_arm):
        """
        :param pulled_arm: float
            the observed arm
        :return: float
            a stochastic reward given by the expected number of clicks and the noise
                (the number of clicks is not deterministic)
        """
        return np.random.normal(self.clicks_budget[pulled_arm], self.__sigma)

    @property
    def clicks_budget(self):
        return self.__clicks_budget

    @clicks_budget.setter
    def clicks_budget(self, value):
        self.__clicks_budget = value
