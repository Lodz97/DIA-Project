from environment import AbstractClassEnvironment
import numpy as np


class BudgetEnvironment(AbstractClassEnvironment):
    """
    A class to represent the Gaussian Process between the number of click w.r.t budgets.
    The bid is fixed

    Attributes
    ----------
    __budget: list
        the possible budgets (arms)
    __sigma : int
        the noise of the process
    __means: list
        means of each budget value w.r.t number of clicks
    """
    def __init__(self, budget, sigma, func):
        """
        Parameters
        ----------
        func : method
            maps budget values to the corresponding expected number of clicks
        """
        self.__budget = budget
        self.__sigma = sigma
        self.__means = func(budget)

    def round(self, pulled_arm):
        """
        Parameters
        ----------
        pulled_arm : int
            the observed arm

        Returns
        ----------
        float
            a stochastic reward given by the expected number of clicks and the noise
            (the number of clicks is not deterministic)
        """
        return np.random.normal(self.means[pulled_arm], self.sigma[pulled_arm])

    @property
    def budget(self):
        return self.__budget

    @property
    def means(self):
        return self.__means
