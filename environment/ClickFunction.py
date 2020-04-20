import matplotlib.pyplot as plt
import numpy as np


class ClickFunction:
    """
    Class which represents the function that maps a budget value to the corresponding expected number of clicks
    """
    def __init__(self, bound, slope):
        self.__bound = bound
        self.__slope = slope

    def apply_func(self, x):
        """
        :param x: float
            point to compute f(x)
        :return: f(x)
        """
        return (1 - np.exp(- self.__slope * x)) * self.__bound

    @staticmethod
    def visualize_func(budget, function):
        plt.figure()
        plt.plot(function(budget), 'r:', label='NumberClicks')
        plt.title("Clicks / Budget curve")
        plt.xlabel("Budget")
        plt.ylabel("Number of clicks")
        plt.legend()
        plt.show()
