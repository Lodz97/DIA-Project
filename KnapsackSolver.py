import numpy as np


class KnapsackSolver:

    def __init__(self, arms_dict):
        self.budgets = self.__set_budgets(arms_dict)


    def __set_budgets(self, arms_dict):
        budget = [element for sublist in arms_dict for element in sublist.keys()]
        budget = np.unique(budget)                              # NB np.unique already returns the array sorted
        return budget
