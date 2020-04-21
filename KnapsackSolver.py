import numpy as np


class KnapsackSolver:

    def __init__(self):
        self.budgets = []
        self.sub_campaigns_matrix = []

    def __set_budgets(self, arms_dict):
        budget = [element for sublist in arms_dict for element in sublist.keys()]
        self.budgets = np.unique(budget)                              # NB np.unique already returns the array sorted

    def __create_sub_campaigns_matrix(self, arms_dict):
        self.sub_campaigns_matrix = []
        for dct in arms_dict:
            for b in self.budgets:
                if b not in dct.keys():
                    dct.update({b: -np.inf})
            self.sub_campaigns_matrix.append([dct[key] for key in sorted(dct.keys(), reverse=False)])

    def solve(self, arms_dict):
        self.__set_budgets(arms_dict)
        self.__create_sub_campaigns_matrix(arms_dict)