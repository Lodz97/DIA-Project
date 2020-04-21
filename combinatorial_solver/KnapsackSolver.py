import numpy as np


class KnapsackSolver:

    def __init__(self):
        self.budgets = []
        self.sub_campaigns_matrix = []

    def __set_budgets(self, arms_dict):
        budget = [element for sublist in arms_dict for element in sublist.keys()]
        self.budgets = np.unique(budget)                              # NB np.unique already returns the array sorted

    def __create_sub_campaigns_matrix(self, arms_dict):
        matrix = []
        for dct in arms_dict:
            for b in self.budgets:
                if b not in dct.keys():
                    dct.update({b: -np.inf})
            matrix.append([dct[key] for key in sorted(dct.keys(), reverse=False)])

        self.sub_campaigns_matrix = np.array(matrix)

    def solve(self, arms_dict):
        self.__set_budgets(arms_dict)
        self.__create_sub_campaigns_matrix(arms_dict)

        #matrix = np.empty((subcamp_matrix.shape[0] + 1, subcamp_matrix.shape[1]), dtype=Cell)
