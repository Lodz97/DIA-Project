import numpy as np


class KnapsackSolver:

    def __init__(self, arms_dict):
        self.budgets = self.__set_budgets(arms_dict)
        self.sub_campaigns_matrix = self.__create_sub_campaigns_matrix(arms_dict)


    def __set_budgets(self, arms_dict):
        budget = [element for sublist in arms_dict for element in sublist.keys()]
        budget = np.unique(budget)                              # NB np.unique already returns the array sorted
        return budget

    def __create_sub_campaigns_matrix(self, arms_dict):
        matrix = []
        for dct in arms_dict:
            for b in self.budgets:
                if b not in dct.keys():
                    dct.update({b: -np.inf})
            matrix.append([dct[key] for key in sorted(dct.keys(), reverse=False)])

        return matrix
